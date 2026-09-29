#include "search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_set>

#include "movegen.h"
#include "nnue.h"

namespace search {

std::atomic<bool> stop_requested{false};
std::atomic<bool> pondering{false};

namespace {

constexpr int INF = 32000;

// A depth whose best move changed, or whose score fell by this much, is worth more time.
constexpr int INSTABILITY_DROP = 30;
constexpr double INSTABILITY_FACTOR = 1.5;

// The other side of the same coin: a best move that has survived several iterations is
// settled, so hand the unspent time to a later move that needs it.
constexpr double STABLE_STEP = 0.075;
constexpr double STABLE_FLOOR = 0.70;
// Contracting on two or three trivial early iterations says nothing.
constexpr int STABLE_MIN_DEPTH = 5;

// An aspiration re-search re-runs a whole depth, so a failed window is bounded by this
// multiple of the soft limit rather than by the hard limit.
constexpr double STRETCH_MULTIPLE = 2.0;

// How much longer the next iteration is expected to take than the last one, measured from
// the two most recent iterations and clamped so one anomalous depth cannot poison it.
constexpr double GROWTH_DEFAULT = 2.2;
constexpr double GROWTH_MIN = 1.6;
constexpr double GROWTH_MAX = 4.0;

// How often to read the clock, as a mask.
constexpr int64_t CLOCK_INTERVAL_MASK = 2047;

constexpr double RESERVE_MS = 300.0;
// Held back from a fixed `go movetime`, for the time it takes to answer.
constexpr double MOVETIME_OVERHEAD_MS = 10.0;

// The share of the remaining clock spent on one move, rising with the ply, so that a
// draining clock does not drag the allocation down with it (Stockfish's optScale).
constexpr double SOFT_BASE = 0.0084;
constexpr double SOFT_PLY_SCALE = 0.0042;
constexpr double SOFT_CAP_DIVISOR = 12.0;
constexpr double HARD_DIVISOR = 6.0;
constexpr double HARD_MULTIPLE = 3.0;

// Ordering scores. Captures and killers sit above every quiet move, and history fills the
// space below, so the bands can never cross.
constexpr int SCORE_TT = 1 << 24;
constexpr int SCORE_GOOD_CAPTURE = 1 << 22;
constexpr int SCORE_KILLER_1 = (1 << 21) + 2;
constexpr int SCORE_KILLER_2 = (1 << 21) + 1;
constexpr int SCORE_COUNTER = 1 << 21;
constexpr int SCORE_BAD_CAPTURE = -(1 << 22);
constexpr int HISTORY_MAX = 1 << 14;

constexpr int HISTORY_CAPACITY = 2048;

// Most valuable victim, least valuable attacker. Indexed [victim][attacker].
int MVV_LVA[6][6];
// Late move reductions, indexed [depth][move number].
int LMR_TABLE[64][64];

// Every mutable array the search touches.
struct Work {
    Position state[STACK_PLIES];
    nnue::Accumulator acc[STACK_PLIES];
    Move moves[STACK_PLIES][MAX_MOVES];
    int32_t scores[STACK_PLIES][MAX_MOVES];
    Move killers[STACK_PLIES][2];
    int32_t history[2][64][64];
    Move counter[2][64][64];
    int32_t static_evals[STACK_PLIES];
    Move played[STACK_PLIES];
    int8_t moved_piece[STACK_PLIES];
    // [distance][piece][to][piece][to]: how a reply fared after a given earlier move, one
    // and two plies back.
    int32_t cont_hist[2][6][64][6][64];
    Key hist_keys[HISTORY_CAPACITY];
    int hist_len;

    int64_t nodes;
    int64_t node_limit;
    bool abort;
    int age;
    int seldepth;
    // Consecutive completed iterations whose best move did not change.
    int stable;
    // The last iteration that finished.
    int completed_depth;
    int completed_score;
    Move completed_best;

    // When the search began, in seconds on the steady clock, for reporting.
    double began;
    // Limits in seconds from `clock_start`: when the search began, or the ponderhit.
    double soft, soft_span, hard, stretch;
    // How long the last two iterations took.
    double last_iter, prev_iter;
};

std::unique_ptr<Work> work_storage;
Work* W = nullptr;
std::unique_ptr<TranspositionTable> tt_storage;

// The root moves `go searchmoves` allows, or empty for all of them.
std::vector<Move> root_moves;

inline bool searchable(Move move) {
    return root_moves.empty() || std::find(root_moves.begin(), root_moves.end(), move) != root_moves.end();
}

double read_clock() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Where our clock started. Atomic because ponderhit() moves it from the UCI thread.
std::atomic<double> clock_start{0.0};

// Seconds on our clock, or zero while pondering, when the time is the opponent's.
inline double used() {
    if (pondering.load(std::memory_order_relaxed)) return 0.0;
    return read_clock() - clock_start.load(std::memory_order_relaxed);
}

// Abort the search when the hard limit, the node limit or a stop is reached. Reading the
// clock every node would cost more than the search; every 2048 nodes it is free.
inline void check_time() {
    if (W->nodes & CLOCK_INTERVAL_MASK) return;
    if (W->node_limit != 0 && W->nodes >= W->node_limit) {
        W->abort = true;
        return;
    }
    if (stop_requested.load(std::memory_order_relaxed) || used() >= W->hard) W->abort = true;
}

// A position seen before, either in this line or earlier in the real game. One repetition
// scores as a draw: waiting for a third inside the search loses far more than it gains.
bool is_repetition(int ply) {
    Key key = W->state[ply].key;
    int half = W->state[ply].halfmove;
    int back = 0;
    for (int node = ply - 2; node >= 0 && back < half; node -= 2, back += 2)
        if (W->state[node].key == key) return true;
    // The game history ends with the root and alternates side to move from there, so an
    // even ply matches index length-3 and an odd ply length-2. The root itself is already
    // covered by the loop above, through state[0].
    int length = W->hist_len;
    for (int index = length - 3 + (ply & 1); index >= 0 && (length - 1 - index) + ply <= half;
         index -= 2)
        if (W->hist_keys[index] == key) return true;
    return false;
}

int continuation_score(int ply, int piece, int to) {
    int total = 0;
    for (int distance = 0; distance < 2; ++distance) {
        int previous = ply - 1 - distance;
        if (previous < 0 || W->played[previous] == 0) continue;
        int prior_piece = W->moved_piece[previous];
        if (prior_piece < 0) continue;
        total += W->cont_hist[distance][prior_piece][move_to(W->played[previous])][piece][to];
    }
    return total;
}

// Assign an ordering score to each generated move. Never sorts the whole list.
void score_moves(int ply, int count, Move tt_move) {
    const Position& pos = W->state[ply];
    int black = pos.stm;
    Move previous = ply > 0 ? W->played[ply - 1] : 0;
    Move counter_move = previous ? W->counter[black][move_from(previous)][move_to(previous)] : 0;

    Move* moves = W->moves[ply];
    int32_t* scores = W->scores[ply];
    for (int i = 0; i < count; ++i) {
        Move move = moves[i];
        if (move == tt_move && tt_move != 0) {
            scores[i] = SCORE_TT;
            continue;
        }
        int from = move_from(move);
        int to = move_to(move);
        int victim = pos.mail[to];
        if (victim >= 0 || move_flag(move) == FLAG_EP) {
            int base = MVV_LVA[victim >= 0 ? victim : 0][pos.mail[from]];
            scores[i] = see(pos, move) >= 0 ? SCORE_GOOD_CAPTURE + base : SCORE_BAD_CAPTURE + base;
        } else if (move == W->killers[ply][0]) {
            scores[i] = SCORE_KILLER_1;
        } else if (move == W->killers[ply][1]) {
            scores[i] = SCORE_KILLER_2;
        } else if (move == counter_move && counter_move != 0) {
            scores[i] = SCORE_COUNTER;
        } else {
            scores[i] = W->history[black][from][to] + continuation_score(ply, pos.mail[from], to);
        }
    }
}

// Selection sort one step: swap the best remaining move into `index`. Cheaper than sorting
// the list, because a cutoff usually happens in the first few moves.
Move pick_move(int ply, int index, int count) {
    Move* moves = W->moves[ply];
    int32_t* scores = W->scores[ply];
    int best = index;
    for (int i = index + 1; i < count; ++i)
        if (scores[i] > scores[best]) best = i;
    if (best != index) {
        std::swap(moves[index], moves[best]);
        std::swap(scores[index], scores[best]);
    }
    return moves[index];
}

void update_continuation(int ply, int piece, int to, int bonus) {
    if (piece < 0) return;
    for (int distance = 0; distance < 2; ++distance) {
        int previous = ply - 1 - distance;
        if (previous < 0 || W->played[previous] == 0) continue;
        int prior_piece = W->moved_piece[previous];
        if (prior_piece < 0) continue;
        int32_t& entry = W->cont_hist[distance][prior_piece][move_to(W->played[previous])][piece][to];
        entry = std::clamp(entry + bonus, -HISTORY_MAX, HISTORY_MAX);
    }
}

// Reward the move that caused a cutoff and punish the moves tried before it.
void update_history(int ply, Move best_move, int depth, int tried) {
    const Position& pos = W->state[ply];
    int black = pos.stm;
    int bonus = std::min(depth * depth, 400);

    if (W->killers[ply][0] != best_move) {
        W->killers[ply][1] = W->killers[ply][0];
        W->killers[ply][0] = best_move;
    }

    int from = move_from(best_move);
    int to = move_to(best_move);
    W->history[black][from][to] += bonus;
    update_continuation(ply, pos.mail[from], to, bonus);
    if (W->history[black][from][to] > HISTORY_MAX) {
        for (auto& row : W->history[black])
            for (int32_t& entry : row) entry = static_cast<int32_t>(floor_div(entry, 2));
    }

    Move previous = ply > 0 ? W->played[ply - 1] : 0;
    if (previous) W->counter[black][move_from(previous)][move_to(previous)] = best_move;

    for (int i = 0; i < tried; ++i) {
        Move move = W->moves[ply][i];
        if (move == best_move) continue;
        W->history[black][move_from(move)][move_to(move)] -= bonus;
        update_continuation(ply, pos.mail[move_from(move)], move_to(move), -bonus);
    }
}

inline int evaluate(int ply) { return nnue::evaluate(W->acc[ply], W->state[ply]); }

// Search captures until the position is quiet, so the evaluation is not measured halfway
// through an exchange.
int qsearch(int ply, int alpha, int beta) {
    ++W->nodes;
    check_time();
    if (W->abort) return 0;
    if (ply >= STACK_PLIES - 2) return evaluate(ply);
    W->seldepth = std::max(W->seldepth, ply);

    const Position& pos = W->state[ply];
    bool checked = in_check(pos);
    int stand_pat, best, count;
    if (checked) {
        // Standing pat while in check would claim a score the side to move cannot hold,
        // so every evasion has to be searched.
        stand_pat = best = -INF;
        count = generate(pos, W->moves[ply]);
    } else {
        stand_pat = evaluate(ply);
        if (stand_pat >= beta) return stand_pat;
        alpha = std::max(alpha, stand_pat);
        best = stand_pat;
        count = generate_captures(pos, W->moves[ply]);
    }
    score_moves(ply, count, 0);

    int black = pos.stm;
    int legal = 0;
    for (int index = 0; index < count; ++index) {
        Move move = pick_move(ply, index, count);
        if (!checked) {
            // A capture that loses material cannot rescue a position this far behind.
            int victim = pos.mail[move_to(move)];
            int gain = victim >= 0 ? SEE_VALUE[victim] : 100;
            if (stand_pat + gain + 200 < alpha) continue;
            if (see(pos, move) < 0) continue;
        }
        make(pos, W->state[ply + 1], move);
        if (!legal_after(W->state[ply + 1], black)) continue;
        nnue::apply(W->acc[ply], W->acc[ply + 1], pos, move);
        ++legal;
        W->played[ply] = move;
        W->moved_piece[ply] = pos.mail[move_from(move)];
        int value = -qsearch(ply + 1, -beta, -alpha);
        if (W->abort) return 0;
        if (value > best) {
            best = value;
            if (value > alpha) {
                alpha = value;
                if (alpha >= beta) break;
            }
        }
    }
    if (checked && legal == 0) return -MATE + ply;
    return best;
}

// Principal variation search.
int negamax(int ply, int depth, int alpha, int beta, bool is_pv, bool can_null = true) {
    ++W->nodes;
    check_time();
    if (W->abort) return 0;
    if (ply >= STACK_PLIES - 4) return evaluate(ply);

    const Position& pos = W->state[ply];
    bool checked = in_check(pos);

    if (ply > 0) {
        if (pos.halfmove >= 100 || insufficient_material(pos) || is_repetition(ply)) return 0;
        // Mate distance pruning: a mate found elsewhere is already nearer than anything
        // this subtree can produce.
        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    if (checked) ++depth;
    if (depth <= 0) return qsearch(ply, alpha, beta);

    Key key = pos.key;
    TTProbe entry = tt_storage->probe(key, ply);
    if (entry.hit && !is_pv && entry.depth >= depth) {
        if (entry.bound == BOUND_EXACT) return entry.score;
        if (entry.bound == BOUND_LOWER && entry.score >= beta) return entry.score;
        if (entry.bound == BOUND_UPPER && entry.score <= alpha) return entry.score;
    }

    int static_eval = entry.hit && entry.static_eval != 0 ? entry.static_eval : evaluate(ply);
    W->static_evals[ply] = static_eval;

    int black = pos.stm;
    bool prunable = !is_pv && !checked && beta < MATE_IN_MAX && beta > -MATE_IN_MAX;

    if (prunable) {
        // Reverse futility: so far ahead that giving back a margin per remaining ply still
        // beats beta, so the opponent would have avoided this line.
        if (depth <= 8 && static_eval - 75 * depth >= beta) return static_eval;

        // Razoring: so far behind that only a capture sequence could rescue it.
        if (depth <= 3 && static_eval + 200 * depth < alpha) {
            int razor = qsearch(ply, alpha, beta);
            if (razor <= alpha) return razor;
        }

        // Null move. Skipping a turn and still failing high means the real move will too.
        // Not tried without a piece: a side with only pawns can be in zugzwang.
        if (can_null && depth >= 3 && static_eval >= beta && has_non_pawn_material(pos, black)) {
            int reduction = 3 + depth / 4 + std::min((static_eval - beta) / 200, 3);
            make_null(pos, W->state[ply + 1]);
            W->acc[ply + 1] = W->acc[ply];
            W->played[ply] = 0;
            W->moved_piece[ply] = -1;
            int null_value = -negamax(ply + 1, depth - reduction - 1, -beta, -beta + 1, false, false);
            if (W->abort) return 0;
            if (null_value >= beta) {
                // A mate score proved by passing is not a real mate.
                if (null_value >= MATE_IN_MAX) null_value = beta;
                if (depth < 10) return null_value;
                // Deep enough that zugzwang is worth ruling out explicitly.
                int verify = negamax(ply, depth - reduction - 1, beta - 1, beta, false, false);
                if (W->abort) return 0;
                if (verify >= beta) return null_value;
            }
        }
    }

    int count = generate(pos, W->moves[ply]);
    score_moves(ply, count, entry.hit ? entry.move : 0);

    int best = -INF;
    Move best_move = 0;
    int original_alpha = alpha;
    int legal = 0;

    // Internal iterative reduction: with no TT move the ordering is poor, so a full-depth
    // search here is mostly wasted.
    if (depth >= 4 && !(entry.hit && entry.move != 0)) --depth;

    int quiets_tried = 0;
    bool improving = ply < 2 || static_eval > W->static_evals[ply - 2];

    for (int index = 0; index < count; ++index) {
        Move move = pick_move(ply, index, count);
        int to = move_to(move);
        bool is_capture = pos.mail[to] >= 0 || move_flag(move) != FLAG_NORMAL;
        int move_score = W->scores[ply][index];

        if (!is_pv && !checked && legal > 0 && best > -MATE_IN_MAX) {
            if (!is_capture) {
                // Late move pruning: this far down a well-ordered list, at low depth, a
                // quiet move is not going to be the best one.
                int cap = 3 + depth * depth;
                if (!improving) cap /= 2;
                if (depth <= 8 && quiets_tried >= cap) continue;
                // Futility: too far below alpha for a quiet move to close the gap.
                if (depth <= 6 && static_eval + 100 + 90 * depth <= alpha) continue;
                if (depth <= 8 && see(pos, move) < -50 * depth) continue;
            } else if (depth <= 8 && see(pos, move) < -100 * depth) {
                continue;
            }
        }

        make(pos, W->state[ply + 1], move);
        if (!legal_after(W->state[ply + 1], black)) continue;
        nnue::apply(W->acc[ply], W->acc[ply + 1], pos, move);
        ++legal;
        W->played[ply] = move;
        W->moved_piece[ply] = pos.mail[move_from(move)];
        if (!is_capture) ++quiets_tried;

        int value;
        if (legal == 1) {
            value = -negamax(ply + 1, depth - 1, -beta, -alpha, is_pv);
        } else {
            int reduction = 0;
            if (depth >= 3 && legal >= 3 && !is_capture) {
                reduction = LMR_TABLE[std::min(depth, 63)][std::min(legal, 63)];
                if (is_pv) --reduction;
                if (move_score >= SCORE_COUNTER) --reduction;
                if (!improving) ++reduction;
                reduction = std::clamp(reduction, 0, std::max(depth - 2, 0));
            }
            value = -negamax(ply + 1, depth - 1 - reduction, -alpha - 1, -alpha, false);
            // A reduced search that beat alpha proves nothing until it is repeated at full
            // depth. Skipping this is how an engine looks fine in tests and plays badly.
            if (reduction > 0 && value > alpha)
                value = -negamax(ply + 1, depth - 1, -alpha - 1, -alpha, false);
            if (alpha < value && value < beta)
                value = -negamax(ply + 1, depth - 1, -beta, -alpha, is_pv);
        }
        if (W->abort) return 0;

        if (value > best) {
            best = value;
            best_move = move;
            if (value > alpha) {
                alpha = value;
                if (alpha >= beta) {
                    if (pos.mail[to] < 0) update_history(ply, move, depth, index + 1);
                    break;
                }
            }
        }
    }

    if (legal == 0) return checked ? -MATE + ply : 0;

    int bound = best <= original_alpha ? BOUND_UPPER : best >= beta ? BOUND_LOWER : BOUND_EXACT;
    tt_storage->store(key, ply, best, best_move, depth, bound, static_eval, W->age);
    return best;
}

// Iterative deepening with aspiration windows. Returns the best move found.
Move search_root(int max_depth, const std::function<void(const Report&)>& on_iteration) {
    Move best_move = 0;
    int best_score = 0;
    Move previous_move = 0;
    int previous_score = 0;
    const Position& pos = W->state[0];
    int black = pos.stm;
    Move* moves = W->moves[0];

    // `improving` at ply 2 compares against the root, which negamax never writes.
    W->static_evals[0] = evaluate(0);

    int count = generate(pos, moves);
    // Establish a legal move before anything is allowed to abort.
    for (int index = 0; index < count; ++index) {
        if (!searchable(moves[index])) continue;
        make(pos, W->state[1], moves[index]);
        if (legal_after(W->state[1], black)) {
            best_move = moves[index];
            break;
        }
    }
    if (best_move == 0) return 0;

    double growth = GROWTH_DEFAULT;
    for (int depth = 1; depth <= max_depth; ++depth) {
        double iteration_start = read_clock();
        int window = 18;
        // A failed window re-runs the whole depth. The first failure widens by four, the
        // second goes straight to full width, so a depth costs at most three passes.
        int widenings = 0;
        bool abandoned = false;
        int alpha = -INF, beta = INF;
        if (depth >= 5) {
            alpha = std::max(best_score - window, -INF);
            beta = std::min(best_score + window, INF);
        }

        while (true) {
            int score = -INF;
            Move iteration_move = 0;
            count = generate(pos, moves);
            score_moves(0, count, best_move);
            int legal = 0;
            int local_alpha = alpha;
            for (int index = 0; index < count; ++index) {
                Move move = pick_move(0, index, count);
                if (!searchable(move)) continue;
                make(pos, W->state[1], move);
                if (!legal_after(W->state[1], black)) continue;
                nnue::apply(W->acc[0], W->acc[1], pos, move);
                ++legal;
                W->played[0] = move;
                // Continuation history one and two plies down reads this. Left unwritten it
                // held zero, a pawn, for every root move.
                W->moved_piece[0] = static_cast<int8_t>(pos.mail[move_from(move)]);
                int value;
                if (legal == 1) {
                    value = -negamax(1, depth - 1, -beta, -local_alpha, true);
                } else {
                    value = -negamax(1, depth - 1, -local_alpha - 1, -local_alpha, false);
                    if (local_alpha < value && value < beta)
                        value = -negamax(1, depth - 1, -beta, -local_alpha, true);
                }
                if (W->abort) break;
                // An iteration whose cost was underestimated is otherwise stopped only by
                // the hard limit. The first root move is the previous best, so once one
                // has finished there is always a move to fall back on.
                if (used() >= W->stretch) {
                    W->abort = true;
                    break;
                }
                if (value > score) {
                    score = value;
                    iteration_move = move;
                    local_alpha = std::max(local_alpha, value);
                }
            }

            if (W->abort) break;
            if (score <= alpha) {
                // Failed low: every root score is an upper bound, so the partial result
                // must not be committed; abandoning keeps the last depth's move.
                if (used() >= W->stretch) {
                    abandoned = true;
                    break;
                }
                if (++widenings >= 2) {
                    alpha = -INF;
                } else {
                    window *= 4;
                    alpha = std::max(score - window, -INF);
                }
                continue;
            }
            if (score >= beta) {
                if (used() >= W->stretch) {
                    abandoned = true;
                    break;
                }
                if (++widenings >= 2) {
                    beta = INF;
                } else {
                    window *= 4;
                    beta = std::min(score + window, INF);
                }
                continue;
            }
            best_score = score;
            if (iteration_move != 0) best_move = iteration_move;
            break;
        }

        if (W->abort || abandoned) break;
        W->completed_depth = depth;
        W->completed_score = best_score;
        W->completed_best = best_move;
        if (on_iteration) on_iteration(last_report());
        // A forced mate is found; searching deeper cannot improve on it.
        if (best_score >= MATE_IN_MAX || best_score <= -MATE_IN_MAX) break;

        W->prev_iter = W->last_iter;
        W->last_iter = read_clock() - iteration_start;
        double now = used();

        // Stability, in both directions. A changed best move or a falling score means the
        // position is not settled and is worth more time; a move that has survived several
        // iterations is settled. The soft limit is derived from the base span each
        // iteration, because a limit that is only ever written upwards cannot contract.
        bool unsettled =
            depth >= 4 && (best_move != previous_move || best_score < previous_score - INSTABILITY_DROP);
        W->stable = !unsettled && depth >= STABLE_MIN_DEPTH ? W->stable + 1 : 0;
        double scale = unsettled ? INSTABILITY_FACTOR
                                 : std::max(1.0 - STABLE_STEP * W->stable, STABLE_FLOOR);
        double soft = std::min(W->soft_span * scale, W->hard);
        W->soft = soft;
        // Aim the prediction below at the stretch rather than the soft limit: iterations
        // grow by two to four times, so requiring the next to finish inside the soft limit
        // would throw most of the budget away.
        double stretch = std::min(W->soft_span * scale * STRETCH_MULTIPLE, W->hard);
        W->stretch = stretch;

        previous_move = best_move;
        previous_score = best_score;

        if (W->prev_iter > 0.0) growth = std::clamp(W->last_iter / W->prev_iter, GROWTH_MIN, GROWTH_MAX);
        // On the opponent's time there is nothing to save: keep going until a ponderhit
        // starts the clock or a stop ends the search.
        if (pondering.load(std::memory_order_relaxed)) continue;
        if (now >= soft) break;
        // Do not start an iteration that cannot finish.
        if (now + W->last_iter * growth >= stretch) break;
    }
    return best_move;
}

// Soft and hard limits in milliseconds. Floored rather than allowed to go negative: late in
// a long game play is increment-only, and a negative budget would return no move at all.
std::pair<double, double> budget_ms(int64_t time_left_ms, int64_t increment_ms, int ply) {
    double usable = std::max(static_cast<double>(time_left_ms) - RESERVE_MS, 10.0);
    double fraction = SOFT_BASE + std::sqrt(static_cast<double>(ply) + 3.0) * SOFT_PLY_SCALE;
    // Only half the increment is credited. Over-crediting it is how engines flag.
    double soft = std::min(usable * fraction + 0.5 * static_cast<double>(increment_ms),
                           usable / SOFT_CAP_DIVISOR);
    double hard = std::min(usable / HARD_DIVISOR, soft * HARD_MULTIPLE);
    return {std::min(soft, hard), hard};
}

// Plies played, from the FEN alone. Clamped because a hand-written FEN can say anything.
int ply_of(const Position& pos) {
    return std::clamp((pos.fullmove - 1) * 2 + pos.stm, 0, 400);
}

void prepare(const Position& root, const Limits& limits) {
    W->state[0] = root;
    // The only full rebuild. Every ply below this is reached incrementally.
    nnue::refresh(W->acc[0], root);
    W->nodes = 0;
    W->abort = false;
    W->seldepth = 0;
    W->node_limit = limits.node_limit;
    W->age = (W->age + 1) & AGE_MASK;
    W->stable = 0;
    W->completed_depth = 0;
    W->completed_score = 0;
    W->completed_best = 0;

    double now = read_clock();
    double infinity = std::numeric_limits<double>::infinity();
    double soft_ms = infinity, hard_ms = infinity;
    if (limits.movetime_ms > 0) {
        soft_ms = hard_ms = std::max(static_cast<double>(limits.movetime_ms) - MOVETIME_OVERHEAD_MS, 1.0);
    } else if (limits.has_clock) {
        std::tie(soft_ms, hard_ms) = budget_ms(limits.time_left_ms, limits.increment_ms, ply_of(root));
    }
    W->began = now;
    clock_start = now;
    W->soft_span = soft_ms / 1000.0;
    W->soft = soft_ms / 1000.0;
    W->hard = hard_ms / 1000.0;
    W->stretch = std::min(soft_ms * STRETCH_MULTIPLE / 1000.0, W->hard);
    W->last_iter = 0.0;
    W->prev_iter = 0.0;
}

}  // namespace

void init() {
    for (int victim = 0; victim < 6; ++victim)
        for (int attacker = 0; attacker < 6; ++attacker)
            MVV_LVA[victim][attacker] = 100 * (victim + 1) - attacker;
    // Later moves in a well-ordered list are progressively less likely to be best, so they
    // are searched shallower first and re-searched only if they beat alpha.
    for (int depth = 1; depth < 64; ++depth) {
        for (int index = 1; index < 64; ++index) {
            int reduction = static_cast<int>(0.75 + std::log(depth) * std::log(index) / 2.25);
            // Never reduce below one ply of real search.
            LMR_TABLE[depth][index] = std::max(0, std::min(reduction, depth - 1));
        }
    }
    work_storage = std::make_unique<Work>();
    W = work_storage.get();
    tt_storage = std::make_unique<TranspositionTable>();
}

TranspositionTable& table() { return *tt_storage; }

void ponderhit() {
    clock_start = read_clock();
    pondering = false;
}

void set_game_history(const std::vector<Key>& keys) {
    // The most recent positions, if a game ever outgrows the buffer: the root has to be last.
    size_t length = std::min(keys.size(), static_cast<size_t>(HISTORY_CAPACITY));
    std::copy(keys.end() - static_cast<std::ptrdiff_t>(length), keys.end(), W->hist_keys);
    W->hist_len = static_cast<int>(length);
}

void clear_tables() {
    std::fill(&W->history[0][0][0], &W->history[0][0][0] + 2 * 64 * 64, 0);
    std::fill(&W->counter[0][0][0], &W->counter[0][0][0] + 2 * 64 * 64, 0);
    std::fill(&W->killers[0][0], &W->killers[0][0] + STACK_PLIES * 2, 0);
    std::fill(W->played, W->played + STACK_PLIES, 0);
    std::fill(W->moved_piece, W->moved_piece + STACK_PLIES, 0);
    std::fill(&W->cont_hist[0][0][0][0][0], &W->cont_hist[0][0][0][0][0] + 2 * 6 * 64 * 6 * 64, 0);
}

Move think(const Position& root, const Limits& limits,
           const std::function<void(const Report&)>& on_iteration) {
    prepare(root, limits);
    // Moves that are not legal here are dropped, and a list with nothing legal left in it
    // restricts nothing: searching every move beats answering with no move at all.
    root_moves.clear();
    Move legal[MAX_MOVES];
    int count = generate_legal(root, legal);
    for (Move move : limits.searchmoves)
        if (std::find(legal, legal + count, move) != legal + count) root_moves.push_back(move);
    return search_root(std::clamp(limits.max_depth, 1, MAX_DEPTH), on_iteration);
}

Report last_report() {
    auto elapsed = static_cast<int64_t>((read_clock() - W->began) * 1000.0);
    return {W->completed_best, W->completed_depth, W->completed_score,
            std::max(W->seldepth, W->completed_depth), W->nodes, std::max<int64_t>(elapsed, 1)};
}

std::vector<Move> principal_variation(const Position& root) {
    // There is no PV array: the line is walked out of the table, one probe per ply. A later
    // store can have replaced a step, so the walk stops at the first position the table has
    // no legal move for, or at a repeated position, and never runs past the depth searched.
    std::vector<Move> line;
    Move first = W->completed_best;
    if (first == 0) return line;
    Position walk = root, next;
    make(walk, next, first);
    walk = next;
    line.push_back(first);
    std::unordered_set<Key> seen;
    while (static_cast<int>(line.size()) < W->completed_depth) {
        if (!seen.insert(walk.key).second) break;
        TTProbe entry = tt_storage->probe(walk.key, 0);
        if (!entry.hit || entry.move == 0) break;
        Move move = parse_uci_move(walk, move_to_uci(entry.move));
        if (move == 0) break;
        make(walk, next, move);
        walk = next;
        line.push_back(move);
    }
    return line;
}

std::string uci_info(const Position& root, const Report& report) {
    std::string score;
    if (report.score >= MATE_IN_MAX)
        score = "mate " + std::to_string((MATE - report.score + 1) / 2);
    else if (report.score <= -MATE_IN_MAX)
        score = "mate " + std::to_string(-((MATE + report.score) / 2));
    else
        score = "cp " + std::to_string(report.score);
    std::string line = "info depth " + std::to_string(report.depth) + " seldepth " +
                       std::to_string(report.seldepth) + " score " + score + " nodes " +
                       std::to_string(report.nodes) + " nps " +
                       std::to_string(report.nodes * 1000 / report.elapsed_ms) + " time " +
                       std::to_string(report.elapsed_ms) + " pv";
    for (Move move : principal_variation(root)) line += " " + move_to_uci(move);
    return line;
}

}  // namespace search
