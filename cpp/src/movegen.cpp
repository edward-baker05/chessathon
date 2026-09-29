#include "movegen.h"

namespace {

constexpr Bitboard RANK_2 = 0x000000000000FF00ull;
constexpr Bitboard RANK_7 = 0x00FF000000000000ull;

// Squares that must be empty to castle.
constexpr Bitboard WK_EMPTY = 0x0000000000000060ull;
constexpr Bitboard WQ_EMPTY = 0x000000000000000Eull;
constexpr Bitboard BK_EMPTY = 0x6000000000000000ull;
constexpr Bitboard BQ_EMPTY = 0x0E00000000000000ull;

// Promotion piece indices as stored in a move, 1..4.
constexpr const char* PROMO_PIECES = "nbrq";

// Moves of the piece on `from` to every square in `targets`.
inline Move* spread(Move* out, int from, Bitboard targets) {
    while (targets) *out++ = make_move(from, pop_lsb(targets));
    return out;
}

}  // namespace

int generate(const Position& pos, Move* moves) {
    Move* out = moves;
    int black = pos.stm;
    Bitboard us = pos.colours[black];
    Bitboard them = pos.colours[black ^ 1];
    Bitboard occ = us | them;
    Bitboard empty = ~occ;

    int forward = black ? -8 : 8;
    Bitboard start_rank = black ? RANK_7 : RANK_2;
    int last_low = black ? 0 : 56;
    int last_high = last_low + 8;

    Bitboard bits = pos.pieces[PAWN] & us;
    while (bits) {
        int from = pop_lsb(bits);
        int to = from + forward;
        if (to >= 0 && to < 64 && (empty & square_bit(to))) {
            if (to >= last_low && to < last_high) {
                for (int piece = 1; piece < 5; ++piece) *out++ = make_move(from, to, piece, FLAG_PROMO);
            } else {
                *out++ = make_move(from, to);
                int twice = from + 2 * forward;
                if ((square_bit(from) & start_rank) && (empty & square_bit(twice)))
                    *out++ = make_move(from, twice);
            }
        }
        Bitboard captures = PAWN_ATT[black][from] & them;
        while (captures) {
            int target = pop_lsb(captures);
            if (target >= last_low && target < last_high) {
                for (int piece = 1; piece < 5; ++piece)
                    *out++ = make_move(from, target, piece, FLAG_PROMO);
            } else {
                *out++ = make_move(from, target);
            }
        }
        if (pos.ep != NO_SQUARE && (PAWN_ATT[black][from] & square_bit(pos.ep)))
            *out++ = make_move(from, pos.ep, 0, FLAG_EP);
    }

    bits = pos.pieces[KNIGHT] & us;
    while (bits) {
        int from = pop_lsb(bits);
        out = spread(out, from, KNIGHT_ATT[from] & ~us);
    }
    bits = (pos.pieces[BISHOP] | pos.pieces[QUEEN]) & us;
    while (bits) {
        int from = pop_lsb(bits);
        out = spread(out, from, bishop_attacks(from, occ) & ~us);
    }
    bits = (pos.pieces[ROOK] | pos.pieces[QUEEN]) & us;
    while (bits) {
        int from = pop_lsb(bits);
        out = spread(out, from, rook_attacks(from, occ) & ~us);
    }
    int king = lsb(pos.pieces[KING] & us);
    out = spread(out, king, KING_ATT[king] & ~us);

    // Castling. The destination square is not checked here: the legality filter that runs
    // after make() catches a king landing in check.
    int rights = pos.castle;
    if (black) {
        if ((rights & BLACK_KINGSIDE) && !(occ & BK_EMPTY) && !attacked(pos, 60, WHITE) &&
            !attacked(pos, 61, WHITE))
            *out++ = make_move(60, 62, 0, FLAG_CASTLE);
        if ((rights & BLACK_QUEENSIDE) && !(occ & BQ_EMPTY) && !attacked(pos, 60, WHITE) &&
            !attacked(pos, 59, WHITE))
            *out++ = make_move(60, 58, 0, FLAG_CASTLE);
    } else {
        if ((rights & WHITE_KINGSIDE) && !(occ & WK_EMPTY) && !attacked(pos, 4, BLACK) &&
            !attacked(pos, 5, BLACK))
            *out++ = make_move(4, 6, 0, FLAG_CASTLE);
        if ((rights & WHITE_QUEENSIDE) && !(occ & WQ_EMPTY) && !attacked(pos, 4, BLACK) &&
            !attacked(pos, 3, BLACK))
            *out++ = make_move(4, 2, 0, FLAG_CASTLE);
    }
    return static_cast<int>(out - moves);
}

int generate_captures(const Position& pos, Move* moves) {
    Move* out = moves;
    int black = pos.stm;
    Bitboard us = pos.colours[black];
    Bitboard them = pos.colours[black ^ 1];
    Bitboard occ = us | them;
    Bitboard empty = ~occ;

    int forward = black ? -8 : 8;
    int last_low = black ? 0 : 56;
    int last_high = last_low + 8;

    Bitboard bits = pos.pieces[PAWN] & us;
    while (bits) {
        int from = pop_lsb(bits);
        int to = from + forward;
        // A quiet push is only worth searching here when it promotes.
        if (to >= 0 && to < 64 && to >= last_low && to < last_high && (empty & square_bit(to)))
            *out++ = make_move(from, to, QUEEN, FLAG_PROMO);
        Bitboard captures = PAWN_ATT[black][from] & them;
        while (captures) {
            int target = pop_lsb(captures);
            if (target >= last_low && target < last_high)
                *out++ = make_move(from, target, QUEEN, FLAG_PROMO);
            else
                *out++ = make_move(from, target);
        }
        if (pos.ep != NO_SQUARE && (PAWN_ATT[black][from] & square_bit(pos.ep)))
            *out++ = make_move(from, pos.ep, 0, FLAG_EP);
    }

    bits = pos.pieces[KNIGHT] & us;
    while (bits) {
        int from = pop_lsb(bits);
        out = spread(out, from, KNIGHT_ATT[from] & them);
    }
    bits = (pos.pieces[BISHOP] | pos.pieces[QUEEN]) & us;
    while (bits) {
        int from = pop_lsb(bits);
        out = spread(out, from, bishop_attacks(from, occ) & them);
    }
    bits = (pos.pieces[ROOK] | pos.pieces[QUEEN]) & us;
    while (bits) {
        int from = pop_lsb(bits);
        out = spread(out, from, rook_attacks(from, occ) & them);
    }
    int king = lsb(pos.pieces[KING] & us);
    out = spread(out, king, KING_ATT[king] & them);
    return static_cast<int>(out - moves);
}

int generate_legal(const Position& pos, Move* moves) {
    Move pseudo[MAX_MOVES];
    int count = generate(pos, pseudo);
    int legal = 0;
    Position after;
    for (int i = 0; i < count; ++i) {
        make(pos, after, pseudo[i]);
        if (legal_after(after, pos.stm)) moves[legal++] = pseudo[i];
    }
    return legal;
}

uint64_t perft(const Position& pos, int depth) {
    if (depth == 0) return 1;
    Move moves[MAX_MOVES];
    int count = generate(pos, moves);
    uint64_t total = 0;
    Position after;
    for (int i = 0; i < count; ++i) {
        make(pos, after, moves[i]);
        if (legal_after(after, pos.stm)) total += perft(after, depth - 1);
    }
    return total;
}

std::string move_to_uci(Move move) {
    if (move == 0) return "0000";
    std::string text = square_name(move_from(move)) + square_name(move_to(move));
    if (move_flag(move) == FLAG_PROMO) text += PROMO_PIECES[move_promo(move) - 1];
    return text;
}

Move parse_uci_move(const Position& pos, const std::string& text) {
    Move moves[MAX_MOVES];
    int count = generate_legal(pos, moves);
    for (int i = 0; i < count; ++i)
        if (move_to_uci(moves[i]) == text) return moves[i];
    return 0;
}
