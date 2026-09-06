"""Search invariants. These are what tell a broken heuristic from a merely weak one."""

import os
import time

import chess
import numpy as np
import pytest

import nnue
import position
import search
import tt
from bitboard import HALF, KEY
from evaluate import evaluate
from tests.conftest import random_positions

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"  # Ra8 is mate
MATE_IN_ONE_CAPTURE = "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"  # Rxd8 is mate
FREE_QUEEN = "4k3/8/8/3q4/4B3/8/8/4K3 w - - 0 1"  # Bxd5


def best_move(fen: str, depth: int = 6, nodes: int = 0) -> str:
    return search.think(chess.Board(fen), time_left_ms=600_000, node_limit=nodes, max_depth=depth)


# --------------------------------------------------------------- basic behaviour


def test_returns_a_legal_move_from_many_positions() -> None:
    for board in random_positions(count=40, seed=71):
        uci = search.think(board, time_left_ms=2_000, max_depth=4)
        assert chess.Move.from_uci(uci) in board.legal_moves, board.fen()


def test_finds_mate_in_one() -> None:
    assert best_move(MATE_IN_ONE, depth=3) == "a1a8"


def test_finds_mate_in_one_by_capture() -> None:
    assert best_move(MATE_IN_ONE_CAPTURE, depth=3) == "d1d8"


def test_mate_score_is_ply_adjusted() -> None:
    """The bookkeeping that goes wrong silently. A mate in one scores MATE - 1."""
    assert search.search_value(chess.Board(MATE_IN_ONE), depth=3) == search.MATE - 1


def test_being_mated_scores_negative_mate() -> None:
    mated = chess.Board(MATE_IN_ONE)
    mated.push_uci("a1a8")
    assert search.search_value(mated, depth=3) == -search.MATE


def test_stalemate_scores_zero() -> None:
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stalemate.is_stalemate()
    assert search.search_value(stalemate, depth=3) == 0


def test_takes_a_free_queen() -> None:
    assert best_move(FREE_QUEEN, depth=4) == "e4d5"


def test_does_not_hang_a_queen_to_a_pawn() -> None:
    # The pawn on c6 attacks b5 and d5 but not d4, so the queen is safe where it stands
    # and d4d5 would simply hang it.
    assert best_move("4k3/8/2p5/8/3Q4/8/8/4K3 w - - 0 1", depth=6) != "d4d5"


def test_quiescence_does_not_stand_pat_in_check() -> None:
    """Standing pat while in check claims a score the side to move cannot hold."""
    # Black is checkmated; a quiescence that stood pat would report material instead.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    board.push_uci("a1a8")
    assert search.search_value(board, depth=0) == -search.MATE


# ---------------------------------------------------------------------- the oracle
#
# Every heuristic in the search reshapes the tree but must not change the value of a
# full-width search. With pruning disabled the search must agree exactly with an
# independent, obviously correct implementation written against python-chess.


def reference_qsearch(board: chess.Board, alpha: int, beta: int, ply: int) -> int:
    """Deliberately naive quiescence, mirroring search.qsearch's contract."""
    checked = board.is_check()
    if checked:
        best = -search.INF
        moves = list(board.legal_moves)
    else:
        stand_pat = reference_evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        best = stand_pat
        moves = [
            m
            for m in board.legal_moves
            if board.is_capture(m) or m.promotion == chess.QUEEN
        ]

    legal = 0
    for move in moves:
        board.push(move)
        legal += 1
        value = -reference_qsearch(board, -beta, -alpha, ply + 1)
        board.pop()
        if value > best:
            best = value
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    if checked and legal == 0:
        return -search.MATE + ply
    return best


def reference_evaluate(board: chess.Board) -> int:
    """The network, rebuilt from the board rather than carried incrementally.

    That makes the oracle below a differential test of the accumulator as well as of the
    search: the engine reaches every node by updating the accumulator one move at a time,
    and this reference reaches the same node by rebuilding it from scratch. If they ever
    disagree the oracle fails, which is the cheapest possible place to catch drift.
    """
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    acc = nnue.new_accumulator(2)
    nnue.refresh(acc, 0, state[0], mailbox[0])
    return int(nnue.forward(acc, 0, state[0]))


def reference_alpha_beta(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    """Slow, obviously correct, and the only thing we trust."""
    if ply > 0 and (
        board.halfmove_clock >= 100
        or board.is_insufficient_material()
        or board.is_repetition(2)
    ):
        return 0
    if depth <= 0:
        return reference_qsearch(board, alpha, beta, ply)

    best = -search.INF
    legal = 0
    for move in board.legal_moves:
        board.push(move)
        legal += 1
        value = -reference_alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if value > best:
            best = value
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    if legal == 0:
        return -search.MATE + ply if board.is_check() else 0
    return best


def assert_matches_oracle(seed: int, depth: int, count: int, max_plies: int = 10) -> None:
    search.set_pruning(False)
    try:
        for board in random_positions(count=count, seed=seed, max_plies=max_plies):
            got = search.search_value(board, depth=depth)
            want = reference_alpha_beta(board, depth, -search.INF, search.INF, 0)
            assert got == want, f"{got} != {want} at depth {depth} in {board.fen()}"
    finally:
        search.set_pruning(True)


def test_pruning_disabled_search_equals_plain_alpha_beta() -> None:
    """Breadth at shallow depth. Every part of the node algorithm is still exercised:
    quiescence, mate and stalemate scoring, draw detection, ordering and the TT."""
    assert_matches_oracle(seed=72, depth=2, count=20)


def test_oracle_holds_at_depth_three() -> None:
    assert_matches_oracle(seed=73, depth=3, count=3)


@pytest.mark.skipif(
    os.environ.get("CHESSATHON_SLOW_TESTS") != "1",
    reason="the python-chess reference quiescence is slow; set CHESSATHON_SLOW_TESTS=1",
)
def test_oracle_holds_deeply() -> None:
    """The thorough version. Minutes, not seconds, so it is opt-in rather than default."""
    assert_matches_oracle(seed=74, depth=4, count=8, max_plies=16)


# ---------------------------------------------------------------------- pruning


def test_null_move_guard_rejects_a_pawns_only_side() -> None:
    """Null move assumes passing is never better than moving. A side with only pawns can
    be in zugzwang, where that assumption is false, so the guard has to refuse it."""
    state, mailbox = position.new_stacks()
    position.encode(chess.Board("8/8/8/8/1k6/8/1P6/1K6 w - - 0 1"), state[0], mailbox[0])
    assert not position.has_non_pawn_material(state[0], 0)
    assert not position.has_non_pawn_material(state[0], 1)

    position.encode(chess.Board("8/8/8/8/1k6/8/1P6/1KR5 w - - 0 1"), state[0], mailbox[0])
    assert position.has_non_pawn_material(state[0], 0), "a rook makes null move safe again"


def test_pawn_endgame_is_searched_without_crashing() -> None:
    fen = "8/8/8/8/1k6/8/1P6/1K6 w - - 0 1"
    uci = best_move(fen, depth=12)
    assert chess.Move.from_uci(uci) in chess.Board(fen).legal_moves


def test_lmr_table_is_monotonic() -> None:
    for depth in range(4, 64):
        for index in range(4, 64):
            assert search.LMR_TABLE[depth][index] >= search.LMR_TABLE[depth][index - 1], (
                "later moves should be reduced at least as much as earlier ones"
            )


def test_reductions_never_reduce_below_one_ply() -> None:
    for depth in range(1, 64):
        for index in range(64):
            assert depth - search.LMR_TABLE[depth][index] >= 1


def test_first_moves_are_not_reduced() -> None:
    for depth in range(1, 64):
        assert search.LMR_TABLE[depth][0] == 0
        assert search.LMR_TABLE[depth][1] == 0


def test_think_on_a_terminal_position_fails_loudly() -> None:
    """The referee ends a game before asking, so this is about a legible failure, not play."""
    stalemate = chess.Board("8/8/8/8/8/6k1/6p1/6K1 w - - 0 1")
    assert stalemate.is_stalemate()
    with pytest.raises(ValueError, match="no legal move"):
        search.think(stalemate, time_left_ms=1_000, max_depth=2)


def test_pruning_does_not_break_a_won_pawn_endgame() -> None:
    # White queens by force. A search that pruned the winning line would shuffle instead.
    assert search.search_value(chess.Board("8/8/8/8/8/1k6/1P6/1K6 w - - 0 1"), depth=8) >= 0


def test_pruning_keeps_finding_the_mate() -> None:
    assert best_move(MATE_IN_ONE, depth=10) == "a1a8"
    assert search.search_value(chess.Board(MATE_IN_ONE), depth=10) == search.MATE - 1


# ------------------------------------------------------------------ determinism


def test_node_limit_is_respected() -> None:
    search.think(chess.Board(), time_left_ms=600_000, node_limit=50_000, max_depth=127)
    assert search.nodes() <= 55_000, "node limit should stop the search promptly"


def test_node_limited_search_is_deterministic() -> None:
    # The tables are cleared between the two searches because the engine deliberately
    # keeps its transposition table across moves within a game, so a second search from
    # the same position legitimately sees a warmer table and can pick a different move
    # among equals. What is being asserted here is that the search itself has no
    # nondeterminism, not that it ignores what it already learned.
    tt.tt_clear(tt.TT)
    search.clear_tables()
    first = best_move(chess.STARTING_FEN, depth=127, nodes=100_000)
    tt.tt_clear(tt.TT)
    search.clear_tables()
    second = best_move(chess.STARTING_FEN, depth=127, nodes=100_000)
    assert first == second


def test_respects_a_tight_clock() -> None:
    search.think(chess.Board(), time_left_ms=2_000)
    started = time.perf_counter()
    search.think(chess.Board(), time_left_ms=1_000, max_depth=127)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.9, f"took {elapsed:.2f}s of a 1.0s clock, which risks a flag"


def test_deeper_search_never_loses_the_free_queen() -> None:
    for depth in range(4, 10):
        assert best_move(FREE_QUEEN, depth=depth) == "e4d5", (
            f"lost a free queen at depth {depth}"
        )


# ------------------------------------------------------- continuation history


def test_continuation_history_is_populated_by_a_search() -> None:
    # The transposition table has to be cleared too. It deliberately survives between
    # moves, so a warm table from an earlier test lets the search finish on TT hits
    # alone and record no history at all.
    tt.tt_clear(tt.TT)
    search.clear_tables()
    search.think(chess.Board(), time_left_ms=5_000, max_depth=8)
    assert search.WORK.cont_hist.any(), "a search should leave continuation history behind"


def test_clear_tables_resets_every_history() -> None:
    search.think(chess.Board(), time_left_ms=3_000, max_depth=6)
    search.clear_tables()
    assert not search.WORK.cont_hist.any()
    assert not search.WORK.history.any()
    assert not search.WORK.counter.any()
    assert not search.WORK.killers.any()


def test_continuation_history_stays_within_bounds() -> None:
    """Unbounded history overflows its int32 and inverts the ordering it is meant to fix."""
    tt.tt_clear(tt.TT)
    search.clear_tables()
    search.think(chess.Board(), time_left_ms=5_000, max_depth=10)
    assert abs(int(search.WORK.cont_hist.min())) <= search.HISTORY_MAX
    assert int(search.WORK.cont_hist.max()) <= search.HISTORY_MAX


def test_oracle_holds_with_continuation_history() -> None:
    assert_matches_oracle(seed=111, depth=3, count=3)


def test_game_history_repetition_is_found_from_both_plies() -> None:
    """`agent.py` records the game as [us, them, us, ..., us], so a position repeats one
    two of its own moves back at an even ply and one of the opponent's at an odd ply. A
    scan that starts from a fixed index sees only one of those parities, and the one it
    misses is our own turn, which is the shape a threefold actually takes."""
    work = search.new_work(tt.new_table())
    repeated = np.uint64(0xDEADBEEFCAFE1234)

    # Seven entries: indices 0, 2, 4 and the root at 6 are ours; 1, 3, 5 are theirs.
    for index in range(7):
        work.hist_keys[index] = np.uint64(index + 1)
    work.ints[search.I_HIST_LEN] = 7

    for ply, index in ((2, 4), (3, 5)):
        for slot in range(7):
            work.hist_keys[slot] = np.uint64(slot + 1)
        work.hist_keys[index] = repeated
        work.state[:] = 0
        work.state[ply][KEY] = repeated
        work.state[ply][HALF] = np.uint64(40)
        for node in range(ply):
            work.state[node][KEY] = np.uint64(0xAAAA + node)
        assert search.is_repetition(work, ply), (
            f"a repetition of history entry {index} was missed at ply {ply}"
        )


def test_an_unrepeated_history_is_not_called_a_repetition() -> None:
    work = search.new_work(tt.new_table())
    for index in range(7):
        work.hist_keys[index] = np.uint64(index + 1)
    work.ints[search.I_HIST_LEN] = 7
    for ply in (2, 3):
        work.state[:] = 0
        work.state[ply][KEY] = np.uint64(0x123456789)
        work.state[ply][HALF] = np.uint64(40)
        for node in range(ply):
            work.state[node][KEY] = np.uint64(0xAAAA + node)
        assert not search.is_repetition(work, ply)


def test_the_halfmove_clock_stops_the_history_scan() -> None:
    """An irreversible move makes everything before it unreachable, so a low halfmove
    clock has to stop the walk before it reaches a position that cannot recur."""
    work = search.new_work(tt.new_table())
    repeated = np.uint64(0xFEEDFACE)
    for index in range(7):
        work.hist_keys[index] = np.uint64(index + 1)
    work.hist_keys[4] = repeated
    work.ints[search.I_HIST_LEN] = 7
    work.state[:] = 0
    work.state[2][KEY] = repeated
    for node in range(2):
        work.state[node][KEY] = np.uint64(0xAAAA + node)

    work.state[2][HALF] = np.uint64(40)
    assert search.is_repetition(work, 2)
    work.state[2][HALF] = np.uint64(2)
    assert not search.is_repetition(work, 2), "scanned past an irreversible move"


def test_the_root_static_evaluation_is_recorded_for_the_improving_test() -> None:
    """`improving` at ply 2 compares against static_evals[0]. search_root calls negamax at
    ply 1 and never at ply 0, so unless the root writes that slot itself nothing ever does,
    and the comparison silently degrades into `static > 0`."""
    board = chess.Board("r2q1rk1/pp2bppp/2np1n2/2p1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 9")
    search.WORK.static_evals[0] = np.int32(12345)
    search.think(board, 3000)
    recorded = int(search.WORK.static_evals[0])
    assert recorded != 12345, "search_root left a stale value in static_evals[0]"

    position.encode(board, search.WORK.state[0], search.WORK.mail[0])
    nnue.refresh(search.WORK.acc, 0, search.WORK.state[0], search.WORK.mail[0])
    expected = int(evaluate(search.WORK.acc, 0, search.WORK.state[0]))
    assert recorded == expected, "static_evals[0] is not the root's static evaluation"
