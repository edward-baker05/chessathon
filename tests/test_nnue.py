"""The network evaluation's contract.

The test that matters most here is `test_incremental_update_matches_a_full_refresh`. The
whole design rests on the accumulator being carried correctly from parent to child, and a
drift bug there would not crash, would not fail a perft, and would not look like anything
except an engine that plays slightly worse than it should.
"""

import chess
import numpy as np
import pytest

import movegen
import nnue
import position
from tests.conftest import random_positions

# Kiwipete and friends: positions dense in castling, en passant and promotion, so the
# rarely taken branches of `apply` are actually exercised rather than assumed.
AWKWARD = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
]


def encoded(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    state, mail = position.new_stacks()
    position.encode(board, state[0], mail[0])
    return state, mail


def refreshed(board: chess.Board) -> np.ndarray:
    """Both accumulators, rebuilt from the board rather than carried."""
    state, mail = encoded(board)
    acc = nnue.new_accumulator(2)
    nnue.refresh(acc, 0, state[0], mail[0])
    copied: np.ndarray = acc[0].copy()
    return copied


def evaluate_board(board: chess.Board) -> int:
    state, mail = encoded(board)
    acc = nnue.new_accumulator(2)
    nnue.refresh(acc, 0, state[0], mail[0])
    return int(nnue.forward(acc, 0, state[0]))


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """python-chess move to the engine's int32 encoding, via the engine's own generator.

    Going through `generate` rather than packing the fields by hand means the test cannot
    disagree with the engine about what flags a move carries, which is the thing being
    tested.
    """
    state, _mail = encoded(board)
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    count = movegen.generate(state[0], moves, 0)
    for index in range(count):
        if movegen.move_to_uci(int(moves[index])) == move.uci():
            return int(moves[index])
    raise AssertionError(f"{move.uci()} is not generated in {board.fen()}")


def test_forward_matches_a_plain_numpy_reimplementation() -> None:
    """The jitted arithmetic against something readable, so the quantisation is checked
    against a statement of intent rather than against itself."""
    for board in random_positions(count=25, seed=11):
        state, mail = encoded(board)
        acc = nnue.new_accumulator(2)
        nnue.refresh(acc, 0, state[0], mail[0])

        stm = int(state[0][8])  # STM
        pieces = bin(int(state[0][6]) | int(state[0][7])).count("1")
        bucket = (pieces - 2) // nnue.BUCKET_DIVISOR
        weights = nnue.OUT_WEIGHT[bucket].astype(np.int64)
        us = np.clip(acc[0, stm].astype(np.int64), 0, nnue.QA)
        them = np.clip(acc[0, 1 - stm].astype(np.int64), 0, nnue.QA)
        total = int((us * us * weights[: nnue.L1]).sum() + (them * them * weights[nnue.L1 :]).sum())
        want = (total // nnue.QA + int(nnue.OUT_BIAS[bucket])) * nnue.SCALE // (nnue.QA * nnue.QB)
        want = max(min(want, nnue.EVAL_LIMIT), -nnue.EVAL_LIMIT)

        assert int(nnue.forward(acc, 0, state[0])) == want, board.fen()


def test_evaluation_is_symmetric_under_colour_flip() -> None:
    """`board.mirror()` flips the board vertically and swaps both the colours and the side
    to move. A side-relative evaluation must return the same number, and it only does so
    if the perspective flip in `feature` is exactly right. This is the test that catches a
    reversed square flip or a swapped colour, which otherwise produce a net that trains
    fine and plays badly."""
    for board in random_positions(count=120, seed=61):
        assert evaluate_board(board) == evaluate_board(board.mirror()), board.fen()


def test_evaluation_stays_inside_the_clamp() -> None:
    """A score near MATE would be read by the search as a forced win that does not exist."""
    for board in random_positions(count=120, seed=62):
        assert abs(evaluate_board(board)) <= nnue.EVAL_LIMIT, board.fen()


@pytest.mark.parametrize("fen", AWKWARD)
def test_incremental_update_matches_a_full_refresh_on_awkward_positions(fen: str) -> None:
    board = chess.Board(fen)
    state, mail = encoded(board)
    acc = nnue.new_accumulator(4)
    nnue.refresh(acc, 0, state[0], mail[0])
    for move in board.legal_moves:
        nnue.apply(acc, 0, state[0], mail[0], np.int32(encode_move(board, move)))
        board.push(move)
        want = refreshed(board)
        board.pop()
        assert np.array_equal(acc[1], want), f"{move.uci()} in {fen}"


def test_incremental_update_matches_a_full_refresh_over_playouts() -> None:
    """Carry the accumulator down a real game and compare against a rebuild at every ply.

    This is the differential test the whole design rests on. `apply` duplicates the move
    decoding that `position.make` does, deliberately, so that position.py stays free of
    evaluation concerns. This is what keeps the two copies honest.
    """
    rng = np.random.default_rng(7)
    for game in range(40):
        board = chess.Board()
        acc = nnue.new_accumulator(96)
        state, mail = position.new_stacks()
        position.encode(board, state[0], mail[0])
        nnue.refresh(acc, 0, state[0], mail[0])

        for ply in range(60):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = moves[int(rng.integers(len(moves)))]
            nnue.apply(acc, ply, state[ply], mail[ply], np.int32(encode_move(board, move)))
            board.push(move)
            position.encode(board, state[ply + 1], mail[ply + 1])
            want = refreshed(board)
            assert np.array_equal(acc[ply + 1], want), (
                f"game {game} ply {ply} after {move.uci()} in {board.fen()}"
            )


def test_null_move_carries_the_accumulator_unchanged() -> None:
    for board in random_positions(count=20, seed=63):
        state, mail = encoded(board)
        acc = nnue.new_accumulator(4)
        nnue.refresh(acc, 0, state[0], mail[0])
        nnue.copy(acc, 0)
        assert np.array_equal(acc[0], acc[1]), board.fen()


def test_hot_loops_are_typed_contiguous() -> None:
    """Guards the measured 6x.

    Every hot loop in nnue.py is a separate jitted function with a fully explicit,
    C-contiguous signature. Written any other way, including inline in its caller or with
    lazy type inference, numba emits scalar code and the evaluation costs 522 ns instead
    of 87 ns. Nothing about that failure is visible except the node rate, so it is
    asserted here.
    """
    for function in (nnue._dot, nnue._move_one, nnue._move_two, nnue._set, nnue._add):
        signatures = function.nopython_signatures
        assert len(signatures) == 1, f"{function} compiled more than one signature"
        for argument in signatures[0].args:
            assert getattr(argument, "layout", "C") == "C", (
                f"{function} takes a non-contiguous {argument}, which compiles to scalar code"
            )
