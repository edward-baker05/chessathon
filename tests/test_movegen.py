"""Pseudo-legal generation, checked against python-chess."""

import chess
import numpy as np

import movegen
import position
from tests.conftest import random_positions


def generated_ucis(board: chess.Board, captures_only: bool = False) -> set[str]:
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    generator = movegen.generate_captures if captures_only else movegen.generate
    count = generator(state[0], moves, 0)
    return {movegen.move_to_uci(int(moves[i])) for i in range(count)}


def test_move_codec_round_trips() -> None:
    packed = movegen.pack(chess.E7, chess.E8, promo=4, flag=3)
    assert movegen.move_from(np.int32(packed)) == chess.E7
    assert movegen.move_to(np.int32(packed)) == chess.E8
    assert movegen.move_promo(np.int32(packed)) == 4
    assert movegen.move_flag(np.int32(packed)) == 3
    assert movegen.move_to_uci(packed) == "e7e8q"


def test_pseudo_legal_is_a_superset_of_legal() -> None:
    for board in random_positions(count=200, seed=31):
        legal = {move.uci() for move in board.legal_moves}
        assert legal <= generated_ucis(board), f"missing legal moves in {board.fen()}"


def without_castling(ucis: set[str]) -> set[str]:
    """Drop king two-square moves, detected structurally so both sides are filtered.

    python-chess's pseudo_legal_moves already excludes castling that lands the king on an
    attacked square. We deliberately do not: the post-make legality filter catches it, the
    same way it catches any other move into check. Castling is covered by the dedicated
    tests below and, exhaustively, by perft.
    """
    castling = {"e1g1", "e1c1", "e8g8", "e8c8"}
    return {uci for uci in ucis if uci not in castling}


def test_pseudo_legal_matches_python_chess_exactly() -> None:
    for board in random_positions(count=200, seed=32):
        expected = without_castling({move.uci() for move in board.pseudo_legal_moves})
        got = without_castling(generated_ucis(board))
        assert got == expected, f"mismatch in {board.fen()}"


def test_generate_captures_yields_only_captures_and_queen_promotions() -> None:
    for board in random_positions(count=120, seed=33):
        for uci in generated_ucis(board, captures_only=True):
            move = chess.Move.from_uci(uci)
            assert board.is_capture(move) or move.promotion == chess.QUEEN, uci


def test_generate_captures_finds_every_legal_capture() -> None:
    for board in random_positions(count=120, seed=34):
        expected = {m.uci() for m in board.legal_moves if board.is_capture(m)}
        assert expected <= generated_ucis(board, captures_only=True), board.fen()


def test_en_passant_capture_is_generated() -> None:
    board = chess.Board("8/8/8/8/4pP2/8/8/K6k b - f3 0 1")
    assert "e4f3" in generated_ucis(board)


def test_all_four_promotions_are_generated() -> None:
    board = chess.Board("8/4P3/8/8/8/8/8/K6k w - - 0 1")
    assert {"e7e8q", "e7e8r", "e7e8b", "e7e8n"} <= generated_ucis(board)


def test_castling_is_generated_when_available() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert {"e1g1", "e1c1"} <= generated_ucis(board)


def test_castling_through_an_attacked_square_is_not_generated() -> None:
    # Bishop on a6 attacks f1 along a6-b5-c4-d3-e2-f1. It does NOT attack e1, so the king
    # is not in check: kingside castling is illegal purely because f1 is crossed.
    board = chess.Board("4k3/8/b7/8/8/8/8/R3K2R w KQ - 0 1")
    generated = generated_ucis(board)
    assert "e1g1" not in generated, "f1 is attacked, so kingside castling is illegal"
    assert "e1c1" in generated, "queenside is unaffected and must still be generated"
