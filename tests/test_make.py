"""make() must produce exactly the state python-chess would."""

import chess
import numpy as np

import bitboard as bb
import movegen
import position
from tests.conftest import random_positions


def test_make_matches_python_chess_on_every_legal_move() -> None:
    state, mailbox = position.new_stacks()
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    expected_state, expected_mail = position.new_stacks()
    for board in random_positions(count=150, seed=41):
        position.encode(board, state[0], mailbox[0])
        count = movegen.generate(state[0], moves, 0)
        for i in range(count):
            uci = movegen.move_to_uci(int(moves[i]))
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            position.make(state[0], mailbox[0], state[1], mailbox[1], moves[i])
            board.push(move)
            position.encode(board, expected_state[0], expected_mail[0])
            board.pop()
            for field in (
                bb.PAWN,
                bb.KNIGHT,
                bb.BISHOP,
                bb.ROOK,
                bb.QUEEN,
                bb.KING,
                bb.WOCC,
                bb.BOCC,
                bb.STM,
                bb.CASTLE,
                bb.EP,
                bb.HALF,
            ):
                assert state[1][field] == expected_state[0][field], (
                    f"field {field} wrong after {uci} in {board.fen()}"
                )
            assert list(mailbox[1]) == list(expected_mail[0]), f"mailbox wrong after {uci}"


def test_en_passant_removes_the_pawn_beside_the_target_not_behind_it() -> None:
    # Black plays e4xf3 e.p.; the captured white pawn stands on f4, which is to + 8.
    board = chess.Board("8/8/8/8/4pP2/8/8/K6k b - f3 0 1")
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    count = movegen.generate(state[0], moves, 0)
    move = next(moves[i] for i in range(count) if movegen.move_to_uci(int(moves[i])) == "e4f3")
    position.make(state[0], mailbox[0], state[1], mailbox[1], move)
    assert state[1][bb.PAWN] & (np.uint64(1) << np.uint64(chess.F4)) == 0, "f4 pawn must be gone"
    assert mailbox[1][chess.F4] == -1
    assert mailbox[1][chess.F3] == chess.PAWN - 1


def test_en_passant_that_exposes_the_king_is_rejected_by_the_legality_filter() -> None:
    # The classic horizontal pin: after e2e4, f4xe3 e.p. would expose the black king on h4.
    board = chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    board.push_uci("e2e4")
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    count = movegen.generate(state[0], moves, 0)
    move = next(moves[i] for i in range(count) if movegen.move_to_uci(int(moves[i])) == "f4e3")
    position.make(state[0], mailbox[0], state[1], mailbox[1], move)
    assert not position.legal_after(state[1], 1)
    assert chess.Move.from_uci("f4e3") not in board.legal_moves


def test_castling_moves_the_rook_and_clears_the_rights() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    count = movegen.generate(state[0], moves, 0)
    move = next(moves[i] for i in range(count) if movegen.move_to_uci(int(moves[i])) == "e1g1")
    position.make(state[0], mailbox[0], state[1], mailbox[1], move)
    assert mailbox[1][chess.G1] == chess.KING - 1
    assert mailbox[1][chess.F1] == chess.ROOK - 1
    assert mailbox[1][chess.E1] == -1
    assert mailbox[1][chess.H1] == -1
    assert int(state[1][bb.CASTLE]) == bb.BLACK_KINGSIDE | bb.BLACK_QUEENSIDE


def test_capturing_a_rook_on_its_home_square_clears_that_right() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push_uci("a1a8")
    expected_state, expected_mail = position.new_stacks()
    position.encode(board, expected_state[0], expected_mail[0])
    assert not int(expected_state[0][bb.CASTLE]) & bb.BLACK_QUEENSIDE

    board.pop()
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    count = movegen.generate(state[0], moves, 0)
    move = next(moves[i] for i in range(count) if movegen.move_to_uci(int(moves[i])) == "a1a8")
    position.make(state[0], mailbox[0], state[1], mailbox[1], move)
    assert int(state[1][bb.CASTLE]) == int(expected_state[0][bb.CASTLE])
