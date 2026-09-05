"""Position encoding and attack detection, checked against python-chess."""

import chess
import numpy as np

import bitboard as bb
import position
from tests.conftest import random_positions


def encoded(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    return state[0], mailbox[0]


def test_encode_matches_python_chess_bitboards() -> None:
    for board in random_positions(count=100, seed=21):
        state, _ = encoded(board)
        assert int(state[bb.PAWN]) == board.pawns
        assert int(state[bb.KNIGHT]) == board.knights
        assert int(state[bb.BISHOP]) == board.bishops
        assert int(state[bb.ROOK]) == board.rooks
        assert int(state[bb.QUEEN]) == board.queens
        assert int(state[bb.KING]) == board.kings
        assert int(state[bb.WOCC]) == board.occupied_co[chess.WHITE]
        assert int(state[bb.BOCC]) == board.occupied_co[chess.BLACK]
        assert int(state[bb.STM]) == (0 if board.turn else 1)
        assert int(state[bb.HALF]) == board.halfmove_clock


def test_encode_stores_en_passant_as_square_plus_one() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    state, _ = encoded(board)
    assert int(state[bb.EP]) == chess.E3 + 1, "0 must be reserved to mean 'no en passant'"

    quiet = chess.Board()
    state, _ = encoded(quiet)
    assert int(state[bb.EP]) == 0


def test_encode_stores_castling_rights_bits() -> None:
    state, _ = encoded(chess.Board())
    assert int(state[bb.CASTLE]) == 15
    state, _ = encoded(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"))
    assert int(state[bb.CASTLE]) == 1 | 8


def test_mailbox_agrees_with_the_board() -> None:
    for board in random_positions(count=60, seed=22):
        _, mailbox = encoded(board)
        for square in range(64):
            piece = board.piece_at(square)
            assert mailbox[square] == (-1 if piece is None else piece.piece_type - 1)


def test_attacked_matches_python_chess() -> None:
    for board in random_positions(count=60, seed=23):
        state, _ = encoded(board)
        for square in range(64):
            for by_black, colour in ((0, chess.WHITE), (1, chess.BLACK)):
                assert bool(position.attacked(state, square, by_black)) == board.is_attacked_by(
                    colour, square
                ), f"square {square} by {'black' if by_black else 'white'} in {board.fen()}"


def test_in_check_matches_python_chess() -> None:
    for board in random_positions(count=200, seed=24):
        state, _ = encoded(board)
        assert bool(position.in_check(state)) == board.is_check()


def test_king_square_finds_the_right_king() -> None:
    for board in random_positions(count=60, seed=26):
        state, _ = encoded(board)
        assert position.king_square(state, 0) == board.king(chess.WHITE)
        assert position.king_square(state, 1) == board.king(chess.BLACK)


def test_decode_fen_round_trips() -> None:
    # en_passant="fen" always prints the square when one is set. python-chess's default
    # omits it unless the capture is legal, but we store the raw square, so this is the
    # representation to compare against.
    for board in random_positions(count=60, seed=25):
        state, mailbox = encoded(board)
        assert position.decode_fen(state, mailbox) == board.fen(en_passant="fen")
