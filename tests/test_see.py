"""SEE, checked against hand-computed exchanges."""

import chess
import numpy as np
import pytest

import movegen
import position


def see_of(fen: str, uci: str) -> int:
    board = chess.Board(fen)
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    count = movegen.generate(state[0], moves, 0)
    move = next(moves[i] for i in range(count) if movegen.move_to_uci(int(moves[i])) == uci)
    return int(position.see(state[0], mailbox[0], move))


def test_free_pawn_capture_is_worth_a_pawn() -> None:
    # d5 is undefended, so the bishop just wins it.
    assert see_of("4k3/8/8/3p4/4B3/8/8/4K3 w - - 0 1", "e4d5") == 100


def test_defended_pawn_capture_by_a_bishop_loses_material() -> None:
    # Pawn on c6 defends d5. Bxd5 cxd5 is 100 - 330.
    assert see_of("4k3/8/2p5/3p4/4B3/8/8/4K3 w - - 0 1", "e4d5") == 100 - 330


def test_rook_takes_defended_rook_is_an_even_trade() -> None:
    # One white rook. Rxe7+ Kxe7 is 500 - 500.
    assert see_of("4k3/4r3/8/8/8/8/4R3/5K2 w - - 0 1", "e2e7") == 0


def test_king_may_not_recapture_into_check() -> None:
    """Doubled rooks defend each other, so Kxe7 is illegal and Rxe7 wins a whole rook.

    The swap algorithm gets this right without a special case: the king's 20000 value
    makes the walk-back decline the recapture, which is exactly what illegality means here.
    Verified against python-chess: Kxe7 is not in legal_moves after Rxe7.
    """
    assert see_of("4k3/4r3/8/8/8/8/4R3/4RK2 w - - 0 1", "e2e7") == 500


def test_x_ray_attacker_behind_the_capturer_is_counted() -> None:
    """The doubled rook on e1 is not a direct attacker of e4 until e2 is consumed.

    python-chess's own attackers() does not list it, which is exactly why a SEE that only
    enumerates direct attackers gets this wrong and prunes a winning capture.
    Rxe4 Rxe4 Rxe4 leaves white a pawn up: 100 - 500 + 500.
    """
    assert see_of("4k3/4r3/8/8/4p3/8/4R3/4RK2 w - - 0 1", "e2e4") == 100


def test_same_capture_without_the_battery_loses_a_rook() -> None:
    # Only one white rook, so Rxe4 Rxe4 is 100 - 500. These two positions differ only by
    # the rook on e1.
    assert see_of("4k3/4r3/8/8/4p3/8/4R3/5K2 w - - 0 1", "e2e4") == 100 - 500


def test_quiet_move_onto_an_undefended_square_is_zero() -> None:
    assert see_of("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1", "e2e5") == 0


def test_has_non_pawn_material() -> None:
    state, mailbox = position.new_stacks()
    position.encode(chess.Board("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1"), state[0], mailbox[0])
    assert not position.has_non_pawn_material(state[0], 0)
    assert not position.has_non_pawn_material(state[0], 1)
    position.encode(chess.Board("4k3/4p3/8/8/8/8/4P3/3QK3 w - - 0 1"), state[0], mailbox[0])
    assert position.has_non_pawn_material(state[0], 0)
    assert not position.has_non_pawn_material(state[0], 1)


@pytest.mark.parametrize(
    "fen,expected",
    [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", True),
        ("4k3/8/8/8/8/8/8/3BK3 w - - 0 1", True),
        ("4k3/8/8/8/8/8/8/3NK3 w - - 0 1", True),
        ("4k3/8/8/8/8/8/8/3RK3 w - - 0 1", False),
        ("4k3/4p3/8/8/8/8/8/4K3 w - - 0 1", False),
        ("4k3/8/8/8/8/8/8/2QQK3 w - - 0 1", False),
    ],
)
def test_insufficient_material(fen: str, expected: bool) -> None:
    state, mailbox = position.new_stacks()
    position.encode(chess.Board(fen), state[0], mailbox[0])
    assert bool(position.insufficient_material(state[0])) is expected
