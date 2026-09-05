"""The evaluation contract. Content is the user's to change; the contract is not."""

import chess

import evaluate
import position
from tests.conftest import random_positions


def evaluate_board(board: chess.Board) -> int:
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    return int(evaluate.evaluate(state[0], mailbox[0]))


def test_start_position_is_balanced() -> None:
    assert evaluate_board(chess.Board()) == 0


def test_score_is_relative_to_the_side_to_move() -> None:
    assert evaluate_board(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")) == 900
    assert evaluate_board(chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")) == -900


def test_evaluation_is_symmetric_under_colour_flip() -> None:
    # board.mirror() flips colours and the side to move, so a side-relative evaluation
    # must return the same number.
    for board in random_positions(count=80, seed=61):
        assert evaluate_board(board) == evaluate_board(board.mirror()), board.fen()


def test_material_is_counted_at_the_conventional_values() -> None:
    assert evaluate_board(chess.Board("4k3/8/8/8/8/8/8/3RK3 w - - 0 1")) == 500
    assert evaluate_board(chess.Board("4k3/8/8/8/8/8/8/3BK3 w - - 0 1")) == 330
    assert evaluate_board(chess.Board("4k3/8/8/8/8/8/8/3NK3 w - - 0 1")) == 320
    assert evaluate_board(chess.Board("4k3/8/8/8/8/8/3P4/4K3 w - - 0 1")) == 100


def test_returns_a_plain_int() -> None:
    state, mailbox = position.new_stacks()
    position.encode(chess.Board(), state[0], mailbox[0])
    assert isinstance(int(evaluate.evaluate(state[0], mailbox[0])), int)
