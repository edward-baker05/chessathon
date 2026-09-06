"""The material evaluation's contract.

`material_eval` is no longer what the engine plays with: `evaluate` is the network. It is
kept because it is the reference the network is measured against, as the frozen engine in
snapshots/material and as the A/B opponent that decides whether a net is an improvement.
The network's own contract is in tests/test_nnue.py.
"""

import chess

import evaluate
import position
from tests.conftest import random_positions


def material_of(board: chess.Board) -> int:
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    return int(evaluate.material_eval(state[0], mailbox[0]))


def test_start_position_is_balanced() -> None:
    assert material_of(chess.Board()) == 0


def test_score_is_relative_to_the_side_to_move() -> None:
    assert material_of(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")) == 900
    assert material_of(chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")) == -900


def test_evaluation_is_symmetric_under_colour_flip() -> None:
    # board.mirror() flips colours and the side to move, so a side-relative evaluation
    # must return the same number.
    for board in random_positions(count=80, seed=61):
        assert material_of(board) == material_of(board.mirror()), board.fen()


def test_material_is_counted_at_the_conventional_values() -> None:
    assert material_of(chess.Board("4k3/8/8/8/8/8/8/3RK3 w - - 0 1")) == 500
    assert material_of(chess.Board("4k3/8/8/8/8/8/8/3BK3 w - - 0 1")) == 330
    assert material_of(chess.Board("4k3/8/8/8/8/8/8/3NK3 w - - 0 1")) == 320
    assert material_of(chess.Board("4k3/8/8/8/8/8/3P4/4K3 w - - 0 1")) == 100


def test_returns_a_plain_int() -> None:
    state, mailbox = position.new_stacks()
    position.encode(chess.Board(), state[0], mailbox[0])
    assert isinstance(int(evaluate.material_eval(state[0], mailbox[0])), int)
