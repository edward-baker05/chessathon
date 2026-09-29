"""agent.py's game tracking: which positions continue the game it is playing."""

import chess
import pytest

import agent


@pytest.fixture
def after_our_reply() -> chess.Board:
    """The start position with our first move played, as agent.py remembers it."""
    board = chess.Board()
    board.push_uci("g1f3")
    agent._last_reply = board.copy(stack=False)
    return board


def continues(board: chess.Board) -> bool:
    # The harness sends a FEN, so the agent sees what survives the round trip through one.
    return agent._continues_our_game(chess.Board(board.fen()))


def test_a_double_push_nobody_can_take_continues_the_game(after_our_reply: chess.Board) -> None:
    after_our_reply.push_uci("e7e5")
    assert after_our_reply.ep_square is not None and "e6" not in after_our_reply.fen()
    assert continues(after_our_reply)


def test_a_double_push_that_can_be_taken_continues_the_game() -> None:
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    board.push_uci("h2h3")
    agent._last_reply = board.copy(stack=False)
    board.push_uci("d7d5")
    assert "d6" in board.fen()
    assert continues(board)


def test_an_unrelated_position_is_a_new_game(after_our_reply: chess.Board) -> None:
    assert not continues(chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"))
