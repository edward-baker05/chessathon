"""Proves the test harness can import root modules and generate positions."""

from tests.conftest import random_positions


def test_random_positions_are_varied_and_legal() -> None:
    boards = list(random_positions(count=50, seed=1, max_plies=30))
    assert len(boards) == 50
    fens = {board.fen() for board in boards}
    assert len(fens) > 40, "positions should be varied, not the same opening repeated"
    for board in boards:
        assert board.is_valid()


def test_random_positions_are_deterministic() -> None:
    first = [board.fen() for board in random_positions(count=20, seed=7, max_plies=20)]
    second = [board.fen() for board in random_positions(count=20, seed=7, max_plies=20)]
    assert first == second


def test_root_modules_are_importable_from_tests() -> None:
    """pythonpath must reach the repo root, or every later test fails to collect."""
    import agent

    assert callable(agent.get_move)
