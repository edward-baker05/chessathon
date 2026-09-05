"""The A/B harness itself, tested without playing real games."""

import chess

from tests.match import elo_with_interval
from tests.openings import OPENINGS


def test_openings_are_varied_legal_and_balanced() -> None:
    assert len(OPENINGS) >= 20
    assert len(set(OPENINGS)) == len(OPENINGS)
    for fen in OPENINGS:
        board = chess.Board(fen)
        assert board.is_valid()
        assert not board.is_game_over()


def test_even_score_is_zero_elo() -> None:
    elo, low, high = elo_with_interval(wins=50, draws=0, losses=50)
    assert elo == 0
    assert low < 0 < high


def test_winning_score_is_positive_elo() -> None:
    elo, low, _ = elo_with_interval(wins=70, draws=0, losses=30)
    assert elo > 100
    assert low > 0


def test_losing_score_is_negative_elo() -> None:
    elo, _, high = elo_with_interval(wins=30, draws=0, losses=70)
    assert elo < -100
    assert high < 0


def test_interval_narrows_with_more_games() -> None:
    _, low_small, high_small = elo_with_interval(wins=14, draws=0, losses=6)
    _, low_big, high_big = elo_with_interval(wins=700, draws=0, losses=300)
    assert (high_big - low_big) < (high_small - low_small)


def test_draws_count_as_half() -> None:
    elo, _, _ = elo_with_interval(wins=0, draws=100, losses=0)
    assert elo == 0


def test_all_draws_gives_a_tight_interval() -> None:
    """Every game the same result means no variance, so the interval must not be wide."""
    _, low, high = elo_with_interval(wins=0, draws=200, losses=0)
    assert high - low < 1


def test_no_games_is_handled() -> None:
    assert elo_with_interval(0, 0, 0) == (0.0, 0.0, 0.0)
