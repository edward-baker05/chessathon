"""The clock. A flag loses the game outright, so these are hard requirements."""

import time

import chess
import pytest

import search


@pytest.fixture(autouse=True, scope="module")
def _warm() -> None:
    """Pay any first-call cost before the stopwatch tests, as import does in a real game."""
    search.think(chess.Board(), time_left_ms=1_000)


@pytest.mark.parametrize("time_left_ms", [400, 1_000, 5_000, 30_000, 120_000])
def test_never_exceeds_the_clock(time_left_ms: int) -> None:
    started = time.perf_counter()
    uci = search.think(chess.Board(), time_left_ms=time_left_ms, increment_ms=500)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < time_left_ms, f"used {elapsed_ms:.0f}ms of {time_left_ms}ms"
    assert chess.Move.from_uci(uci) in chess.Board().legal_moves


@pytest.mark.parametrize("time_left_ms", [0, 1, 5, 50])
def test_returns_a_legal_move_with_essentially_no_clock(time_left_ms: int) -> None:
    """Below the 300ms reserve the budget floors out. We cannot avoid flagging there, but
    we must still return a legal move rather than crash or return nothing."""
    uci = search.think(chess.Board(), time_left_ms=time_left_ms, increment_ms=500)
    assert chess.Move.from_uci(uci) in chess.Board().legal_moves


def test_budget_is_floored_when_the_clock_is_below_the_reserve() -> None:
    soft, hard = search.budget_ms(100, 500)
    assert soft > 0 and hard > 0 and soft <= hard


def test_budget_never_goes_negative() -> None:
    for left in (-1000, 0, 1, 299, 300, 301):
        soft, hard = search.budget_ms(left, 500)
        assert soft > 0 and hard > 0, f"negative budget at {left}ms"


def test_budget_grows_with_the_clock() -> None:
    small, _ = search.budget_ms(10_000, 500)
    large, _ = search.budget_ms(100_000, 500)
    assert large > small


def test_budget_never_spends_more_than_a_twelfth_of_the_clock() -> None:
    """The increment credit must not let a single move eat the game."""
    for left in (1_000, 10_000, 60_000, 120_000):
        soft, _ = search.budget_ms(left, 500)
        assert soft <= left / 12.0 + 1e-6, f"soft budget too large at {left}ms"


def test_uses_a_meaningful_share_of_a_healthy_clock() -> None:
    started = time.perf_counter()
    search.think(chess.Board(), time_left_ms=60_000, increment_ms=500)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert 500 < elapsed_ms < 6_000, f"used {elapsed_ms:.0f}ms, which is wasteful or reckless"


def test_a_long_game_never_runs_the_clock_out() -> None:
    """Simulate a whole game's worth of moves against a draining clock.

    This is the failure the rules call the most common self-inflicted loss: a budget that
    looks fine per move but does not shrink fast enough as the clock does.
    """
    clock = 120_000.0
    increment = 500
    board = chess.Board()
    for _ in range(60):
        started = time.perf_counter()
        uci = search.think(board, time_left_ms=int(clock), increment_ms=increment)
        clock -= (time.perf_counter() - started) * 1000
        assert clock > 0, "flagged during a simulated game"
        clock += increment
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)
        if board.is_game_over():
            break
    assert clock > 0
