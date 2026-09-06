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


def test_a_long_game_keeps_a_clock_into_the_middlegame() -> None:
    """Simulate a whole game's worth of moves against a draining clock.

    Not flagging is necessary but nowhere near sufficient. A geometric budget cannot flag,
    so the old form of this test, which asserted only that the clock stayed above zero,
    passed happily while the engine arrived at move 30 with four seconds and played the
    rest of the game at the increment. The clock at move 30 is the number that matters.
    """
    clock = 120_000.0
    increment = 500
    board = chess.Board()
    at_thirty = None
    for move_number in range(1, 41):
        started = time.perf_counter()
        uci = search.think(board, time_left_ms=int(clock), increment_ms=increment)
        clock -= (time.perf_counter() - started) * 1000
        assert clock > 0, f"flagged on move {move_number}"
        clock += increment
        if move_number == 30:
            at_thirty = clock
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)
        if board.is_game_over():
            break
    if at_thirty is not None:
        assert at_thirty > 30_000, f"only {at_thirty / 1000:.1f}s left at move 30"


SMOKE_POSITIONS = [
    "rnbqkb1r/pp1n1ppp/4p3/2ppP3/3P1P2/2NB4/PPP3PP/R1BQK1NR b KQkq - 1 6",
    "r1bqk2r/pp1pppbp/2n2np1/2p5/2P4P/2N2NP1/PP1PPPB1/R1BQK2R b KQkq - 0 6",
]


@pytest.mark.parametrize("fen", SMOKE_POSITIONS)
def test_a_single_move_stays_near_the_soft_limit(fen: str) -> None:
    """The two v4 smoke positions, which are where the overshoot was found.

    An iteration is only started when it is predicted to finish inside the soft limit, and
    an aspiration re-search is bounded by the stretch limit, so a move can land somewhat
    past the soft limit but never near the hard one. Before the fix the second of these
    positions took 3.4x its soft limit and 84% of its hard limit.
    """
    soft, hard = search.budget_ms(120_000, 500, ply=10)
    started = time.perf_counter()
    search.think(chess.Board(fen), time_left_ms=120_000, increment_ms=500)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < hard, f"{elapsed_ms / 1000:.2f}s reached the hard limit"
    ceiling = soft * (search.STRETCH_MULTIPLE + 0.5)
    assert elapsed_ms < ceiling, (
        f"{elapsed_ms / 1000:.2f}s is {elapsed_ms / soft:.2f}x the {soft / 1000:.2f}s soft limit"
    )


def test_the_budget_grows_as_a_share_of_the_clock_later_in_the_game() -> None:
    """A flat fraction of a draining clock spends most on the opening, which is the easiest
    part of the game, and least on the middlegame, which is the hardest. The share has to
    rise with the ply to keep the allocation roughly level."""
    early, _ = search.budget_ms(60_000, 500, ply=14)
    late, _ = search.budget_ms(60_000, 500, ply=60)
    assert late > early, "the budget share does not rise with the ply"


def test_the_hard_limit_can_always_contain_the_instability_stretch() -> None:
    """The soft limit is stretched by INSTABILITY_FACTOR when the best move is unsettled,
    capped at the hard limit. If the hard limit could fall below that stretch the cap would
    be doing the work and the extension would be silently dead."""
    for left in (1_000, 5_000, 20_000, 60_000, 120_000, 300_000):
        for ply in (0, 14, 60, 200):
            soft, hard = search.budget_ms(left, 500, ply=ply)
            assert hard >= soft * search.INSTABILITY_FACTOR, f"at {left}ms, ply {ply}"


def test_a_quiet_position_stops_before_the_soft_limit() -> None:
    """A locked position settles early. Checking the clock only after a completed depth
    means every move runs past its limit; predicting the next iteration means an easy one
    stops short and banks the difference for a position that needs it."""
    fen = "4k3/pp4pp/2p2p2/2Pp1P2/3Pp3/4P3/PP4PP/4K3 w - - 0 1"
    soft, _ = search.budget_ms(60_000, 500, ply=40)
    started = time.perf_counter()
    search.think(chess.Board(fen), time_left_ms=60_000, increment_ms=500)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < soft, f"used {elapsed_ms:.0f}ms of a {soft:.0f}ms quiet-position budget"


def test_a_settled_position_records_consecutive_stable_iterations() -> None:
    """The contraction side of the stability scaling is driven by this counter, so a
    position whose best move never changes has to actually increment it."""
    fen = "4k3/pp4pp/2p2p2/2Pp1P2/3Pp3/4P3/PP4PP/4K3 w - - 0 1"
    search.think(chess.Board(fen), time_left_ms=60_000, increment_ms=500)
    assert int(search.WORK.ints[search.I_STABLE]) > 0
