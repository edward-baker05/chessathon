"""Replay a played game through the engine's clock, to measure time allocation.

An A/B match measures strength and costs hours. This measures *allocation*, costs minutes,
and is the cheap way to reject a bad time-management change before spending those hours.

The opponent's replies come from the PGN, so the positions are fixed and this says nothing
about Elo. What it does say is how the budget is spent across a real game: how far each
move ran past its soft limit, how much clock survives to the middlegame, and what depth the
engine reached once it got there.

The clock is simulated forward under the policy being measured rather than taken from the
PGN, because the policy is what changes the clock.

    uv run python tools/replay.py "logs/Epoch Mate vs Edward.pgn" --side Edward
    uv run python tools/replay.py --fen "<fen>" --moves 20
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

import search  # noqa: E402
from harness.rules import BASE_MS, INCREMENT_MS  # noqa: E402


class Move(argparse.Namespace):
    """One of our moves, as measured."""

    number: int
    before_ms: float
    used_ms: float
    soft_ms: float
    hard_ms: float
    played: str
    expected: str


def measure(board: chess.Board, clock_ms: float, increment_ms: int) -> Move:
    soft, hard = search.budget_ms(int(clock_ms), increment_ms, search.ply_of(board))
    started = time.perf_counter()
    uci = search.think(board, time_left_ms=int(clock_ms), increment_ms=increment_ms)
    used = (time.perf_counter() - started) * 1000
    return Move(
        number=board.fullmove_number,
        before_ms=clock_ms,
        used_ms=used,
        soft_ms=soft,
        hard_ms=hard,
        played=uci,
        expected="",
    )


def replay_pgn(path: Path, side: str, base_ms: int, increment_ms: int, limit: int) -> list[Move]:
    with path.open() as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise SystemExit(f"no game in {path}")
    ours = chess.WHITE if game.headers.get("White", "") == side else chess.BLACK
    board = game.board()
    clock = float(base_ms)
    measured: list[Move] = []
    for node in game.mainline():
        if board.turn == ours:
            move = measure(board, clock, increment_ms)
            move.expected = node.move.uci()
            clock = clock - move.used_ms + increment_ms
            measured.append(move)
            if clock <= 0:
                print(f"FLAGGED on move {move.number}")
                break
            if len(measured) >= limit:
                break
        board.push(node.move)
    return measured


def replay_fen(fen: str, base_ms: int, increment_ms: int, limit: int) -> list[Move]:
    """No opponent, so the engine answers itself. Enough to see the clock's shape."""
    board = chess.Board(fen)
    clock = float(base_ms)
    measured: list[Move] = []
    for _ in range(limit):
        move = measure(board, clock, increment_ms)
        clock = clock - move.used_ms + increment_ms
        measured.append(move)
        if clock <= 0:
            print(f"FLAGGED on move {move.number}")
            break
        board.push(chess.Move.from_uci(move.played))
        if board.is_game_over():
            break
    return measured


def report(name: str, measured: list[Move]) -> None:
    print(f"\n=== {name}")
    print(f"{'mv':>4} {'before':>8} {'used':>7} {'soft':>7} {'x soft':>7} {'%hard':>6}  move")
    for move in measured:
        agreed = "" if not move.expected else ("  =" if move.played == move.expected else "  x")
        print(
            f"{move.number:>4} {move.before_ms / 1000:7.1f}s {move.used_ms / 1000:6.2f}s "
            f"{move.soft_ms / 1000:6.2f}s {move.used_ms / move.soft_ms:6.2f}x "
            f"{move.used_ms / move.hard_ms * 100:5.0f}% {move.played}{agreed}"
        )
    ratios = [move.used_ms / move.soft_ms for move in measured]
    spent = sum(move.used_ms for move in measured)
    last = measured[-1]
    print(
        f"\n  {len(measured)} moves, {spent / 1000:.1f}s spent, "
        f"{(last.before_ms - last.used_ms) / 1000:.1f}s left at the end"
    )
    print(
        f"  overshoot mean {statistics.mean(ratios):.2f}x  median {statistics.median(ratios):.2f}x"
        f"  max {max(ratios):.2f}x"
    )
    for milestone in (10, 20, 30, 40):
        if len(measured) >= milestone:
            at = measured[milestone - 1]
            print(
                f"  after {milestone:>2} of our moves: {(at.before_ms - at.used_ms) / 1000:5.1f}s"
                f" left, spending {at.used_ms / 1000:.2f}s"
            )
    if any(move.expected for move in measured):
        matches = sum(1 for move in measured if move.played == move.expected)
        print(f"  agreed with the game as played on {matches} of {len(measured)} moves")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pgn", nargs="*", type=Path)
    parser.add_argument("--side", default="Edward", help="the PGN name of the side to measure")
    parser.add_argument("--fen", action="append", default=[], help="replay a position instead")
    parser.add_argument("--base-ms", type=int, default=BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    parser.add_argument("--moves", type=int, default=40, help="stop after this many of our moves")
    arguments = parser.parse_args()

    for path in arguments.pgn:
        measured = replay_pgn(
            path, arguments.side, arguments.base_ms, arguments.increment_ms, arguments.moves
        )
        report(path.name, measured)
    for fen in arguments.fen:
        report(fen, replay_fen(fen, arguments.base_ms, arguments.increment_ms, arguments.moves))


if __name__ == "__main__":
    main()
