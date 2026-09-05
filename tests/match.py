"""A/B match harness with varied openings and an Elo interval.

harness/arena.py is left untouched, as CLAUDE.md requires. This drives the same
harness.referee.play_match directly, but passes a start_fen so a deterministic engine does
not replay one game twenty times, and alternates colours within each opening.

Two modes:

  --nodes N    fixed-node search on both sides. Deterministic and immune to machine load,
               so a 15 Elo change resolves in a few hundred games instead of a few thousand.
  --base-ms M  real time control. The only way to measure time management itself.

Opponents are snapshot directories: `--snapshot <tag>` freezes the current engine into
snapshots/<tag>/ so the next change can be measured against the version before it.
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Run directly (`make ab`), not under pytest, so the repo root has to be put on the path
# before harness and tests can be imported.
sys.path.insert(0, str(ROOT))

from harness.referee import FAILED_TERMINATIONS, play_match  # noqa: E402
from harness.sandbox import local  # noqa: E402
from tests.openings import OPENINGS  # noqa: E402

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100


def elo_with_interval(wins: int, draws: int, losses: int) -> tuple[float, float, float]:
    """Elo difference and a 95% interval, from a win/draw/loss record.

    A bare percentage over a few hundred games invites reading noise as progress, so the
    interval is reported alongside it and is the number that decides whether to keep a
    change.
    """
    games = wins + draws + losses
    if games == 0:
        return 0.0, 0.0, 0.0
    score = (wins + draws / 2) / games
    # Standard error of the mean score, treating a draw as a half point.
    variance = (
        wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0 - score) ** 2
    ) / games
    stderr = math.sqrt(variance / games) if games > 1 else 0.5

    def to_elo(value: float) -> float:
        clamped = min(max(value, 1e-6), 1 - 1e-6)
        return -400.0 * math.log10(1.0 / clamped - 1.0)

    return to_elo(score), to_elo(score - 1.96 * stderr), to_elo(score + 1.96 * stderr)


def snapshot(tag: str) -> Path:
    """Freeze the shipped files into snapshots/<tag>/ for use as an A/B opponent."""
    destination = ROOT / "snapshots" / tag
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source in sorted(ROOT.glob("*.py")):
        shutil.copy2(source, destination / source.name)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=ROOT)
    parser.add_argument("--opponent", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=0)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--snapshot", help="freeze the current engine under this tag and exit")
    arguments = parser.parse_args()

    if arguments.snapshot:
        path = snapshot(arguments.snapshot)
        print(f"snapshot written to {path.relative_to(ROOT)}")
        return 0

    if arguments.opponent is None:
        parser.error("--opponent is required unless --snapshot is given")

    # Both processes inherit this, so the two sides search identically.
    if arguments.nodes:
        os.environ["CHESSATHON_NODE_LIMIT"] = str(arguments.nodes)
    os.environ["CHESSATHON_INCREMENT_MS"] = str(arguments.increment_ms)

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    wins = draws = losses = 0
    terminations: dict[str, int] = {}

    for game in range(arguments.games):
        opening = OPENINGS[(game // 2) % len(OPENINGS)]
        plays_white = game % 2 == 0
        white, black = (agent, opponent) if plays_white else (opponent, agent)
        outcome = play_match(
            local(white),
            local(black),
            arguments.base_ms,
            arguments.increment_ms,
            start_fen=opening,
        )
        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
        if outcome.result in ("draw", "void"):
            draws += 1
        elif (outcome.result == "white") == plays_white:
            wins += 1
        else:
            losses += 1
        elo, low, high = elo_with_interval(wins, draws, losses)
        print(
            f"game {game + 1}/{arguments.games}: {outcome.result} by {outcome.termination}"
            f"  (+{wins} ={draws} -{losses}, {elo:+.0f} Elo [{low:+.0f}, {high:+.0f}])"
        )

    elo, low, high = elo_with_interval(wins, draws, losses)
    score = (wins + draws / 2) / max(arguments.games, 1)
    mode = f"{arguments.nodes} nodes" if arguments.nodes else f"{arguments.base_ms}ms"
    print(f"\n{arguments.agent} vs {arguments.opponent} over {arguments.games} games ({mode})")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print(f"Elo {elo:+.0f}, 95% interval [{low:+.0f}, {high:+.0f}]")
    if low > 0:
        print("verdict: a real gain")
    elif high < 0:
        print("verdict: a real loss, revert")
    else:
        print("verdict: not resolved, the interval spans zero. More games, or drop the change")

    print("terminations: " + ", ".join(f"{n} {c}" for n, c in terminations.items()))
    broken = {n: c for n, c in terminations.items() if n in FAILED_TERMINATIONS}
    if broken:
        print("FAILED to finish: " + ", ".join(f"{n} {c}" for n, c in broken.items()))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
