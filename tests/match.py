"""Paired A/B match between two engine directories. `make ab`.

This is the only thing in the repository that measures strength. Everything else measures
nodes, depth or throughput, and none of those decide what ships.

Three things make the number honest.

**Openings are paired.** Every opening is played twice, once with each engine as White, and
the *pair* is the unit of measurement, not the game. Two games from one opening are not two
independent samples: they share the position, most of the opening theory and often most of
the middlegame. Scoring them as two samples makes the interval about 40% narrower than the
evidence supports.

**The increment is forwarded.** `agent.py` freezes its increment from the environment at
import, and defaults to the rated 500 ms. A fast local game runs a 100 ms increment, so
without `CHESSATHON_INCREMENT_MS` the agent budgets for two and a half times the increment
it is actually given and overspends its clock all game. This launcher sets it. It does not
touch the harness, which mirrors the platform and must not be edited.

**Clusters, not pairs, bound the interval.** Openings pulled out of the same source game
are the same game seen several times. Treating them as independent draws reports an
interval narrower than the evidence supports, so when the opening file carries a `cluster`
field the interval is computed over cluster means. The opening order is shuffled with a
recorded seed, so a short run is a random subset rather than the front of a sorted list.

**Wall-time games need a core each.** Games on a clock measure a time-management policy,
and running several on a shared CPU charges every policy for the others. `--pin` gives each
worker one physical core, which both agents of its game inherit; that is close to what the
platform does, since only one side of a game thinks at a time. Without `--pin`, a wall-time
run is held to one worker. Fixed-node games have no clock in them and parallelise freely.

    uv run python tests/match.py --opponent snapshots/net768 --games 200 --nodes 200000
    uv run python tests/match.py --opponent snapshots/net768 --games 100 --base-ms 120000

The interval reported is a plain 95% normal interval on the mean pair score, over a run
length fixed before the run started. It is not a sequential test: stopping early because the
number looks good, or continuing because it does not, invalidates it. A 200-game pilot can
reject a substantial regression. It cannot resolve a ten Elo gain, and reporting one as if
it could is how a worse build gets shipped.
"""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import queue
import random
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from harness.referee import FAILED_TERMINATIONS, Outcome, play_match  # noqa: E402
from harness.rules import BASE_MS, INCREMENT_MS, PLY_CAP  # noqa: E402
from harness.sandbox import local  # noqa: E402

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100
# Below this many independent clusters, the interval is arithmetic rather than evidence and
# no verdict is printed. A handful of pairs can still catch a build that crashes or plays an
# illegal move, which is what a short run is for.
MIN_CLUSTERS_FOR_A_VERDICT = 20
DEFAULT_OPENINGS = ROOT / "audit" / "positions.json"


def load_openings(path: Path | None) -> list[tuple[str, str]]:
    """Openings as (fen, cluster).

    Accepts a bare list of FENs, or `{"positions": [{"fen": ..., "cluster": ...}]}`. An
    opening with no cluster is its own cluster, which is the right default for a set whose
    positions really are independent.
    """
    if path is None:
        return [(chess.STARTING_FEN, "start")]
    payload = json.loads(path.read_text())
    rows = payload["positions"] if isinstance(payload, dict) else payload
    openings: list[tuple[str, str]] = []
    for row in rows:
        fen = str(row["fen"] if isinstance(row, dict) else row)
        cluster = str(row.get("cluster", fen)) if isinstance(row, dict) else fen
        board = chess.Board(fen)
        if not board.is_valid():
            raise SystemExit(f"invalid opening in {path}: {fen}")
        if board.is_game_over():
            raise SystemExit(f"opening is already over in {path}: {fen}")
        openings.append((fen, cluster))
    return openings


def physical_cores() -> list[int]:
    """One CPU per physical core, out of those this process is allowed to use.

    Hyperthread siblings share execution units, so two games on one core's two threads
    still contend and still corrupt a clock.
    """
    allowed = sorted(os.sched_getaffinity(0))
    chosen: list[int] = []
    claimed: set[int] = set()
    for cpu in allowed:
        if cpu in claimed:
            continue
        siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            listed = siblings.read_text().strip().replace("-", ",").split(",")
        except OSError:
            listed = [str(cpu)]
        group = {int(part) for part in listed}
        chosen.append(cpu)
        claimed |= group
    return chosen


def fingerprint(directory: Path) -> str:
    """What actually played. A build is its source and its weights, nothing else."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".npz", ".onnx", ".safetensors", ".pt"}:
            continue
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def score_for(outcome: Outcome, agent_is_white: bool) -> float:
    if outcome.result in ("draw", "void"):
        return 0.5
    return 1.0 if (outcome.result == "white") == agent_is_white else 0.0


def elo(score: float) -> float:
    """Logistic Elo difference for a score. Saturates rather than dividing by zero."""
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200,
                        help="rounded down to a whole number of colour-swapped pairs")
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--nodes", type=int, default=0,
                        help="fixed node budget per move; 0 plays on the clock instead")
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--rated", action="store_true",
                        help=f"play the real control, {BASE_MS}ms + {INCREMENT_MS}ms")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minutes", type=float, default=0.0,
                        help="stop starting new pairs after this long and report what finished")
    parser.add_argument("--pin", action="store_true",
                        help="give each worker its own physical core; required for parallel "
                             "wall-time games")
    parser.add_argument("--shuffle-seed", type=int, default=20260909,
                        help="seed for the opening order, so a short run is a random subset")
    parser.add_argument("--stagger", type=float, default=15.0,
                        help="seconds between each worker's first pair, so their agents do "
                             "not all compile at once and blow the 90s init budget")
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--pgn", type=Path, help="append every game here")
    parser.add_argument("--json", type=Path, help="write the raw per-pair results here")
    arguments = parser.parse_args()

    base_ms = BASE_MS if arguments.rated else arguments.base_ms
    increment_ms = INCREMENT_MS if arguments.rated else arguments.increment_ms
    openings = load_openings(
        arguments.openings if arguments.openings and arguments.openings.exists() else None
    )
    pairs = max(1, arguments.games // 2)

    # A short run should be a random subset of the openings, not the front of a sorted
    # list. The seed is recorded so the same subset can be played again.
    random.Random(arguments.shuffle_seed).shuffle(openings)

    cores = physical_cores() if arguments.pin else []
    if arguments.pin and arguments.workers > len(cores):
        raise SystemExit(
            f"--pin has {len(cores)} physical cores to give out and --workers is "
            f"{arguments.workers}. Two workers on one core is the contention --pin exists "
            "to avoid."
        )
    if arguments.nodes == 0 and arguments.workers > 1 and not arguments.pin:
        raise SystemExit(
            "wall-time games measure a clock, and workers sharing a CPU corrupt it. "
            "Add --pin to give each worker its own core, use --nodes, or use --workers 1."
        )

    # The two settings the agent reads from its environment, set here rather than in the
    # harness. Both sides get the same ones, which is what makes the comparison fair.
    os.environ["CHESSATHON_INCREMENT_MS"] = str(increment_ms)
    os.environ["CHESSATHON_NODE_LIMIT"] = str(arguments.nodes)

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    control = (f"{arguments.nodes:,} nodes/move" if arguments.nodes
               else f"{base_ms}ms + {increment_ms}ms")
    provenance = {
        "agent": str(agent), "agent_build": fingerprint(agent),
        "opponent": str(opponent), "opponent_build": fingerprint(opponent),
        "pairs": pairs, "openings": len(openings), "control": control,
        "node_limit": arguments.nodes, "base_ms": base_ms, "increment_ms": increment_ms,
        "ply_cap": arguments.ply_cap, "workers": arguments.workers,
        "openings_file": str(arguments.openings),
        "clusters": len({cluster for _, cluster in openings}),
        "shuffle_seed": arguments.shuffle_seed, "stagger_s": arguments.stagger,
        "pinned_cores": cores[:arguments.workers] if cores else None,
    }
    print(json.dumps(provenance, indent=2))
    print(f"\n{pairs} pairs ({2 * pairs} games) at {control}\n")

    # Each worker thread claims one core for the life of the run. Affinity is per-thread on
    # Linux and a spawned child inherits the affinity of the thread that spawned it, so both
    # agents of a game land on the core their worker holds.
    free_cores = queue.Queue[int]()
    for cpu in cores[:arguments.workers]:
        free_cores.put(cpu)
    pinned = threading.local()
    stagger_slot = itertools.count()

    def play(index: int) -> tuple[int, float, Outcome, Outcome]:
        if not getattr(pinned, "started", False):
            if cores:
                os.sched_setaffinity(0, {free_cores.get()})
            # Every agent spends most of a minute compiling before it can move, and the
            # harness holds it to the platform's 90 second init budget. Starting every
            # worker at once puts two compilations per worker on the machine
            # simultaneously, which on a small box takes them all past that budget and
            # scores a pile of games as init failures. Spreading the first pairs out costs
            # a few minutes once and nothing afterwards, because workers finish at
            # different times and stay spread out on their own.
            time.sleep(arguments.stagger * next(stagger_slot))
            pinned.started = True
        fen, _cluster = openings[index % len(openings)]
        first = play_match(local(agent), local(opponent), base_ms, increment_ms,
                           ply_cap=arguments.ply_cap, start_fen=fen)
        second = play_match(local(opponent), local(agent), base_ms, increment_ms,
                            ply_cap=arguments.ply_cap, start_fen=fen)
        return index, score_for(first, True) + score_for(second, False), first, second

    # Pairs are started a few at a time rather than all at once, so a deadline can stop the
    # run cleanly: everything already started finishes and is counted, and nothing new
    # begins. A run cut short is a shorter run, not a biased one, because which pair goes
    # first has nothing to do with how it turns out.
    started = time.perf_counter()
    deadline = started + arguments.minutes * 60 if arguments.minutes else float("inf")
    results: list[tuple[int, float, Outcome, Outcome]] = []
    stopped_early = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        queued = iter(range(pairs))
        running = {pool.submit(play, index)
                   for index in itertools.islice(queued, arguments.workers)}
        while running:
            done, running = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                results.append(future.result())
                index, pair_score, first, second = results[-1]
                elapsed = (time.perf_counter() - started) / 60
                print(f"[{elapsed:5.1f} min] pair {len(results)}/{pairs} "
                      f"(opening {index % len(openings)}): {pair_score:.1f}/2  "
                      f"[{first.termination}, {second.termination}]", flush=True)
            if time.perf_counter() >= deadline:
                if not stopped_early:
                    print(f"\n--- {arguments.minutes:.0f} minute deadline reached, "
                          "finishing what is running and stopping ---\n", flush=True)
                stopped_early = True
                continue
            running |= {pool.submit(play, index)
                        for index in itertools.islice(queued, len(done))}
    results.sort()
    if not results:
        raise SystemExit("no pair finished")
    pairs = len(results)

    scores = [pair for _, pair, _, _ in results]
    games = [(g, w) for _, _, a, b in results for g, w in ((a, True), (b, False))]
    wins = sum(1 for outcome, white in games if score_for(outcome, white) == 1.0)
    draws = sum(1 for outcome, white in games if score_for(outcome, white) == 0.5)
    losses = len(games) - wins - draws
    score = statistics.mean(scores) / 2.0

    # The independent unit is the cluster, not the pair. Three openings taken out of one
    # source game are one game seen three times: they share the position, the theory and
    # usually the middlegame, so their results move together. Averaging within a cluster
    # first and taking the interval over cluster means is what stops that correlation from
    # being counted as extra evidence.
    by_cluster: dict[str, list[float]] = {}
    for index, pair, _first, _second in results:
        by_cluster.setdefault(openings[index % len(openings)][1], []).append(pair)
    units = [statistics.mean(values) for values in by_cluster.values()]
    centre = statistics.mean(units)
    deviation = statistics.stdev(units) if len(units) > 1 else 0.0
    error = 1.96 * deviation / math.sqrt(len(units)) if units else 0.0
    low, high = (centre - error) / 2.0, (centre + error) / 2.0

    terminations: dict[str, int] = {}
    for outcome, _ in games:
        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1

    print(f"\n+{wins} ={draws} -{losses} over {len(games)} games "
          f"in {(time.perf_counter() - started) / 60:.1f} min"
          + (" (deadline, run cut short)" if stopped_early else ""))
    print(f"score {score:.3%} on {len(scores)} pairs in {len(units)} clusters")
    print("terminations: " + ", ".join(f"{n} {c}" for n, c in sorted(terminations.items())))

    broken = {n: c for n, c in terminations.items() if n in FAILED_TERMINATIONS}
    if broken:
        print("\nRUN INVALID. " + ", ".join(f"{n} {c}" for n, c in sorted(broken.items()))
              + ".\nA game that ended this way was decided by something other than chess, "
              "and its result\nis in the score above. Fix the cause and run again; do not "
              "read the number.\nAn init failure under a parallel run usually means too "
              "many agents compiled at once:\nlower --workers or raise --stagger.")
    elif len(units) < MIN_CLUSTERS_FOR_A_VERDICT:
        print(f"\n{len(units)} independent clusters is a smoke test, not a measurement. An\n"
              f"interval over this few is arithmetic on noise, so none is reported and there\n"
              f"is no verdict here. Run at least {MIN_CLUSTERS_FOR_A_VERDICT} clusters before "
              "reading anything into the score.")
    else:
        print(f"95% CI [{low:.3%}, {high:.3%}]  (over {len(units)} clusters, "
              f"{len(scores) / len(units):.1f} pairs each)")
        print(f"Elo   {elo(score):+.0f}     95% CI [{elo(low):+.0f}, {elo(high):+.0f}]")
        if low > 0.5:
            print("\nThe interval excludes equality: this build is ahead on this opening set.")
        elif high < 0.5:
            print("\nThe interval excludes equality: this build is BEHIND. Do not ship it.")
        else:
            print("\nThe interval contains equality. This run did not resolve the difference; "
                  "it is not evidence of a gain.")

    if arguments.pgn:
        arguments.pgn.write_text(
            "\n\n".join(outcome.pgn for outcome, _ in games) + "\n"
        )
        print(f"wrote {arguments.pgn}")
    if arguments.json:
        arguments.json.write_text(json.dumps({
            "provenance": provenance,
            "pairs": [
                {"opening": index % len(openings),
                 "fen": openings[index % len(openings)][0],
                 "cluster": openings[index % len(openings)][1],
                 "score": pair, "terminations": [a.termination, b.termination]}
                for index, pair, a, b in results
            ],
            "summary": {"wins": wins, "draws": draws, "losses": losses, "score": score,
                        "pairs": len(scores), "clusters": len(units),
                        "ci": [low, high], "elo": elo(score),
                        "elo_ci": [elo(low), elo(high)]},
        }, indent=2) + "\n")
        print(f"wrote {arguments.json}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
