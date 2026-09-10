"""Two builds on the same positions, judged in one reference pass.

Running `tools/judge.py` twice and comparing the two reports does not compare the builds.
Each run scores its own build's move against the best of its own candidate set, so a move
only one of them ever proposed is missing from the other's set, and whichever build
proposed the better move gets a lower bar to clear. The bias flatters whichever build
happens to explore more.

So the move sets are merged first and every move either build produced is scored once,
under the same restricted-move conditions, against the same set. Then both builds are read
off the same numbers.

Both probe files must have been produced from the same sampler, seed and limit, which makes
them the same positions in the same order; that is checked rather than assumed.

    uv run python tools/compare.py --probe base.json --probe cand.json \\
        --name baseline --name candidate --arm budget --engine /path/to/sf --out cmp.json
"""

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from tools.failures import board_of  # noqa: E402
from tools.judge import Pool, cluster_bootstrap, rank, regret, restricted  # noqa: E402

SEVERE_CP = 100


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--probe", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--arm", default="budget",
                        choices=("shipped", "floor", "equal_nodes", "budget"),
                        help="which of each probe's searches to compare")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.out.exists():
        parser.error(f"{arguments.out} exists; evidence is never overwritten, pick a new name")
    if len(arguments.probe) != len(arguments.name):
        parser.error("one --name per --probe")
    loaded = [json.loads(path.read_text()) for path in arguments.probe]
    keys = [(p["provenance"]["seed"], p["provenance"]["limit"], p["provenance"]["depth"])
            for p in loaded]
    if len({tuple(k) for k in keys}) != 1:
        parser.error(f"probes were not sampled the same way: {keys}")
    lengths = {len(p["positions"]) for p in loaded}
    if len(lengths) != 1:
        parser.error(f"probes have different lengths: {lengths}")
    rows = list(zip(*[p["positions"] for p in loaded], strict=True))
    for group in rows:
        if len({r["fen"] for r in group}) != 1:
            parser.error("probes are not the same positions in the same order")

    pool = Pool(arguments.engine, 64)
    started = time.perf_counter()
    done = 0
    lock = threading.Lock()
    print(f"{len(rows)} positions, {len(loaded)} builds, arm '{arguments.arm}', "
          f"{arguments.nodes:,} reference nodes per move")

    def work(group: tuple[dict[str, Any], ...]) -> dict[str, Any]:
        nonlocal done
        board = board_of(group[0])
        engine = pool.engine()
        moves: list[str] = []
        for row in group:
            for name in ("shipped", "floor", "equal_nodes", "budget"):
                if name in row and str(row[name]["move"]) not in moves:
                    moves.append(str(row[name]["move"]))
        scores = {
            uci: restricted(engine, board, chess.Move.from_uci(uci), arguments.nodes)
            for uci in moves if chess.Move.from_uci(uci) in board.legal_moves
        }
        best = max(scores.values(), key=rank) if scores else None
        with lock:
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(rows)}  {time.perf_counter() - started:.0f}s", flush=True)
        return {
            "fen": group[0]["fen"], "cluster": group[0]["cluster"],
            "stratum": group[0]["stratum"], "phase": group[0]["phase"],
            "scores": scores,
            "best": max(scores, key=lambda u: rank(scores[u])) if scores else None,
            "moves": {name: str(row[arguments.arm]["move"])
                      for name, row in zip(arguments.name, group, strict=True)},
            "nodes": {name: int(row[arguments.arm]["nodes"])
                      for name, row in zip(arguments.name, group, strict=True)},
            "seconds": {name: float(row[arguments.arm]["seconds"])
                        for name, row in zip(arguments.name, group, strict=True)},
            "regret": {
                name: regret(best, scores[str(row[arguments.arm]["move"])])
                if best is not None and str(row[arguments.arm]["move"]) in scores else None
                for name, row in zip(arguments.name, group, strict=True)
            },
        }

    with ThreadPoolExecutor(max_workers=arguments.workers) as runner:
        judged = list(runner.map(work, rows))
    pool.close()

    report(arguments.name, judged)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "provenance": {"probes": [str(p) for p in arguments.probe], "names": arguments.name,
                       "arm": arguments.arm, "engine": arguments.engine,
                       "nodes": arguments.nodes, "severe_cp": SEVERE_CP,
                       "sources": [p["provenance"] for p in loaded],
                       "seconds": round(time.perf_counter() - started, 1)},
        "positions": judged,
    }, indent=1) + "\n")
    print(f"wrote {arguments.out}")
    return 0


def report(names: list[str], judged: list[dict[str, Any]]) -> None:
    print(f"\n{'build':<14} {'best':>6} {'<50cp':>6} {'severe':>7} {'mate class':>11} "
          f"{'mean cp':>9} {'95% over clusters':>20} {'kn':>8} {'s':>7}")
    for name in names:
        best = close = severe = mates = 0
        cp_rows: list[dict[str, Any]] = []
        nodes = 0
        seconds = 0.0
        for row in judged:
            hurt = row["regret"].get(name)
            nodes += int(row["nodes"][name])
            seconds += float(row["seconds"][name])
            if hurt is None:
                continue
            if hurt["klass"] in ("missed_mate", "into_mate", "slower_mate"):
                mates += 1
                continue
            amount = int(hurt["cp"] or 0)
            cp_rows.append({**row, "_cp": amount})
            if amount == 0:
                best += 1
            elif amount < SEVERE_CP:
                close += 1
            else:
                severe += 1
        mean, low, high = cluster_bootstrap(cp_rows)
        print(f"{name:<14} {best:>6} {close:>6} {severe:>7} {mates:>11} {mean:>8.1f}c "
              f"[{low:>7.1f}, {high:>7.1f}] {nodes / 1000:>8.0f} {seconds:>7.1f}")
    print("\nSame positions, same reference searches, same candidate set. 'mate class' is "
          "counted, never averaged into the centipawn mean.")
    for key in ("phase", "stratum"):
        groups = sorted({str(row[key]) for row in judged})
        print(f"\nmean centipawn regret by {key}:")
        header = "  " + " " * 14 + "".join(f"{g:>14}" for g in groups)
        print(header)
        for name in names:
            cells = []
            for group in groups:
                amounts = [int(row["regret"][name]["cp"] or 0) for row in judged
                           if str(row[key]) == group and row["regret"].get(name)
                           and row["regret"][name]["klass"] in ("best", "cp")]
                cells.append(f"{statistics.fmean(amounts):>13.1f}c" if amounts else f"{'-':>14}")
            print(f"  {name:<14}" + "".join(cells))


if __name__ == "__main__":
    raise SystemExit(main())
