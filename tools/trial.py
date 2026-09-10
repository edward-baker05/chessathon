"""Try candidate search settings on the failures a reference has already confirmed.

`tools/attribute.py` says which mechanism removes a good move. This says whether a proposed
change puts it back, and it asks the same outside reference under the same restricted-move
conditions, so "better" means the reference scores the new move higher and not that the
move changed.

Each configuration is a set of `CHESS_*` overrides. numba bakes a module constant into the
jitted code at import, so a configuration cannot be switched inside a process: every one is
run in a subprocess of its own with that environment, which is the same thing a snapshot
does and is why a snapshot A/B and this agree.

Two budgets per position, because they answer different questions. A fixed depth says
whether the change makes the line reachable at all. A fixed node count says whether it is
still reachable once the change has been charged for the work it costs, which is the
comparison a game makes.

    uv run python tools/trial.py --classes run1/classes-cold.json --engine /path/to/sf \\
        --config baseline --config "endgame:CHESS_RFP_MIN_MEN=12" --out run1/trial.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from tools.judge import Pool, rank, restricted  # noqa: E402

BROKEN_CP = 50


def parse_config(text: str) -> tuple[str, dict[str, str]]:
    """`name` or `name:VAR=value,VAR=value`."""
    name, _, rest = text.partition(":")
    settings: dict[str, str] = {}
    for pair in filter(None, rest.split(",")):
        key, _, value = pair.partition("=")
        settings[key] = value
    return name, settings


def worker() -> int:
    """Search every position given on stdin with whatever this process was started with."""
    import search
    from tools.failures import board_of, fresh, history_keys, probe

    request = json.loads(sys.stdin.read())
    work = search.WORK
    out: list[dict[str, Any]] = []
    for row in request["positions"]:
        board = board_of(row)
        keys = history_keys(board)
        fresh(work, keys)
        fixed = probe(board, keys, None, work, depth=int(request["depth"]))
        fresh(work, keys)
        budget = probe(board, keys, None, work, depth=search.MAX_DEPTH,
                       node_limit=int(request["nodes"]))
        out.append({"id": row["id"], "depth_move": fixed["move"], "depth_nodes": fixed["nodes"],
                    "depth_seconds": fixed["seconds"], "budget_move": budget["move"],
                    "budget_depth": budget["depth"], "budget_seconds": budget["seconds"]})
    print(json.dumps({"rows": out}))
    return 0


def run_config(settings: dict[str, str], rows: list[dict[str, Any]], depth: int,
               nodes: int) -> dict[str, Any]:
    finished = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=json.dumps({"positions": rows, "depth": depth, "nodes": nodes}),
        cwd=ROOT, check=True, capture_output=True, text=True,
        env={**os.environ, **settings},
    )
    parsed: dict[str, Any] = json.loads(finished.stdout.strip().splitlines()[-1])
    return parsed


def verdict(scores: dict[str, dict[str, Any]], move: str) -> str:
    if move not in scores:
        return "unscored"
    best = max(scores.values(), key=rank)
    if rank(scores[move]) >= rank(best):
        return "best"
    if best["kind"] == "mate" or scores[move]["kind"] == "mate":
        return "mate lost"
    return "broken" if int(best["cp"]) - int(scores[move]["cp"]) >= BROKEN_CP else "close"


def main() -> int:
    if "--worker" in sys.argv:
        return worker()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--nodes", type=int, default=200_000)
    parser.add_argument("--reference-nodes", type=int, default=1_000_000)
    parser.add_argument("--only-attributable", action="store_true",
                        help="keep only the positions the additive floor itself gets right")
    parser.add_argument("--attribution", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.out.exists():
        parser.error(f"{arguments.out} exists; evidence is never overwritten, pick a new name")
    rows = json.loads(arguments.classes.read_text())["findings"]
    if arguments.only_attributable:
        if not arguments.attribution:
            parser.error("--only-attributable needs --attribution")
        attribution = json.loads(arguments.attribution.read_text())["positions"]
        usable = {str(r["id"]) for r in attribution if not r["broken"]["floor"]}
        rows = [row for row in rows if str(row["id"]) in usable]

    configs = [parse_config(text) for text in arguments.config]
    started = time.perf_counter()
    print(f"{len(rows)} positions, {len(configs)} configurations, depth {arguments.depth} "
          f"and {arguments.nodes:,} nodes")

    results: dict[str, dict[str, Any]] = {}
    for name, settings in configs:
        results[name] = run_config(settings, rows, arguments.depth, arguments.nodes)
        results[name]["settings"] = settings
        print(f"  searched {name}  {time.perf_counter() - started:.0f}s", flush=True)

    # One reference pass over every move any configuration produced, so the scores every
    # configuration is judged against are the same numbers from the same searches.
    pool = Pool(arguments.engine, 64)
    engine = pool.engine()
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        board = _with_history(row)
        moves: set[str] = {str(row["shipped"]), str(row["floor"]), str(row["reference_best"])}
        for name in results:
            entry = _by_id(results[name]["rows"], str(row["id"]))
            moves.add(str(entry["depth_move"]))
            moves.add(str(entry["budget_move"]))
        scored[str(row["id"])] = {
            uci: restricted(engine, board, chess.Move.from_uci(uci), arguments.reference_nodes)
            for uci in sorted(moves)
            if chess.Move.from_uci(uci) in board.legal_moves
        }
        if (index + 1) % 5 == 0:
            print(f"  scored {index + 1}/{len(rows)}  "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    pool.close()

    report(rows, results, scored)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "provenance": {"classes": str(arguments.classes), "engine": arguments.engine,
                       "depth": arguments.depth, "nodes": arguments.nodes,
                       "reference_nodes": arguments.reference_nodes,
                       "broken_cp": BROKEN_CP, "positions": len(rows),
                       "seconds": round(time.perf_counter() - started, 1)},
        "configs": {name: results[name] for name in results},
        "scores": scored,
    }, indent=1) + "\n")
    print(f"wrote {arguments.out}")
    return 0


def _with_history(row: dict[str, Any]) -> chess.Board:
    board = chess.Board(str(row["root_fen"]))
    for uci in str(row["history"]).split():
        board.push(chess.Move.from_uci(uci))
    return board


def _by_id(rows: list[dict[str, Any]], wanted: str) -> dict[str, Any]:
    for row in rows:
        if str(row["id"]) == wanted:
            return row
    raise KeyError(wanted)


def report(rows: list[dict[str, Any]], results: dict[str, dict[str, Any]],
           scored: dict[str, dict[str, dict[str, Any]]]) -> None:
    print(f"\n{'configuration':<22} {'at depth':>22}   {'at a node budget':>22}")
    print(f"{'':<22} {'best':>6} {'close':>6} {'bad':>8}   "
          f"{'best':>6} {'close':>6} {'bad':>8}   seconds")
    for name, result in results.items():
        counts = {"depth": {"best": 0, "close": 0, "bad": 0},
                  "budget": {"best": 0, "close": 0, "bad": 0}}
        seconds = 0.0
        for row in rows:
            entry = _by_id(result["rows"], str(row["id"]))
            seconds += float(entry["depth_seconds"])
            for which in ("depth", "budget"):
                got = verdict(scored[str(row["id"])], str(entry[f"{which}_move"]))
                key = got if got in ("best", "close") else "bad"
                counts[which][key] += 1
        print(f"{name:<22} {counts['depth']['best']:>6} {counts['depth']['close']:>6} "
              f"{counts['depth']['bad']:>8}   {counts['budget']['best']:>6} "
              f"{counts['budget']['close']:>6} {counts['budget']['bad']:>8}   {seconds:>7.1f}")
    print("\n'bad' is the reference putting the move at least "
          f"{BROKEN_CP}cp behind the best of the set, or losing a mate to it.")


if __name__ == "__main__":
    raise SystemExit(main())
