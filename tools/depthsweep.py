"""At what depth does a confirmed good move become reachable, with selectivity and without.

`tools/attribute.py` says which mechanism removes a good move at one depth. That is not the
same as saying the move is out of reach: a mechanism that costs two plies on a position
costs nothing at all once the budget is two plies larger, and a rated move on this machine
is seventy times the depth-6 search these diagnostics run at.

So this sweeps the depth instead of the mechanism set. For each confirmed failure it runs
the shipped search and the additive floor at every depth from 1 up, records the move each
picks, and then asks the reference about every distinct move once, under the same
restricted-move conditions as the original judgement. What comes out is the first depth at
which each build picks a move the reference is happy with, and the difference between those
two depths is what selectivity actually costs on that position, in the only unit that
transfers to a game.

    uv run python tools/depthsweep.py --classes run1/classes-cold.json \\
        --engine /path/to/stockfish --max-depth 12 --out run1/depthsweep.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

import search  # noqa: E402
from tools.failures import board_of, fresh, history_keys, probe  # noqa: E402
from tools.judge import Pool, rank, restricted  # noqa: E402

GOOD_CP = 50


def sweep(row: dict[str, Any], max_depth: int, work: search.Work,
          node_cap: int) -> dict[str, list[dict[str, Any]]]:
    board = board_of(row)
    keys = history_keys(board)
    out: dict[str, list[dict[str, Any]]] = {"shipped": [], "floor": []}
    for label, mechanisms in (("shipped", None), ("floor", ())):
        for depth in range(1, max_depth + 1):
            fresh(work, keys)
            result = probe(board, keys, mechanisms, work, depth=depth, node_limit=node_cap)
            out[label].append({"depth": depth, "move": result["move"],
                               "nodes": result["nodes"], "seconds": result["seconds"]})
            # The floor spends tens of times the nodes for the same depth. Without a cap a
            # single position can run for minutes and the sweep never finishes.
            if result["nodes"] >= node_cap:
                break
    return out


def first_good(rows: list[dict[str, Any]], scores: dict[str, dict[str, Any]]) -> int | None:
    """Shallowest depth whose move the reference is happy with, or None."""
    if not scores:
        return None
    best = max(scores.values(), key=rank)
    for row in rows:
        got = scores.get(str(row["move"]))
        if got is None:
            continue
        if rank(got) >= rank(best):
            return int(row["depth"])
        if best["kind"] == "cp" and got["kind"] == "cp" \
                and int(best["cp"]) - int(got["cp"]) < GOOD_CP:
            return int(row["depth"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--attribution", type=Path)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--node-cap", type=int, default=3_000_000)
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.out.exists():
        parser.error(f"{arguments.out} exists; evidence is never overwritten, pick a new name")
    findings = json.loads(arguments.classes.read_text())["findings"]
    if arguments.attribution:
        usable = {str(r["id"]) for r in json.loads(arguments.attribution.read_text())["positions"]
                  if not r["broken"]["floor"]}
        findings = [row for row in findings if str(row["id"]) in usable]

    work = search.WORK
    pool = Pool(arguments.engine, 64)
    engine = pool.engine()
    started = time.perf_counter()
    print(f"{len(findings)} positions, depths 1 to {arguments.max_depth}, "
          f"node cap {arguments.node_cap:,}")

    rows: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        curves = sweep(finding, arguments.max_depth, work, arguments.node_cap)
        board = board_of(finding)
        moves = {str(entry["move"]) for curve in curves.values() for entry in curve}
        moves.add(str(finding["reference_best"]))
        scores = {
            uci: restricted(engine, board, chess.Move.from_uci(uci), arguments.nodes)
            for uci in sorted(moves) if chess.Move.from_uci(uci) in board.legal_moves
        }
        rows.append({
            "id": finding["id"], "fen": finding["fen"], "phase": finding["phase"],
            "verdict": finding["verdict"], "curves": curves, "scores": scores,
            "shipped_first_good": first_good(curves["shipped"], scores),
            "floor_first_good": first_good(curves["floor"], scores),
        })
        print(f"  {index + 1}/{len(findings)}  {time.perf_counter() - started:.0f}s", flush=True)
    pool.close()

    report(rows, arguments.max_depth)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "provenance": {"classes": str(arguments.classes), "engine": arguments.engine,
                       "max_depth": arguments.max_depth, "node_cap": arguments.node_cap,
                       "nodes": arguments.nodes, "good_cp": GOOD_CP,
                       "seconds": round(time.perf_counter() - started, 1)},
        "positions": rows,
    }, indent=1) + "\n")
    print(f"wrote {arguments.out}")
    return 0


def report(rows: list[dict[str, Any]], max_depth: int) -> None:
    print(f"\n{'id':>6} {'phase':<11} {'verdict':<12} {'shipped':>8} {'floor':>7} "
          f"{'cost':>6}  fen")
    costs: list[int] = []
    never = 0
    for row in rows:
        shipped = row["shipped_first_good"]
        floor = row["floor_first_good"]
        if shipped is None:
            never += 1
        cost = (shipped - floor) if (shipped is not None and floor is not None) else None
        if cost is not None:
            costs.append(cost)
        print(f"{row['id']:>6} {row['phase']:<11} {row['verdict']:<12} "
              f"{shipped if shipped is not None else '>' + str(max_depth):>8} "
              f"{floor if floor is not None else '>' + str(max_depth):>7} "
              f"{cost if cost is not None else '-':>6}  {row['fen']}")
    reached = [row for row in rows if row["shipped_first_good"] is not None]
    print(f"\nthe shipped search reaches a move the reference is happy with in "
          f"{len(reached)} of {len(rows)} within depth {max_depth}; {never} never do")
    if reached:
        depths = [int(row["shipped_first_good"]) for row in reached]
        print(f"  when it does, median depth {statistics.median(depths):.0f}, "
              f"max {max(depths)}")
    if costs:
        print(f"  plies selectivity costs, where both get there: median "
              f"{statistics.median(costs):.0f}, mean {statistics.fmean(costs):.1f}, "
              f"max {max(costs)}")


if __name__ == "__main__":
    raise SystemExit(main())
