"""Whole-search check for board-only changes. Same nodes, moves and scores required."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import chess

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--engine", required=True)
parser.add_argument("--name", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.engine).resolve()))
start = time.perf_counter()
import search  # noqa: E402
import tt  # noqa: E402

load_seconds = time.perf_counter() - start
positions = json.loads((OUT.parent / "positions.json").read_text())["positions"]
# Fixed, phase-spanning subsample; every pass repeats the same positions.
positions = positions[::3]
result = {"engine": args.engine, "import_seconds": load_seconds, "positions": []}
for row in positions:
    runs = []
    for _ in range(3):
        tt.tt_clear(search.WORK.table)
        search.clear_tables()
        search.set_game_history([])
        before = time.perf_counter()
        cpu_before = time.process_time()
        move = search.think(chess.Board(row["fen"]), 3600000, increment_ms=0, node_limit=1048576)
        cpu = time.process_time() - cpu_before
        elapsed = time.perf_counter() - before
        depth, seldepth, score, nodes, _ = search.last_search()
        runs.append(
            {
                "move": move,
                "depth": depth,
                "seldepth": seldepth,
                "score": score,
                "nodes": nodes,
                "seconds": elapsed,
                "cpu_seconds": cpu,
            }
        )
    result["positions"].append({**row, "runs": runs})
    (OUT / f"{args.name}-search.json").write_text(json.dumps(result, indent=2) + "\n")
    print(row["id"], round(statistics.median(x["seconds"] for x in runs), 3), flush=True)
