"""Benchmark: node rate, depth reached, and the import budget.

Run with `make bench`. Not a pytest module: it is a stopwatch, and its numbers are the
before/after evidence for any change that claims to make the engine faster or deeper.
"""

import subprocess
import sys
import time
from pathlib import Path

# Run directly (`make bench`), not under pytest, so the repo root is not already on the
# path the way pytest's pythonpath setting puts it there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

POSITIONS = [
    ("startpos", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("midgame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("tactical", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
]

# The platform allows 90 s to import the agent. Fail well below that so drift is caught
# here rather than as a rejected upload.
IMPORT_BUDGET_S = 30.0
NODES_PER_POSITION = 400_000


def measure_import() -> float:
    """Import cost in a fresh process, which is the only measurement that means anything."""
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "-c", "import agent"],
        check=True,
        capture_output=True,
    )
    return time.perf_counter() - started


def main() -> int:
    import_seconds = measure_import()
    print(f"import (fresh process): {import_seconds:.2f}s of the platform's 90s budget")

    import search

    total_nodes = 0
    total_seconds = 0.0
    print(f"\n{'position':<10} {'nodes':>10} {'time':>8} {'knps':>8}  move")
    for name, fen in POSITIONS:
        board = chess.Board(fen)
        started = time.perf_counter()
        move = search.think(
            board, time_left_ms=600_000, node_limit=NODES_PER_POSITION, max_depth=127
        )
        elapsed = time.perf_counter() - started
        nodes = search.nodes()
        total_nodes += nodes
        total_seconds += elapsed
        print(f"{name:<10} {nodes:>10,} {elapsed:>7.2f}s {nodes / elapsed / 1000:>7.0f}  {move}")

    print(f"\ntotal: {total_nodes:,} nodes in {total_seconds:.2f}s")
    print(f"search rate: {total_nodes / total_seconds / 1000:.0f} knps")

    if import_seconds > IMPORT_BUDGET_S:
        print(
            f"\nFAIL: import takes {import_seconds:.1f}s, over the {IMPORT_BUDGET_S:.0f}s "
            f"guard. The platform allows 90s, so this is drift worth investigating now."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
