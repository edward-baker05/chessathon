"""What each pruning margin is actually worth, measured without playing a game.

The six margins in `search.py` were tuned against a material evaluation. Five of them read
the static evaluation, so a network changes what they mean: the same centipawn scale, but a
different distribution of scores across it. The two SEE margins compare a pure material
exchange score against depth and cannot have been affected, which is worth knowing before
spending games on them.

Node counts here are counts of positions entered, one per position. Do not compare a
number from this tool against one produced before the double-counted quiescence handover
was fixed; they are different units.

The measurement is node counts, not counters inside the search. `search.py` reads every
margin from the environment and numba bakes the value into the jitted code at import, so a
margin can be disabled by setting it out of reach and re-importing. Nodes with a margin
disabled, against nodes with everything at its default, is exactly the work that margin is
doing. No instrumented fork of the engine, and nothing added to the code that ships.

Each setting needs its own process, because the value is frozen at import. That is what the
subprocess below is for, and it is also why `--engine` can point at a snapshot directory:
the same positions under two different networks say whether the distribution really moved.

    uv run python tools/margins.py --engine snapshots/halfka-v3-e18 --depth 8
    uv run python tools/margins.py --engine snapshots/net768 --depth 8

Not shipped: tools/ never reaches the zip.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Large enough that no margin can fire, small enough that `margin * depth` stays far inside
# int32 once numba has typed the expression. Evaluations never leave +/- 32000, so 100000
# per ply puts every one of these predicates permanently out of reach.
DISABLED = 100_000

# The five that read the static evaluation, and so are the ones a network can invalidate.
# FUTILITY is two constants in one predicate and is disabled by lifting the base, which is
# why FUTILITY_MARGIN is not listed separately: on its own it cannot switch the branch off.
EVAL_MARGINS = (
    ("RFP_MARGIN", "reverse futility", "static - M*depth >= beta"),
    ("RAZOR_MARGIN", "razoring", "static + M*depth < alpha"),
    ("FUTILITY_BASE", "futility", "static + M + 90*depth <= alpha"),
    ("DELTA_MARGIN", "delta (in qsearch)", "stand_pat + gain + M < alpha"),
)
# Listed so a run reports them, and reports that they did not move. `see` is material only.
SEE_MARGINS = (
    ("SEE_QUIET_MARGIN", "SEE, quiet moves", "see(move) < -M*depth"),
    ("SEE_CAPTURE_MARGIN", "SEE, captures", "see(move) < -M*depth"),
)

# The worker. Run as `python -c`, in a fresh process so that the margins in the environment
# are the ones numba compiles in. It prints one JSON object so the parent does not have to
# parse an engine's chatter.
# Every measurement starts from an empty engine. `clear_tables()` does not touch the
# transposition table, which is deliberate in play and wrong here: without `tt_clear` each
# position inherits the previous one's entries, and the move test inherits the work of the
# node count that ran on the same position immediately before it. Neither run is then the
# independent fixed-budget measurement it is reported as.
WORKER = """
import json, sys
sys.path.insert(0, {engine!r})
import chess, search, tt

def fresh():
    tt.tt_clear(search.WORK.table)
    search.clear_tables()
    search.set_game_history([])

total = 0
moves = []
for line in json.load(open({fens!r})):
    board = chess.Board(line)
    fresh()
    search.search_value(board, {depth})
    total += search.nodes()
    fresh()
    moves.append(search.think(board, 3_600_000, node_limit={nodes}))
print("RESULT" + json.dumps({{"nodes": total, "moves": moves}}))
"""


def measure(engine: Path, fens: Path, depth: int, nodes: int,
            overrides: dict[str, int]) -> tuple[int, list[str]]:
    """Total nodes and the move chosen in each position, under one margin setting."""
    environment = dict(os.environ)
    for name, value in overrides.items():
        environment[f"CHESS_{name}"] = str(value)
    source = WORKER.format(engine=str(engine), fens=str(fens), depth=depth, nodes=nodes)
    finished = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, env=environment,
        cwd=str(engine),
    )
    marker = finished.stdout.rfind("RESULT")
    if finished.returncode != 0 or marker < 0:
        raise RuntimeError(f"worker failed:\n{finished.stdout[-2000:]}\n{finished.stderr[-2000:]}")
    payload = json.loads(finished.stdout[marker + len("RESULT"):].splitlines()[0])
    return int(payload["nodes"]), list(payload["moves"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT,
                        help="directory holding the engine and its weights")
    parser.add_argument("--fens", type=Path, required=True, help="JSON list of FEN strings")
    parser.add_argument("--depth", type=int, default=8, help="fixed depth for the node count")
    parser.add_argument("--nodes", type=int, default=200_000,
                        help="fixed node budget for the move each setting picks")
    arguments = parser.parse_args()

    engine = arguments.engine.resolve()
    fens = arguments.fens.resolve()
    count = len(json.loads(fens.read_text()))
    print(f"{engine.name}: {count} positions, depth {arguments.depth}\n")

    baseline_nodes, baseline_moves = measure(engine, fens, arguments.depth, arguments.nodes, {})
    print(f"baseline (every margin at its default): {baseline_nodes:,} nodes\n")

    print(f"{'margin':<20} {'nodes if off':>14} {'saved':>8} {'moves changed':>14}   predicate")
    print("-" * 96)
    for name, label, predicate in EVAL_MARGINS + SEE_MARGINS:
        off_nodes, off_moves = measure(
            engine, fens, arguments.depth, arguments.nodes, {name: DISABLED}
        )
        # What the margin is worth: the nodes it removes from the tree that would otherwise
        # be searched. Reported against the disabled run, so it reads as a share of the
        # work avoided rather than as a share of the work still done.
        saved = 1.0 - baseline_nodes / off_nodes if off_nodes else 0.0
        changed = sum(1 for a, b in zip(baseline_moves, off_moves, strict=True) if a != b)
        print(f"{label:<20} {off_nodes:>14,} {saved:>7.1%} {changed:>10}/{count}   {predicate}")

    print(
        "\nA margin that saves almost nothing is not pruning under this evaluation, whatever\n"
        "it was worth under the one it was tuned against. A margin that saves a great deal\n"
        "and changes many moves is cutting lines the search wanted; both are worth a game\n"
        "level A/B, and nothing else here is."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
