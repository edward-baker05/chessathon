"""Fixed-budget search benchmark. `make bench`.

Two numbers, and they answer different questions.

**Nodes to depth** is the search-efficiency number. It is deterministic: the same build on
the same position always searches the same tree, so a change that reorders moves better or
prunes more shows up immediately and without a single game. It is the right measurement for
move ordering, history, reductions and margins.

**Nodes per second** is the throughput number, and it is not deterministic. It moves with
CPU frequency, with whatever else is running, and with cache state. Treat a difference under
about 5% as noise unless several runs agree, and never compare a throughput figure taken on
one machine against one taken on another.

Neither is strength. A change that searches fewer nodes to the same depth can still play
worse, and a change that plays better can cost nodes. `tests/match.py` is the only thing
here that measures strength, and it costs hours rather than a minute.

Every position starts from an empty transposition table, empty history tables and an empty
repetition record. Without that, position N inherits position N-1 and the suite measures the
order the positions happen to be in.

    uv run python tests/bench.py                       # nodes to depth 11, then throughput
    uv run python tests/bench.py --depth 13
    uv run python tests/bench.py --nodes 1000000 --json bench.json

Node counts are counts of positions entered, one per position. Do not compare against a
figure produced before the double-counted quiescence handover was fixed: different unit.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

import search  # noqa: E402
import tt  # noqa: E402

# Eight positions, chosen to span the phases the engine actually has to play rather than to
# be hard. Two openings, three middlegames of different character, three endgames including
# one pawn ending and one where the only progress is a promotion race.
SUITE = (
    ("start", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("closed-centre", "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8"),
    ("sicilian-attack", "2rq1rk1/pp1bppbp/3p1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 12"),
    ("heavy-pieces", "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 13"),
    ("rook-endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("pawn-endgame", "8/5p2/4k3/8/3K4/8/5P2/8 w - - 0 40"),
    ("promotion-race", "8/8/4k3/8/2p5/8/B2P2KP/8 w - - 0 1"),
)


def fresh() -> None:
    """An empty engine. Every measurement starts here or it measures the one before it."""
    tt.tt_clear(search.WORK.table)
    search.clear_tables(search.WORK)
    search.set_game_history([], search.WORK)


def run(fen: str, depth: int, node_limit: int) -> dict[str, float | int | str]:
    fresh()
    board = chess.Board(fen)
    started = time.perf_counter()
    move = search.think(
        board, 3_600_000, increment_ms=0, node_limit=node_limit,
        max_depth=depth if depth else 127,
    )
    elapsed = time.perf_counter() - started
    reached, seldepth, score, nodes, _spent = search.last_search()
    return {
        "move": move, "depth": reached, "seldepth": seldepth, "score": score,
        "nodes": nodes, "seconds": elapsed, "knps": nodes / elapsed / 1000 if elapsed else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--depth", type=int, default=11,
                        help="fixed depth for the nodes-to-depth table")
    parser.add_argument("--nodes", type=int, default=400_000,
                        help="fixed node budget for the throughput table")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat the throughput pass; the median is reported")
    parser.add_argument("--json", type=Path, help="also write the raw numbers here")
    arguments = parser.parse_args()

    report: dict[str, object] = {
        "depth": arguments.depth,
        "nodes": arguments.nodes,
        "increment_ms": os.environ.get("CHESSATHON_INCREMENT_MS", "unset"),
        "margins": {
            name: getattr(search, name) for name in dir(search)
            if name.endswith(("_MARGIN", "_BASE", "_BREAK"))
        },
    }

    print(f"nodes to depth {arguments.depth}: deterministic, this is the one to A/B\n")
    print(f"{'position':<18} {'nodes':>12} {'depth':>6} {'score':>8}  move")
    print("-" * 60)
    fixed_depth = []
    for name, fen in SUITE:
        result = run(fen, arguments.depth, 0)
        fixed_depth.append({"name": name, "fen": fen, **result})
        print(f"{name:<18} {result['nodes']:>12,} {result['depth']:>6} "
              f"{result['score']:>+7}cp  {result['move']}")
    total = sum(int(row["nodes"]) for row in fixed_depth)
    print(f"{'TOTAL':<18} {total:>12,}")
    report["fixed_depth"] = fixed_depth

    print(f"\nthroughput at {arguments.nodes:,} nodes: noisy, do not A/B on this\n")
    print(f"{'position':<18} {'nodes':>12} {'seconds':>9} {'knps':>9}")
    print("-" * 52)
    throughput = []
    for name, fen in SUITE:
        runs = [run(fen, 0, arguments.nodes) for _ in range(arguments.repeat)]
        best = sorted(runs, key=lambda row: float(row["knps"]))[len(runs) // 2]
        throughput.append({"name": name, "fen": fen, **best})
        print(f"{name:<18} {best['nodes']:>12,} {best['seconds']:>8.2f}s "
              f"{best['knps']:>8.0f}")
    counted = sum(int(row["nodes"]) for row in throughput)
    overall = counted / sum(float(row["seconds"]) for row in throughput)
    median = statistics.median(float(r["knps"]) for r in throughput)
    print(f"{'OVERALL':<18} {'':>12} {'':>9} {overall / 1000:>8.0f}")
    print(f"\nmedian {median:,.0f} knps per position, {overall / 1000:,.0f} knps overall")
    report["throughput"] = throughput

    if arguments.json:
        arguments.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
