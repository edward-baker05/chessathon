"""Which mechanism, or which pair of them, removes an independently confirmed good move.

`tools/selectivity.py` does this for one composed fixture at a time. This does it across
every position an outside reference has already judged the shipped move wrong in, which is
the difference between a fixture that can be fixed and a pattern worth changing the search
for.

Both directions, because one of them cannot answer the question.

**Subtractive**, everything on and one mechanism off, finds a mechanism that is solely
responsible. It cannot clear pruning: if two mechanisms would each remove the same line,
turning either off changes nothing and every row looks innocent.

**Additive**, everything off and one mechanism on, then pairs among the ones that were
clean alone, finds the mechanisms that are sufficient. Everything off still keeps the check
extension and the transposition policy, so the starting point is the real search minus
selectivity rather than plain alpha-beta.

Every move any variant selects is then scored by the same reference under the same
restricted-move conditions as the original judgement, so "this mechanism broke it" means
the reference says the move got worse and not that the move changed.

    uv run python tools/attribute.py --classes run1/classes-cold.json \\
        --engine /path/to/stockfish --out run1/attribution.json
"""

import argparse
import itertools
import json
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

# How far below the best move the reference has to put a variant's choice before the
# variant counts as having broken the position. Matches tools/judge.py's escalation floor.
BROKEN_CP = 50


def variants() -> list[tuple[str, tuple[str, ...] | None]]:
    """Shipped, floor, each mechanism removed, each mechanism alone."""
    rows: list[tuple[str, tuple[str, ...] | None]] = [("all", None), ("floor", ())]
    rows += [(f"-{name}", tuple(m for m in search.MECHANISMS if m != name))
             for name in search.MECHANISMS]
    rows += [(f"+{name}", (name,)) for name in search.MECHANISMS]
    return rows


def run_variants(row: dict[str, Any], depth: int, names: list[tuple[str, tuple[str, ...] | None]],
                 work: search.Work) -> dict[str, str]:
    board = board_of(row)
    keys = history_keys(board)
    out: dict[str, str] = {}
    for label, mechanisms in names:
        fresh(work, keys)
        out[label] = str(probe(board, keys, mechanisms, work, depth=depth)["move"])
    return out


def score_moves(row: dict[str, Any], moves: set[str], pool: Pool,
                nodes: int) -> dict[str, dict[str, Any]]:
    board = board_of(row)
    engine = pool.engine()
    scores: dict[str, dict[str, Any]] = {}
    for uci in sorted(moves):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            continue
        scores[uci] = restricted(engine, board, move, nodes)
    return scores


def broken(scores: dict[str, dict[str, Any]], move: str) -> bool:
    """Does the reference put this move materially behind the best of the set?"""
    if move not in scores:
        return False
    best = max(scores.values(), key=rank)
    if rank(scores[move]) >= rank(best):
        return False
    if best["kind"] == "mate" or scores[move]["kind"] == "mate":
        return True
    return int(best["cp"]) - int(scores[move]["cp"]) >= BROKEN_CP


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=1,
                        help="also try pairs of mechanisms that were clean alone")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.out.exists():
        parser.error(f"{arguments.out} exists; evidence is never overwritten, pick a new name")
    loaded = json.loads(arguments.classes.read_text())
    findings = loaded["findings"]
    pool = Pool(arguments.engine, arguments.hash)
    work = search.WORK
    base = variants()
    started = time.perf_counter()
    print(f"{len(findings)} confirmed failures, depth {arguments.depth}, "
          f"{arguments.nodes:,} reference nodes per move")

    rows: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        chosen = run_variants(finding, arguments.depth, base, work)
        # Pairs are only informative among the mechanisms that were clean on their own:
        # if one already breaks the position, adding a second says nothing new.
        pair_labels: list[tuple[str, tuple[str, ...] | None]] = []
        if arguments.pairs:
            scores = score_moves(finding, set(chosen.values()), pool, arguments.nodes)
            clean = [name for name in search.MECHANISMS
                     if not broken(scores, chosen[f"+{name}"])]
            pair_labels = [(f"+{a}+{b}", (a, b))
                           for a, b in itertools.combinations(clean, 2)]
            chosen.update(run_variants(finding, arguments.depth, pair_labels, work))
        scores = score_moves(finding, set(chosen.values()), pool, arguments.nodes)
        rows.append({
            "id": finding["id"], "fen": finding["fen"], "verdict": finding["verdict"],
            "klass": finding["klass"], "phase": finding["phase"],
            "chosen": chosen,
            "scores": scores,
            "broken": {label: broken(scores, move) for label, move in chosen.items()},
        })
        print(f"  {index + 1}/{len(findings)}  {time.perf_counter() - started:.0f}s", flush=True)
    pool.close()

    report(rows)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "provenance": {"classes": str(arguments.classes), "engine": arguments.engine,
                       "depth": arguments.depth, "nodes": arguments.nodes,
                       "broken_cp": BROKEN_CP,
                       "seconds": round(time.perf_counter() - started, 1)},
        "positions": rows,
    }, indent=1) + "\n")
    print(f"wrote {arguments.out}")
    return 0


def report(rows: list[dict[str, Any]]) -> None:
    usable = [row for row in rows if not row["broken"]["floor"]]
    print(f"\n{len(usable)} of {len(rows)} positions the floor itself gets right, so only "
          f"those can attribute anything\n")
    if not usable:
        return
    print("added alone to the floor, this many of those positions break:")
    for name in search.MECHANISMS:
        count = sum(1 for row in usable if row["broken"].get(f"+{name}"))
        print(f"  +{name:<9} {count:>3} of {len(usable)}")
    print("\nremoved alone from the full search, this many stop being broken:")
    for name in search.MECHANISMS:
        count = sum(1 for row in usable
                    if row["broken"].get("all") and not row["broken"].get(f"-{name}"))
        print(f"  -{name:<9} {count:>3}")
    pairs: dict[str, int] = {}
    for row in usable:
        for label, is_broken in row["broken"].items():
            if label.count("+") == 2 and is_broken:
                pairs[label] = pairs.get(label, 0) + 1
    if pairs:
        print("\npairs that break it where neither member does alone:")
        for label, count in sorted(pairs.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {label:<22} {count:>3}")
    def alone(row: dict[str, Any]) -> bool:
        return any(row["broken"].get(f"+{name}") for name in search.MECHANISMS)

    def only_pair(row: dict[str, Any]) -> bool:
        return not alone(row) and any(
            broke for label, broke in row["broken"].items() if label.count("+") == 2
        )

    print(f"\npositions at least one mechanism breaks on its own: "
          f"{sum(1 for row in usable if alone(row))} of {len(usable)}")
    print(f"positions only a pair breaks:                       "
          f"{sum(1 for row in usable if only_pair(row))}")


if __name__ == "__main__":
    raise SystemExit(main())
