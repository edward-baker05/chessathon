"""What the judged positions say about where the strength is going.

`tools/judge.py` says how bad each shipped move was. This asks the question that decides
what to build: when the shipped move was bad, would turning selectivity off have helped?
The additive floor and the equal-node re-search are scored by the same reference and on the
same scale, so the three are comparable, and the answer separates two very different
diagnoses that a score gap cannot tell apart.

A position where the floor is judged better is one selectivity cost. A position where the
floor is judged no better, or worse, is one where the evaluation or the depth is the limit
and no pruning change will recover it. Reported separately, because the second kind is the
larger of the two here and reading them together would send the next change to the wrong
place.

    uv run python tools/classify.py --judged run1/judged-cold.json --out run1/classes.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from tools.failures import board_of  # noqa: E402
from tools.judge import rank  # noqa: E402

BAD = ("severe", "missed_mate", "into_mate", "slower_mate")


def loss_of(final: dict[str, Any], move: str) -> tuple[str, int | None]:
    """The reference's verdict on one move, as a class and a centipawn amount."""
    hurt = final["regret"].get(move)
    if hurt is None:
        return ("absent", None)
    return (str(hurt["klass"]), hurt["cp"])


def better(final: dict[str, Any], left: str, right: str) -> bool:
    """Is `left` strictly better than `right` by the reference's own scores?"""
    scores = final["scores"]
    if left not in scores or right not in scores:
        return False
    return rank(scores[left]) > rank(scores[right])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.out.exists():
        parser.error(f"{arguments.out} exists; evidence is never overwritten, pick a new name")
    judged = json.loads(arguments.judged.read_text())
    rows = judged["positions"]

    classes: Counter[str] = Counter()
    findings: list[dict[str, Any]] = []
    for row in rows:
        judgement = row["judgement"]
        verdict = str(judgement["verdict"])
        if verdict not in BAD:
            continue
        final = judgement["rounds"][-1]
        shipped = str(row["shipped"]["move"])
        floor = str(row["floor"]["move"])
        equal = str(row["equal_nodes"]["move"])
        board = board_of(row)
        move = chess.Move.from_uci(shipped)
        helped = better(final, floor, shipped)
        helped_equal = better(final, equal, shipped)
        if helped and helped_equal:
            klass = "selectivity, and more nodes also find it"
        elif helped:
            klass = "selectivity at equal depth only"
        elif helped_equal:
            klass = "more nodes, not selectivity"
        else:
            klass = "neither: evaluation or horizon"
        classes[klass] += 1
        findings.append({
            "id": row["id"], "fen": row["fen"], "root_fen": row["root_fen"],
            "history": row["history"], "cluster": row["cluster"],
            "stratum": row["stratum"], "phase": row["phase"],
            "verdict": verdict, "klass": klass,
            "resolved": judgement["resolved"],
            "reference_best": final["best"],
            "reference_move": judgement["reference_move"],
            "shipped": shipped, "shipped_loss": loss_of(final, shipped),
            "floor": floor, "floor_loss": loss_of(final, floor),
            "equal_nodes": equal, "equal_nodes_loss": loss_of(final, equal),
            "shipped_depth": row["shipped"]["depth"],
            "equal_nodes_depth": row["equal_nodes"]["depth"],
            "floor_nodes": row["floor"]["nodes"],
            "shipped_nodes": row["shipped"]["nodes"],
            "gives_check": board.gives_check(move) if move in board.legal_moves else None,
            "is_capture": board.is_capture(move) if move in board.legal_moves else None,
            "fired": row["shipped"]["fired"],
            "best_pv": final["scores"].get(final["best"], {}).get("pv", []),
        })

    print(f"{len(rows)} judged, {len(findings)} where the shipped move is a real error\n")
    for name, count in classes.most_common():
        print(f"  {count:>3}  {name}")
    print("\nby verdict and phase:")
    grid: Counter[tuple[str, str]] = Counter(
        (str(f["verdict"]), str(f["phase"])) for f in findings
    )
    for (verdict, phase), count in sorted(grid.items()):
        print(f"  {verdict:<12} {phase:<11} {count}")
    unresolved = [f for f in findings if not f["resolved"]]
    print(f"\nunresolved at the top reference budget: {len(unresolved)}")

    print("\nthe ones selectivity is responsible for:")
    for finding in findings:
        if not str(finding["klass"]).startswith("selectivity"):
            continue
        print(f"  {finding['id']} {finding['verdict']:<12} shipped {finding['shipped']} "
              f"{finding['shipped_loss']}  floor {finding['floor']} {finding['floor_loss']}")
        print(f"       {finding['fen']}")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "provenance": {"judged": str(arguments.judged),
                       "judge_provenance": judged["provenance"]},
        "classes": dict(classes),
        "findings": findings,
    }, indent=1) + "\n")
    print(f"\nwrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
