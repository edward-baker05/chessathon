"""Summarize fresh exploratory measurements without turning centipawns into Elo."""

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def main() -> None:
    judged = json.loads((ROOT / "reference.json").read_text())
    runs = {
        p.stem.removeprefix("bench-"): json.loads(p.read_text()) for p in ROOT.glob("bench-*.json")
    }
    summary: dict[str, Any] = {"positions_judged": len(judged), "runs": {}}
    for name, run in runs.items():
        metrics = []
        for index in range(4):
            losses = []
            depths = []
            for record, reference in zip(run["positions"], judged, strict=True):
                assert record["id"] == reference["id"]
                choice = record["searches"][index]
                anchor = max(reference["choices"].values())
                losses.append(max(0, anchor - reference["choices"][choice["move"]]))
                depths.append(choice["depth"])
            metrics.append(
                {
                    "limit": run["positions"][0]["searches"][index]["limit"],
                    "mean_capped_1000cp_loss": statistics.mean(min(1000, x) for x in losses),
                    "median_loss": statistics.median(losses),
                    "loss_ge_100cp": sum(x >= 100 for x in losses),
                    "loss_ge_300cp": sum(x >= 300 for x in losses),
                    "median_depth": statistics.median(depths),
                    "losses": losses,
                }
            )
        summary["runs"][name] = metrics
    baseline = runs["baseline"]
    high = [p["searches"][-1] for p in baseline["positions"]]
    total = sum(r["nodes"] for r in high)
    summary["baseline_counters"] = {
        "reported_nodes": total,
        "qnode_fraction": sum(r["counters"][0] for r in high) / total,
        "post_beta_fraction": sum(r["counters"][1] for r in high) / total,
        "double_count_fraction": sum(r["counters"][5] for r in high) / total,
        "uncommitted_fraction": sum(r["nodes"] - r["counters"][4] for r in high) / total,
        "median_reported_nps": statistics.median(r["nodes"] / r["seconds"] for r in high),
    }
    summary["static_by_piece_count"] = {}
    import chess

    for low, high_count in ((2, 7), (8, 16), (17, 24), (25, 32)):
        rows = [
            (r["static"], ref["reference_score"])
            for r, ref in zip(baseline["positions"], judged, strict=True)
            if low <= chess.Board(r["fen"]).occupied.bit_count() <= high_count
        ]
        if rows:
            summary["static_by_piece_count"][f"{low}-{high_count}"] = {
                "n": len(rows),
                "mean_abs_disagreement_cp": statistics.mean(abs(a - b) for a, b in rows),
                "mean_signed_disagreement_cp": statistics.mean(a - b for a, b in rows),
            }
    match = json.loads((ROOT / "tracking-match.json").read_text())
    summary["tracking_pilot"] = {
        "games": len(match),
        "wins": sum(r["fixed_score"] == 1 for r in match),
        "draws": sum(r["fixed_score"] == 0.5 for r in match),
        "losses": sum(r["fixed_score"] == 0 for r in match),
        "resets_fixed": sum(r["resets_fixed_baseline"][0] for r in match),
        "resets_baseline": sum(r["resets_fixed_baseline"][1] for r in match),
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    printable = json.loads(json.dumps(summary))
    for metrics in printable["runs"].values():
        for metric in metrics:
            del metric["losses"]
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
