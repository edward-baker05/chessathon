"""Score networks against the fresh holdout, on the things that decide games.

The trainer's own holdout loss is a sigmoid mean-squared error over a tail of the training
file. It is the right thing to optimise and the wrong thing to compare architectures on:
it is drawn from the same distribution the weights were fitted to, and it says nothing
about whether the net orders moves the way a strong reference does, which is what a search
actually asks of an evaluation.

This reports four things per network.

**Centipawn error** against the reference label, mean absolute and in the sigmoid space the
loss is written in, with tactically unstable positions counted separately. A deep search
score is not automatically a sound static target and a mean taken over positions that are
really tactics measures the tactics.

**Ranking.** Every position carries the reference's top few moves with their scores. The
net evaluates the position after each of them and is scored on whether it puts the
reference's best move first, and on pairwise agreement over the rest. A net can halve its
centipawn error and order candidate moves worse.

**By phase and by texture.** A net that is fine on average and hopeless in rook endings
does not get to hide behind the average.

**Float against quantised.** The shipped arithmetic is int16, and it is only the same
function as the trained one if nothing overflowed and nothing was clamped away. The
quantised numbers come from the engine's own `nnue` module in a subprocess, not from a
second implementation written here that could agree with the trainer and disagree with the
thing that plays.

    uv run python tools/evalnet.py --holdout run1/holdout.json --out run1/evalnet.json \\
        --checkpoint control/epoch010.pt --checkpoint kb4/epoch010.pt --npz weights/net.npz
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools import dataset, freshset  # noqa: E402
from tools.checkpoint import load_checkpoint  # noqa: E402
from tools.train import SCALE, Network  # noqa: E402

RUNTIME = ROOT / "tools" / "net_eval_runtime.py"


def float_evaluations(checkpoint: Path, fens: list[str], batch: int = 4096) -> list[float]:
    """Centipawns from a training checkpoint, from the side to move's point of view."""
    blob = load_checkpoint(checkpoint)
    king_buckets = int(blob.get("king_buckets", 1))
    model = Network(int(blob["l1"]), int(blob["buckets"]), king_buckets)
    model.load_state_dict(blob["state"])
    model.eval()

    out: list[float] = []
    for start in range(0, len(fens), batch):
        chunk = fens[start : start + batch]
        records = np.frombuffer(
            b"".join(dataset.from_board(chess.Board(fen)) for fen in chunk), dtype=np.uint8
        ).reshape(-1, dataset.RECORD)
        index, white, black, stm, _score = dataset.unpack(records, king_buckets)
        counts = np.bincount(index, minlength=len(chunk))
        offsets = np.zeros(len(chunk), dtype=np.int64)
        np.cumsum(counts[:-1], out=offsets[1:])
        bucket = np.clip((counts - 2) // ((32 - 2) // int(blob["buckets"]) + 1),
                         0, int(blob["buckets"]) - 1)
        with torch.no_grad():
            value = model(
                torch.from_numpy(white), torch.from_numpy(black), torch.from_numpy(offsets),
                torch.from_numpy(stm), torch.from_numpy(bucket),
            )
        out.extend(float(v) * SCALE for v in value)
    return out


def quantised_evaluations(npz: Path, fens: list[str]) -> tuple[list[float], float, float]:
    """Centipawns from the engine's own runtime, plus its import cost and peak memory."""
    payload = json.dumps(fens)
    finished = subprocess.run(
        [sys.executable, str(RUNTIME)], input=payload, cwd=ROOT, check=True,
        capture_output=True, text=True,
        env={**os.environ, "CHESSATHON_WEIGHTS": str(npz)},
    )
    result = json.loads(finished.stdout.strip().splitlines()[-1])
    return ([float(v) for v in result["cp"]], float(result["import_seconds"]),
            float(result["peak_mb"]))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values / SCALE))


def measure(rows: list[dict[str, Any]], owned: list[list[int]],
            evaluations: list[float]) -> dict[str, Any]:
    """Error and ranking for one network over the whole holdout."""
    labels = np.array([float(row["cp"]) for row in rows])
    predicted = np.array([evaluations[mine[0]] for mine in owned])
    stable = np.array([not row["unstable"] for row in rows])
    # A mate is stored saturated, at the same sentinel the training labels use, so that a
    # net trained on that convention is scored on it. It is still not a centipawn quantity:
    # a net that says +900 in a position labelled +12800 contributes 11,900 to a centipawn
    # mean and swamps everything else in it. Every centipawn figure below is over the
    # non-mate rows only, and the mate rows are reported as their own line.
    scored = np.array([row["mate"] is None for row in rows])

    top1 = 0
    pairs_right = 0
    pairs_total = 0
    ranked_rows = 0
    for row, mine in zip(rows, owned, strict=True):
        entries = row["ranked"]
        if len(entries) < 2:
            continue
        ranked_rows += 1
        # The reference scores the move from the mover's side; after the move it is the
        # opponent to move, so the net's score of the child is negated to compare.
        ours = [-evaluations[slot] for slot in mine[1:]]
        theirs = [float(entry["cp"]) for entry in entries]
        if int(np.argmax(ours)) == int(np.argmax(theirs)):
            top1 += 1
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if theirs[i] == theirs[j]:
                    continue
                pairs_total += 1
                if (ours[i] > ours[j]) == (theirs[i] > theirs[j]):
                    pairs_right += 1

    def errors(mask: np.ndarray, mates_out: bool = True) -> dict[str, float]:
        if mates_out:
            mask = mask & scored
        if not mask.any():
            return {"positions": 0, "mae_cp": 0.0, "rmse_cp": 0.0, "sigmoid_mse": 0.0}
        gap = predicted[mask] - labels[mask]
        return {
            "positions": int(mask.sum()),
            "mae_cp": float(np.abs(gap).mean()),
            "rmse_cp": float(np.sqrt((gap**2).mean())),
            "sigmoid_mse": float(
                ((sigmoid(predicted[mask]) - sigmoid(labels[mask])) ** 2).mean()
            ),
        }

    groups: dict[str, dict[str, float]] = {}
    for key in ("phase",):
        for name in sorted({str(row[key]) for row in rows}):
            groups[name] = errors(np.array([str(row[key]) == name for row in rows]))
    tags = {str(tag) for row in rows for tag in row["tags"]}
    for tag in sorted(tags):
        groups[tag] = errors(np.array([tag in row["tags"] for row in rows]))

    return {
        "all": errors(np.ones(len(rows), dtype=bool)),
        "stable": errors(stable),
        "unstable": errors(~stable),
        # Mate-labelled rows, kept apart. The sigmoid figure is the only one that means
        # anything here, because the target is saturated by construction.
        "mate_labelled": errors(~scored, mates_out=False),
        "top1": {"right": top1, "of": ranked_rows,
                 "rate": top1 / max(ranked_rows, 1)},
        "pairs": {"right": pairs_right, "of": pairs_total,
                  "rate": pairs_right / max(pairs_total, 1)},
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--npz", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.out.exists():
        parser.error(f"{arguments.out} exists; evidence is never overwritten, pick a new name")
    holdout = json.loads(arguments.holdout.read_text())
    rows = holdout["positions"][: arguments.limit or None]
    # The same boards, in the same order, that tools/train.py scores itself on each epoch.
    built = freshset.load(arguments.holdout, arguments.limit)
    fens, owned = built.fens, built.owners
    print(f"{len(rows)} holdout positions, {len(fens)} boards to evaluate")

    results: dict[str, Any] = {}
    for checkpoint in arguments.checkpoint:
        started = time.perf_counter()
        values = float_evaluations(checkpoint, fens)
        results[f"float:{checkpoint}"] = measure(rows, owned, values)
        results[f"float:{checkpoint}"]["seconds"] = round(time.perf_counter() - started, 1)
        report(f"float {checkpoint}", results[f"float:{checkpoint}"])
    for npz in arguments.npz:
        values, import_seconds, peak = quantised_evaluations(npz, fens)
        results[f"quantised:{npz}"] = measure(rows, owned, values)
        results[f"quantised:{npz}"]["import_seconds"] = import_seconds
        results[f"quantised:{npz}"]["peak_mb"] = peak
        report(f"quantised {npz}", results[f"quantised:{npz}"])
        print(f"  import {import_seconds:.1f}s, peak {peak:.0f} MB")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "provenance": {"holdout": str(arguments.holdout),
                       "holdout_provenance": holdout["provenance"],
                       "positions": len(rows), "boards": len(fens)},
        "results": results,
    }, indent=1) + "\n")
    print(f"wrote {arguments.out}")
    return 0


def report(name: str, result: dict[str, Any]) -> None:
    print(f"\n{name}")
    for key in ("all", "stable", "unstable", "mate_labelled"):
        row = result[key]
        print(f"  {key:<9} {row['positions']:>6} positions   MAE {row['mae_cp']:>7.1f}cp   "
              f"RMSE {row['rmse_cp']:>7.1f}cp   sigmoid MSE {row['sigmoid_mse']:.5f}")
    print("  centipawn figures exclude mate-labelled positions; the mate line is those.")
    print(f"  top-1 move {result['top1']['right']}/{result['top1']['of']} "
          f"({result['top1']['rate']:.1%}), pairwise {result['pairs']['rate']:.1%} "
          f"of {result['pairs']['of']:,}")
    interesting = [(n, g) for n, g in result["groups"].items() if g["positions"] >= 50]
    for group_name, group in sorted(interesting, key=lambda kv: -kv[1]["mae_cp"])[:8]:
        print(f"    {group_name:<14} {group['positions']:>6}  MAE {group['mae_cp']:>7.1f}cp")


if __name__ == "__main__":
    raise SystemExit(main())
