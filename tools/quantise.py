"""Convert a training checkpoint into the int16 network the engine ships.

The float and quantised networks are the same function by construction, so this is a scale
and a round rather than a calibration. What this tool actually adds is a proof.

`nnue._dot` accumulates the output sum in int32, because int64 halves the vector width and
measured six times slower. int32 can overflow: at QA 255 the worst case over 1024 terms is
9.3e9 against an int32 maximum of 2.1e9. Rather than pick a scale that looks safe, this
computes the exact worst case from the weights it is about to write and refuses to write a
file it cannot prove. If the largest scale overflows it steps down until one fits, and says
which it used. The engine reads QA back out of the file, so nothing has to agree by
convention.

Not shipped: tools/ never reaches the zip.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Largest first. QA is the input and activation scale, so a bigger one quantises the
# accumulator more finely; it is reduced only when the overflow proof demands it.
QA_CANDIDATES = (255, 181, 127, 96, 64)
QB = 64
SCALE = 400

INT32_MAX = 2**31 - 1
INT16_MAX = 2**15 - 1
# Both kings, plus at most thirty other pieces, is the most features that can ever be
# active in one accumulator.
MAX_ACTIVE_FEATURES = 32


def output_bound(out_weight: np.ndarray, qa: int) -> int:
    """Largest magnitude `nnue._dot` can reach, over every bucket.

    Every activation saturated at QA, every product agreeing in sign. Unreachable on a
    real board, but "unreachable in practice" is how an engine loses a game at move 60.
    """
    return int(qa * qa * np.abs(out_weight.astype(np.int64)).sum(axis=1).max())


def accumulator_bound(ft_weight: np.ndarray, ft_bias: np.ndarray) -> int:
    """Largest magnitude an int16 accumulator can reach.

    The worst case is the bias plus the largest feature weights that could be active at
    once. Taking the largest per neuron across all features is looser than any real
    position and is meant to be.
    """
    largest = np.abs(ft_weight.astype(np.int64)).max(axis=0)
    return int((np.abs(ft_bias.astype(np.int64)) + MAX_ACTIVE_FEATURES * largest).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "weights" / "net.npz")
    parser.add_argument("--qa", type=int, default=0, help="force a scale instead of choosing")
    arguments = parser.parse_args()

    blob = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    state = blob["state"]
    hidden = int(blob["l1"])
    buckets = int(blob["buckets"])
    print(f"{arguments.checkpoint}: L1 {hidden}, {buckets} buckets, epoch {blob.get('epoch')}, "
          f"holdout loss {blob.get('holdout_loss'):.6f}")

    ft = state["transformer.weight"].numpy()
    ft_bias_float = state["transformer_bias"].numpy()
    out = state["output"].numpy()
    out_bias_float = state["output_bias"].numpy()

    candidates = (arguments.qa,) if arguments.qa else QA_CANDIDATES
    for qa in candidates:
        ft_weight = np.rint(ft * qa).astype(np.int64)
        ft_bias = np.rint(ft_bias_float * qa).astype(np.int64)
        out_weight = np.rint(out * QB).astype(np.int64)
        out_bias = np.rint(out_bias_float * qa * QB).astype(np.int64)

        sum_bound = output_bound(out_weight, qa)
        acc_bound = accumulator_bound(ft_weight, ft_bias)
        weight_fits = int(np.abs(np.concatenate([ft_weight.ravel(), out_weight.ravel()])).max())
        ok = sum_bound <= INT32_MAX and acc_bound <= INT16_MAX and weight_fits <= INT16_MAX
        print(
            f"  QA {qa:>3}: output sum worst case {sum_bound:,} of {INT32_MAX:,} "
            f"({sum_bound / INT32_MAX:5.1%}), accumulator {acc_bound:,} of {INT16_MAX:,} "
            f"({acc_bound / INT16_MAX:5.1%})  {'ok' if ok else 'OVERFLOWS'}"
        )
        if ok:
            break
    else:
        print(
            "\nno scale in "
            f"{QA_CANDIDATES} keeps the arithmetic inside its types. The output weights "
            "are too large; lower OUT_CLAMP in tools/train.py and train again.",
            file=sys.stderr,
        )
        return 1

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.out,
        ft_weight=ft_weight.astype(np.int16),
        ft_bias=ft_bias.astype(np.int16),
        out_weight=out_weight.astype(np.int16),
        out_bias=out_bias.astype(np.int32),
        qa=np.int32(qa),
        qb=np.int32(QB),
        scale=np.int32(SCALE),
    )
    print(f"\n{arguments.out} ({arguments.out.stat().st_size:,} bytes)")

    # What rounding cost, measured rather than assumed. A large error here means the
    # clamps in tools/train.py are binding and the float net is not the net that ships.
    error = float(np.abs(ft * qa - ft_weight).mean())
    print(f"  mean input weight rounding error: {error:.4f} of a quantisation step")
    print(f"  clipped input weights:  {int((np.abs(ft_weight) == np.abs(ft_weight).max()).sum())}")
    print("\nverify with: uv run pytest tests/test_nnue.py -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
