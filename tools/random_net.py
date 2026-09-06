"""Write a randomly initialised network in the shipped format.

This exists so the runtime can be built and tested before any training has happened. A
random net plays badly by construction. What it proves is that the accumulator, the
forward pass and the search agree with each other and run at the speed the design
predicted, which is the part that has to be right before training is worth starting.

`tools/quantise.py` writes the real thing in the same format.

Not shipped: harness/package.py globs root *.py and weights/, so tools/ never reaches the
zip.
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1", type=int, default=512)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--qa", type=int, default=181)
    parser.add_argument("--qb", type=int, default=64)
    parser.add_argument("--scale", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "weights" / "net.npz")
    arguments = parser.parse_args()

    rng = np.random.default_rng(arguments.seed)
    # Small enough that a random accumulator cannot overflow int16 with 32 pieces on the
    # board, and small enough that the output sum stays far inside int32.
    ft_weight = rng.integers(-24, 25, (768, arguments.l1)).astype(np.int16)
    ft_bias = rng.integers(-24, 25, arguments.l1).astype(np.int16)
    out_weight = rng.integers(-8, 9, (arguments.buckets, 2 * arguments.l1)).astype(np.int16)
    out_bias = np.zeros(arguments.buckets, dtype=np.int32)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.out,
        ft_weight=ft_weight,
        ft_bias=ft_bias,
        out_weight=out_weight,
        out_bias=out_bias,
        qa=np.int32(arguments.qa),
        qb=np.int32(arguments.qb),
        scale=np.int32(arguments.scale),
    )
    print(f"{arguments.out} ({arguments.out.stat().st_size:,} bytes)")
    print(f"  L1={arguments.l1} buckets={arguments.buckets} QA={arguments.qa} QB={arguments.qb}")


if __name__ == "__main__":
    main()
