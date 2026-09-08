"""Shuffle a packed training file on disk. Offline, never imported by anything that ships.

`tools/extract.py` writes records in the order the Lichess dump lists them, and that order
carries a lot of structure. Measured over consecutive slabs of two million records, the
fraction of positions with a mate-range score moves between 0.105 and 0.154, a spread of
0.0116 against the 0.0006 that independent sampling would give. The dump is grouped, so
neighbouring records are not independent draws.

That matters twice over. `tools/train.py` shuffles inside a slab rather than across the
file, so every batch is a biased sample and every epoch ends on whichever slab happened to
be drawn last; and a holdout taken as a slice of that file is a pocket of it rather than a
sample of it. Shuffling once here makes the file itself the thing that is random, and then
a sequential read of it is an unbiased one.

Two passes. The first scatters every record into one of `--buckets` files, choosing the
bucket uniformly at random, so a bucket is a random subset of the file rather than a
region of it. The second reads each bucket back, permutes it in memory and appends it.
Assigning uniformly and then permuting within each bucket is a uniform permutation of the
whole file, and no pass ever holds more than one bucket at a time.

Not shipped: tools/ never reaches the zip.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import dataset  # noqa: E402

# Records per read of the input. Sized so one chunk plus its scatter is a few hundred
# megabytes, which keeps the first pass sequential without needing the file in memory.
CHUNK = 1 << 22


def scatter(records: np.ndarray, scratch: Path, buckets: int,
            rng: np.random.Generator) -> list[int]:
    """Write every record to a randomly chosen bucket file. Returns the record counts."""
    counts = [0] * buckets
    handles = [(scratch / f"bucket{index:04d}.bin").open("wb") for index in range(buckets)]
    started = time.perf_counter()
    try:
        for start in range(0, records.shape[0], CHUNK):
            block = np.asarray(records[start : start + CHUNK])
            assignment = np.asarray(rng.integers(0, buckets, size=block.shape[0]))
            # argsort groups the chunk by bucket in one pass, so each bucket takes one
            # contiguous write rather than a write per record.
            order = np.argsort(assignment, kind="stable")
            sorted_block = block[order]
            edges: np.ndarray = np.searchsorted(assignment[order], np.arange(buckets + 1))
            for index in range(buckets):
                piece = sorted_block[edges[index] : edges[index + 1]]
                handles[index].write(piece.tobytes())
                counts[index] += piece.shape[0]
            done = min(start + CHUNK, records.shape[0])
            rate = done / max(time.perf_counter() - started, 1e-9)
            print(f"\rpass 1: {done:,} of {records.shape[0]:,} scattered, {rate:,.0f}/s",
                  end="", flush=True)
    finally:
        for handle in handles:
            handle.close()
    print()
    return counts


def gather(out: Path, scratch: Path, buckets: int, rng: np.random.Generator, total: int) -> int:
    """Permute each bucket in memory and append it to the output. Returns records written."""
    written = 0
    started = time.perf_counter()
    with out.open("wb") as sink:
        for index in range(buckets):
            path = scratch / f"bucket{index:04d}.bin"
            block = np.fromfile(path, dtype=np.uint8).reshape(-1, dataset.RECORD)
            sink.write(block[rng.permutation(block.shape[0])].tobytes())
            written += block.shape[0]
            # Deleted as we go, so the run needs one file's worth of extra space at the
            # end rather than a second whole copy of the training set.
            path.unlink()
            rate = written / max(time.perf_counter() - started, 1e-9)
            print(f"\rpass 2: {written:,} of {total:,} written, {rate:,.0f}/s", end="", flush=True)
    print()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.bin")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: the input with .shuffled.bin in place of .bin")
    parser.add_argument("--buckets", type=int, default=64,
                        help="more buckets means less memory in pass 2 and more open files")
    parser.add_argument("--scratch", type=Path, default=None,
                        help="where bucket files live; default: beside the output")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    if not arguments.data.exists():
        parser.error(f"{arguments.data} does not exist. Build it with tools/extract.py")
    out = arguments.out or arguments.data.with_suffix(".shuffled.bin")
    if out.resolve() == arguments.data.resolve():
        parser.error("refusing to shuffle in place; --out must differ from --data")

    scratch = arguments.scratch or out.parent / f".shuffle-{out.stem}"
    scratch.mkdir(parents=True, exist_ok=True)
    records = dataset.load(arguments.data)
    print(f"{records.shape[0]:,} records, {arguments.buckets} buckets, scratch in {scratch}")

    rng = np.random.default_rng(arguments.seed)
    counts = scatter(records, scratch, arguments.buckets, rng)
    # A bucket that came out far from the mean would mean the assignment was not uniform,
    # and pass 2 sizes its memory from the largest bucket.
    print(f"  bucket sizes: min {min(counts):,} max {max(counts):,} "
          f"mean {sum(counts) / len(counts):,.0f}")
    written = gather(out, scratch, arguments.buckets, rng, records.shape[0])
    if written != records.shape[0]:
        raise ValueError(f"wrote {written:,} records from an input of {records.shape[0]:,}")
    scratch.rmdir()

    sidecar = arguments.data.with_suffix(".meta.json")
    if sidecar.exists():
        shutil.copyfile(sidecar, out.with_suffix(".meta.json"))
    print(f"\n{out} ({out.stat().st_size:,} bytes)")
    print(f"train on it with: uv run python tools/train.py --data {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
