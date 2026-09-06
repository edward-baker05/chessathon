"""Train the network. torch, offline, never imported by anything that ships.

The architecture is `768 -> L1x2 -> 1` with squared clipped ReLU and output buckets by
piece count. See docs/superpowers/specs/2026-09-06-nnue-evaluation-design.md for why that
shape and not a deeper one: an output stack of `2*L1 -> 16 -> 32 -> 1` measured at 1386 ns
per node in the engine against 87 ns for a single output row.

The float network and the quantised one are the same function. With input weights scaled
by QA, output weights by QB and clipped ReLU saturating at QA, the quantised arithmetic in
nnue.py reduces exactly to

    eval_cp = SCALE * (sum_i screlu(acc_i) * w_i + bias)

so with SCALE at 400 and the target written as `sigmoid(cp / 400)`, the loss below is
simply `MSE(sigmoid(out), sigmoid(label / 400))`. Keeping that identity exact is what lets
tools/quantise.py convert without retuning anything.

Device order is CUDA, then MPS, then CPU, so the same script runs on a desktop GPU and on
an Apple laptop without edits.
"""

import argparse
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import dataset  # noqa: E402

# One training batch, already on the device: white features, black features, bag offsets,
# side to move, output bucket, and the label in centipawns.
Batch = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

FEATURES = 768
# Centipawns per unit of the network's output, and the divisor that turns a centipawn
# score into a win probability. Equal, so that the loss is a plain sigmoid of the output.
SCALE = 400

# Weights are clamped during training rather than only at export, so the network is
# trained as the thing that will actually be shipped instead of being distorted by
# quantisation afterwards. The bound is what int16 accumulators and an int32 output sum
# can carry; tools/quantise.py proves the exact case for the weights it writes.
FT_CLAMP = 1.98
OUT_CLAMP = 1.98


class Network(nn.Module):
    def __init__(self, hidden: int, buckets: int) -> None:
        super().__init__()
        # EmbeddingBag sums the active features without ever materialising a dense input,
        # which is the whole reason a 768 wide sparse layer is cheap to train.
        self.transformer = nn.EmbeddingBag(FEATURES, hidden, mode="sum")
        self.transformer_bias = nn.Parameter(torch.zeros(hidden))
        self.output = nn.Parameter(torch.zeros(buckets, 2 * hidden))
        self.output_bias = nn.Parameter(torch.zeros(buckets))
        self.hidden = hidden

        bound = 1.0 / FEATURES**0.5
        nn.init.uniform_(self.transformer.weight, -bound, bound)
        nn.init.uniform_(self.output, -bound, bound)

    def forward(
        self,
        white: torch.Tensor,
        black: torch.Tensor,
        offsets: torch.Tensor,
        stm: torch.Tensor,
        bucket: torch.Tensor,
    ) -> torch.Tensor:
        accumulated_white = self.transformer(white, offsets) + self.transformer_bias
        accumulated_black = self.transformer(black, offsets) + self.transformer_bias

        # The evaluation is always from the side to move's point of view, so the two
        # halves are ordered by whose turn it is rather than by colour.
        side = stm.unsqueeze(1).to(accumulated_white.dtype)
        us = accumulated_white * (1 - side) + accumulated_black * side
        them = accumulated_black * (1 - side) + accumulated_white * side

        # Squared clipped ReLU. The square is what makes this worth about 20 to 30 Elo
        # over a plain clipped ReLU for one extra multiply at inference.
        activated = torch.cat([us, them], dim=1).clamp(0.0, 1.0) ** 2
        weights = self.output[bucket]
        return (activated * weights).sum(dim=1) + self.output_bias[bucket]

    def clamp_(self) -> None:
        with torch.no_grad():
            self.transformer.weight.clamp_(-FT_CLAMP, FT_CLAMP)
            self.output.clamp_(-OUT_CLAMP, OUT_CLAMP)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(
    records: np.ndarray,
    batch_size: int,
    device: torch.device,
    buckets: int,
    rng: np.random.Generator,
    shuffle: bool = True,
    slab: int = 1 << 21,
) -> Iterator[Batch]:
    """Yield batches, shuffling inside large contiguous slabs.

    The training file is far larger than memory, so it is memory mapped. Gathering fully
    random indices out of a memory mapped file is a random read per position and is disk
    bound; reading a large contiguous slab and shuffling inside it gets the same mixing
    for sequential reads. The file's own order carries no structure to defeat this.
    """
    starts = np.arange(0, records.shape[0], slab)
    if shuffle:
        rng.shuffle(starts)
    for start in starts:
        block = np.asarray(records[start : start + slab])
        order = rng.permutation(block.shape[0]) if shuffle else np.arange(block.shape[0])
        for at in range(0, block.shape[0], batch_size):
            chosen = order[at : at + batch_size]
            # The last batch of a slab is short, and so is a holdout set smaller than one
            # batch. Sizing from the slice rather than from batch_size is what stops a
            # small holdout silently evaluating nothing and reporting a loss of zero.
            size = chosen.shape[0]
            if size == 0:
                continue
            index, white, black, stm, score = dataset.unpack(block[chosen])

            counts = np.bincount(index, minlength=size)
            offsets = np.zeros(size, dtype=np.int64)
            np.cumsum(counts[:-1], out=offsets[1:])
            bucket = np.clip((counts - 2) // ((32 - 2) // buckets + 1), 0, buckets - 1)

            yield (
                torch.from_numpy(white).to(device, non_blocking=True),
                torch.from_numpy(black).to(device, non_blocking=True),
                torch.from_numpy(offsets).to(device, non_blocking=True),
                torch.from_numpy(stm).to(device, non_blocking=True),
                torch.from_numpy(bucket).to(device, non_blocking=True),
                torch.from_numpy(score).float().to(device, non_blocking=True),
            )


def evaluate_loss(model: Network, records: np.ndarray, batch_size: int, device: torch.device,
                  buckets: int, rng: np.random.Generator) -> float:
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for white, black, offsets, stm, bucket, score in batches(
            records, batch_size, device, buckets, rng, shuffle=False
        ):
            predicted = torch.sigmoid(model(white, black, offsets, stm, bucket))
            target = torch.sigmoid(score / SCALE)
            total += float(((predicted - target) ** 2).sum())
            seen += score.shape[0]
    model.train()
    return total / max(seen, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.bin")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "data" / "checkpoints")
    parser.add_argument("--l1", type=int, default=512)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--holdout", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    if not arguments.data.exists():
        parser.error(f"{arguments.data} does not exist. Build it with `make data`")

    device = pick_device(arguments.device)
    records = dataset.load(arguments.data)
    split = int(records.shape[0] * (1 - arguments.holdout))
    train_records, holdout_records = records[:split], records[split:]
    print(f"device {device}, L1 {arguments.l1}, {train_records.shape[0]:,} positions, "
          f"{holdout_records.shape[0]:,} held out")

    model = Network(arguments.l1, arguments.buckets).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=arguments.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=arguments.epochs)
    rng = np.random.default_rng(arguments.seed)
    arguments.checkpoints.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, arguments.epochs + 1):
        started = time.perf_counter()
        running = 0.0
        seen = 0
        for white, black, offsets, stm, bucket, score in batches(
            train_records, arguments.batch, device, arguments.buckets, rng
        ):
            predicted = torch.sigmoid(model(white, black, offsets, stm, bucket))
            target = torch.sigmoid(score / SCALE)
            loss = ((predicted - target) ** 2).mean()

            optimiser.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            model.clamp_()

            running += float(loss.detach()) * score.shape[0]
            seen += score.shape[0]
            if seen % (arguments.batch * 200) == 0:
                rate = seen / (time.perf_counter() - started)
                print(f"\repoch {epoch}: {seen:,} positions, loss {running / seen:.6f}, "
                      f"{rate:,.0f}/s", end="", flush=True)

        schedule.step()
        validation = evaluate_loss(
            model, holdout_records, arguments.batch, device, arguments.buckets, rng
        )
        elapsed = time.perf_counter() - started
        print(f"\repoch {epoch}: train {running / max(seen, 1):.6f}  holdout {validation:.6f}  "
              f"{elapsed:.0f}s ({seen / max(elapsed, 1e-9):,.0f}/s)".ljust(90))

        # Every epoch, so a long run can be stopped at any point without losing the day.
        torch.save(
            {
                "state": model.state_dict(),
                "l1": arguments.l1,
                "buckets": arguments.buckets,
                "epoch": epoch,
                "holdout_loss": validation,
                "positions": train_records.shape[0],
            },
            arguments.checkpoints / f"epoch{epoch:03d}.pt",
        )

    print(f"\ncheckpoints in {arguments.checkpoints}")
    print("quantise one with: uv run python tools/quantise.py --checkpoint <file>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
