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

FEATURES = dataset.FEATURES
# The factoriser: every HalfKA feature (king bucket, piece slot, square) also updates a
# shared (piece slot, square) weight, and the HalfKA index is bucket * 704 + slot * 64 +
# square, so the virtual index is simply the index modulo 704. Most (king, piece, square)
# triples are rare, and without this they train slowly from scratch; the shared weight
# gives every position a gradient on the general case as well as the specific one.
# tools/quantise.py folds it into the table at export, so nothing ships knowing about it.
VIRTUAL_FEATURES = dataset.PIECE_SLOTS * 64
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
        self.factoriser = nn.EmbeddingBag(VIRTUAL_FEATURES, hidden, mode="sum")
        self.transformer_bias = nn.Parameter(torch.zeros(hidden))
        self.output = nn.Parameter(torch.zeros(buckets, 2 * hidden))
        self.output_bias = nn.Parameter(torch.zeros(buckets))
        self.hidden = hidden

        bound = 1.0 / FEATURES**0.5
        # The king-specific weights start at zero so the network begins as a plain
        # piece-square net and earns its king conditioning, rather than starting as noise
        # spread over 32 buckets that every position has to train back out.
        nn.init.zeros_(self.transformer.weight)
        nn.init.uniform_(self.factoriser.weight, -bound, bound)
        nn.init.uniform_(self.output, -bound, bound)
        # The bias stays at zero. Starting it at 0.5, half way up the activation's live
        # band, was tried because the gradient through squared clipped ReLU is
        # 2 * clamp(acc, 0, 1) and a zero bias over a zero transformer starts the feature
        # transformer at a gradient of near zero. It does nothing: over 600 steps on the
        # shuffled file the live fraction of the accumulator went 0.491 -> 0.867 from zero
        # and 1.000 -> 0.880 from 0.5, and the loss at step 600 was 0.01389 against
        # 0.01400, so Adam rescales its way out within the first hundred steps.
        #
        # The narrow accumulator late in training is real and is not this. At epoch 17 of
        # the run before this one, 65.9% of entries sat at or below zero and 8.7% above
        # one, so three quarters of the network took no gradient on any given position.
        # That is a scale the network chooses over a whole run, and moving it wants its
        # own A/B rather than a guess folded into a schedule change.

    def forward(
        self,
        white: torch.Tensor,
        black: torch.Tensor,
        offsets: torch.Tensor,
        stm: torch.Tensor,
        bucket: torch.Tensor,
    ) -> torch.Tensor:
        accumulated_white = (
            self.transformer(white, offsets)
            + self.factoriser(white % VIRTUAL_FEATURES, offsets)
            + self.transformer_bias
        )
        accumulated_black = (
            self.transformer(black, offsets)
            + self.factoriser(black % VIRTUAL_FEATURES, offsets)
            + self.transformer_bias
        )

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
        """Bound the weights that ship, which is the sum and not either half alone.

        `tools/quantise.py` folds the factoriser into the table, so the exported weight for
        feature `f` is `transformer[f] + factoriser[f % VIRTUAL_FEATURES]`. Clamping the
        two halves separately bounds neither the number that is written nor the number the
        overflow proof measures: on the epoch 17 checkpoint of the run this replaces, both
        halves were inside +/- 1.98 and 0.02% of the folded weights were not, reaching 2.87.

        The factoriser is bounded first and the transformer then carries whatever the sum
        has left over, so a transformer row that is still zero stays zero: for those rows
        the sum is the factoriser alone, which has already been brought inside the bound,
        and subtracting it back leaves zero.
        """
        with torch.no_grad():
            self.factoriser.weight.clamp_(-FT_CLAMP, FT_CLAMP)
            folded = self.transformer.weight.view(-1, VIRTUAL_FEATURES, self.hidden)
            folded += self.factoriser.weight
            folded.clamp_(-FT_CLAMP, FT_CLAMP)
            folded -= self.factoriser.weight
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
    keep: np.ndarray | None = None,
) -> Iterator[Batch]:
    """Yield batches, shuffling inside large contiguous slabs.

    The training file is far larger than memory, so it is memory mapped. Gathering fully
    random indices out of a memory mapped file is a random read per position and is disk
    bound; reading a large contiguous slab and shuffling inside it gets the same mixing
    for sequential reads, but only if the file's own order carries no structure.

    The file `tools/extract.py` writes does carry structure, so this is only an unbiased
    sampler once `tools/shuffle.py` has been run over it. Measured on the unshuffled file,
    the fraction of mate-range scores varies by 0.0116 across consecutive slabs against
    the 0.0006 of independent sampling, which makes every batch a biased sample and every
    epoch end on whichever slab was drawn last. Train on the shuffled file.

    `keep` is a boolean mask over `records` marking what may be yielded. It is how the
    holdout is withheld: the holdout is a random sample of the whole file rather than a
    slice of it, so it cannot be expressed as a range.
    """
    starts = np.arange(0, records.shape[0], slab)
    if shuffle:
        rng.shuffle(starts)
    for start in starts:
        block = np.asarray(records[start : start + slab])
        live = (
            np.arange(block.shape[0])
            if keep is None
            else np.flatnonzero(keep[start : start + block.shape[0]])
        )
        order = rng.permutation(live) if shuffle else live
        for at in range(0, order.shape[0], batch_size):
            chosen = order[at : at + batch_size]
            # The last batch of a slab is short, and so is a holdout set smaller than one
            # batch. Sizing from the slice rather than from batch_size is what stops a
            # small holdout silently evaluating nothing and reporting a loss of zero.
            size = chosen.shape[0]
            if size == 0:
                continue
            index, white, black, stm, score, pieces = dataset.unpack(block[chosen])

            counts = np.bincount(index, minlength=size)
            offsets = np.zeros(size, dtype=np.int64)
            np.cumsum(counts[:-1], out=offsets[1:])
            # From the real piece count, not from `counts`: each perspective omits its
            # own king, so `counts` is one short and would put the trainer in a
            # different output bucket from `nnue.forward` on every position.
            bucket = np.clip((pieces - 2) // ((32 - 2) // buckets + 1), 0, buckets - 1)

            yield (
                torch.from_numpy(white).to(device, non_blocking=True),
                torch.from_numpy(black).to(device, non_blocking=True),
                torch.from_numpy(offsets).to(device, non_blocking=True),
                torch.from_numpy(stm).to(device, non_blocking=True),
                torch.from_numpy(bucket).to(device, non_blocking=True),
                torch.from_numpy(score).float().to(device, non_blocking=True),
            )


def target_of(score: torch.Tensor) -> torch.Tensor:
    """The win probability to train towards, from a centipawn score.

    Nothing is clipped, because the sigmoid has already done it. `tools/extract.py` writes
    mate scores as +/-12800 and sigmoid(12800/400) is 1.0 to every bit a float has, but
    sigmoid(2000/400) is 0.9933 and sigmoid(3000/400) is 0.9994: there is no information
    left above about 2000 cp for a clip to remove. Clipping the label at 2000 was measured
    against the epoch 24 checkpoint and moved the holdout loss by 1.7%, from 0.008612 to
    0.008467, which is not worth a knob.

    That saturation is worth understanding rather than working around. It is why the net
    barely ranks degrees of winning: on the holdout its median output is +871 cp for
    labels between 800 and 1500, and +952 cp for labels between 1500 and 5000. If that
    wants fixing, the lever is the temperature of this sigmoid and not a clip. SCALE does
    two jobs here, as the engine's centipawns per output unit, which the quantisation
    identity fixes at 400, and as the temperature of the loss, which is free. Separating
    them, so the target is sigmoid(score / K) against a prediction of sigmoid(out * SCALE
    / K), buys resolution on won positions at the cost of resolution near equality, and is
    an experiment with an A/B at the end of it rather than a change to make quietly.
    """
    return torch.sigmoid(score / SCALE)


# Positions this far from equality are decisive, and the network's error on them says
# something different from its error on the rest. They are 16.5% of the file and produced
# 37.7% of the holdout loss on the first HalfKA run, so a single averaged number is mostly
# a report on them; splitting it is what showed that the run's loss on ordinary positions
# was at its best at epoch 5 and never beat it, which the average had hidden completely.
DECISIVE_CP = 1500.0


def evaluate_loss(model: Network, records: np.ndarray, batch_size: int, device: torch.device,
                  buckets: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Holdout loss overall, on decisive positions, and on ordinary ones."""
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    counts = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for white, black, offsets, stm, bucket, score in batches(
            records, batch_size, device, buckets, rng, shuffle=False
        ):
            predicted = torch.sigmoid(model(white, black, offsets, stm, bucket))
            error = ((predicted - target_of(score)) ** 2).double()
            decisive = (score.abs() > DECISIVE_CP).double()
            totals += torch.stack([(error * decisive).sum(), (error * (1 - decisive)).sum()])
            counts += torch.stack([decisive.sum(), (1 - decisive).sum()])
    model.train()
    split = (totals / counts.clamp(min=1.0)).tolist()
    overall = float(totals.sum() / counts.sum().clamp(min=1.0))
    return overall, float(split[0]), float(split[1])


def draw_holdout(records: np.ndarray, fraction: float, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A random sample of the file, and the mask of what is left to train on.

    A sample rather than a tail slice. The file `tools/extract.py` writes is grouped, so a
    slice of it is a pocket of the data and not a measurement of the whole; drawing at
    random is right whether or not `tools/shuffle.py` has been run.

    Indices are drawn with replacement and then deduplicated, which is far cheaper than a
    permutation of a file this size and, once sorted, turns the gather into a nearly
    sequential read of the memory map. Duplicates cost a redraw rather than a short
    holdout, because a sampler that quietly returns fewer positions than it was asked for
    is the kind of thing that is only noticed much later.
    """
    total = records.shape[0]
    wanted = int(total * fraction)
    drawn = np.zeros(0, dtype=np.int64)
    while drawn.shape[0] < wanted:
        drawn = np.unique(np.concatenate([drawn, rng.integers(0, total, size=wanted)]))
    # `np.unique` returns its result sorted, so trimming it to length would take the
    # lowest indices and hold out the front of the file instead of a sample of it. The
    # surplus has to be dropped at random, and the survivors sorted again so that the
    # gather below still reads the memory map close to in order.
    index = np.sort(rng.choice(drawn, size=wanted, replace=False))
    keep = np.ones(total, dtype=bool)
    keep[index] = False
    return np.asarray(records[index]), keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.bin")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "data" / "checkpoints")
    parser.add_argument("--l1", type=int, default=256)
    parser.add_argument("--buckets", type=int, default=8)
    # Twenty at 0.85, not ten at 0.75. The ten-epoch schedule was adopted because the
    # first HalfKA run's holdout loss on ordinary positions was best at epoch 5 and never
    # beat it, which read as overfitting; it was the unshuffled file, and every epoch
    # ending on whichever biased slab was drawn last. On the shuffled file the run does
    # not overfit at all. It underfits: at epoch 17 the training loss was 0.007308 against
    # a holdout of 0.008170, a gap of 12% over 269M positions, and both were still falling
    # in step with the learning rate rather than flattening. What 0.75 does on that curve
    # is reach 7.5e-5 by epoch 10 and 1.0e-5 by epoch 17, so the run spends its second
    # half unable to move. 0.85 over twenty passes 2.3e-4 at ten and 4.6e-5 at twenty:
    # the same 20x anneal, spent where the network is still learning. Checkpoints are
    # written every epoch, so a run can be stopped early without losing the schedule.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.85, help="learning rate decay per epoch")
    # Zero. AdamW's default of 0.01 decays every row of the feature transformer on every
    # step, including the king-specific rows that a batch never touches, which is not a
    # thing to do to an embedding table that HalfKA has deliberately made sparse.
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--holdout", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    if not arguments.data.exists():
        parser.error(f"{arguments.data} does not exist. Build it with `make data`")

    device = pick_device(arguments.device)
    records = dataset.load(arguments.data)
    rng = np.random.default_rng(arguments.seed)
    holdout_records, keep = draw_holdout(records, arguments.holdout, rng)
    print(f"device {device}, L1 {arguments.l1}, {int(keep.sum()):,} positions, "
          f"{holdout_records.shape[0]:,} held out at random")

    model = Network(arguments.l1, arguments.buckets).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=arguments.lr, weight_decay=arguments.weight_decay
    )
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=arguments.gamma)
    arguments.checkpoints.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, arguments.epochs + 1):
        started = time.perf_counter()
        running = 0.0
        seen = 0
        for steps, (white, black, offsets, stm, bucket, score) in enumerate(batches(
            records, arguments.batch, device, arguments.buckets, rng, keep=keep
        ), start=1):
            predicted = torch.sigmoid(model(white, black, offsets, stm, bucket))
            target = target_of(score)
            loss = ((predicted - target) ** 2).mean()

            optimiser.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            model.clamp_()

            running += float(loss.detach()) * score.shape[0]
            seen += score.shape[0]
            # Counted in steps, not in positions: withholding the holdout leaves a slab
            # that is not a whole number of batches, so `seen` no longer lands on a
            # multiple of the batch size and a test against one would never print.
            if steps % 200 == 0:
                rate = seen / (time.perf_counter() - started)
                print(f"\repoch {epoch}: {seen:,} positions, loss {running / seen:.6f}, "
                      f"{rate:,.0f}/s", end="", flush=True)

        schedule.step()
        validation, decisive, ordinary = evaluate_loss(
            model, holdout_records, arguments.batch, device, arguments.buckets, rng
        )
        elapsed = time.perf_counter() - started
        # The split, not just the average. The average is mostly a report on the
        # decisive positions, and a run can stall on the ordinary ones for twenty epochs
        # without the single number saying so.
        print(f"\repoch {epoch}: train {running / max(seen, 1):.6f}  holdout {validation:.6f} "
              f"(decisive {decisive:.6f}, ordinary {ordinary:.6f})  "
              f"{elapsed:.0f}s ({seen / max(elapsed, 1e-9):,.0f}/s)".ljust(110))

        # Every epoch, so a long run can be stopped at any point without losing the day.
        torch.save(
            {
                "state": model.state_dict(),
                "l1": arguments.l1,
                "buckets": arguments.buckets,
                "epoch": epoch,
                "holdout_loss": validation,
                "holdout_decisive": decisive,
                "holdout_ordinary": ordinary,
                "positions": int(keep.sum()),
            },
            arguments.checkpoints / f"epoch{epoch:03d}.pt",
        )

    print(f"\ncheckpoints in {arguments.checkpoints}")
    print("quantise one with: uv run python tools/quantise.py --checkpoint <file>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
