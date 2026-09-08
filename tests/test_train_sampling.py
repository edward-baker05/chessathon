"""The holdout has to be a sample of the file, and it has to be withheld.

Both of these failed silently in the first HalfKA run. The holdout was the tail slice of a
file that `tools/extract.py` writes in a grouped order, so it measured a pocket of the data
rather than the data; and once the split moved to a random draw, nothing but these tests
says the drawn positions are actually kept out of training. A holdout that leaks reports a
loss that falls beautifully and means nothing.
"""

import numpy as np
import torch

from tools import dataset, train


def make_records(count: int) -> np.ndarray:
    """A file of legal records whose first four bytes number each one."""
    rng = np.random.default_rng(1)
    records = np.zeros((count, dataset.RECORD), dtype=np.uint8)
    for row in range(count):
        # Two kings and a handful of pieces, on distinct squares.
        squares = sorted(rng.choice(64, size=4, replace=False).tolist())
        codes = [0, dataset.WHITE_KING_CODE, 6, dataset.BLACK_KING_CODE]
        occupancy = 0
        for square in squares:
            occupancy |= 1 << square
        packed = dataset.pack(occupancy, codes, bool(row & 1), int(rng.integers(-500, 500)))
        records[row] = np.frombuffer(packed, dtype=np.uint8)
    return records


def test_holdout_is_a_random_sample_not_a_slice() -> None:
    records = make_records(20000)
    rng = np.random.default_rng(0)
    holdout, keep = train.draw_holdout(records, 0.05, rng)

    assert holdout.shape[0] == 1000
    assert int((~keep).sum()) == 1000
    # A tail slice would put every held out index in the last 5% of the file. A sample
    # spreads them, so the mean index sits near the middle rather than near the end.
    held = np.flatnonzero(~keep)
    assert 0.4 < held.mean() / records.shape[0] < 0.6


def test_training_never_sees_a_held_out_record() -> None:
    records = make_records(20000)
    rng = np.random.default_rng(0)
    _, keep = train.draw_holdout(records, 0.05, rng)

    # The score is unique enough per record to identify it, so gathering every score the
    # sampler yields says exactly which records training touched.
    seen: list[int] = []
    for *_, score in train.batches(
        records, 512, torch.device("cpu"), 8, rng, slab=4096, keep=keep
    ):
        seen.extend(score.tolist())

    assert len(seen) == int(keep.sum())
    everything = records[:, dataset.SCORE_OFFSET:dataset.SCORE_OFFSET + 2].copy()
    scores = everything.view(np.int16).ravel().astype(np.int64)
    assert sorted(seen) == sorted(scores[keep].tolist())


def test_the_label_is_the_plain_sigmoid_of_the_score() -> None:
    # Nothing clips the label, because the sigmoid already has. This test exists to record
    # that as a decision rather than an omission: clipping at 2000 cp was measured against
    # the epoch 24 checkpoint and moved the holdout loss by 1.7%, because the targets it
    # would have changed differ by 0.0067 at most.
    score = torch.tensor([-12800.0, -2000.0, -350.0, 0.0, 120.0, 2000.0, 12800.0])
    assert torch.allclose(train.target_of(score), torch.sigmoid(score / train.SCALE))
    mate, decisive = train.target_of(torch.tensor([12800.0, 2000.0]))
    assert float(mate) - float(decisive) < 0.007
