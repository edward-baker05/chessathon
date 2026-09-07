"""The shipped network is the network that was trained.

`tests/test_nnue.py` proves the jitted int path agrees with a numpy reimplementation of the
same int weights. `tests/test_dataset.py` proves the trainer derives the features the engine
derives. Neither of them compares anything against torch, which is what the training loss
was actually measured on, so a bucket rule, a perspective convention or a factoriser fold
that diverged between `tools/train.py` and `nnue.py` would pass both of them and still ship
a network that plays nothing like the one whose loss curve was watched.

This closes that gap. A float checkpoint goes through `tools/quantise.py`, and the result is
evaluated by a numpy transcription of `nnue`'s integer arithmetic and compared against the
torch model it came from. Chained with the two tests above, every link from the trainer to
the jitted engine is covered by something.
"""

import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from tests.conftest import random_positions
from tools import dataset, train
from tools.extract import parse_board

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = 8
HIDDEN = 32  # small: this tests agreement, not strength, and a wide layer only slows it


def packed(boards: list[chess.Board]) -> np.ndarray:
    records = []
    for board in boards:
        parsed = parse_board(board.board_fen().encode())
        assert parsed is not None, board.fen()
        occupancy, codes, _ = parsed
        records.append(dataset.pack(occupancy, codes, board.turn == chess.BLACK, 0))
    raw = np.frombuffer(b"".join(records), dtype=np.uint8)
    return raw.reshape(-1, dataset.RECORD)


def a_checkpoint(path: Path, seed: int) -> None:
    """A random but realistically scaled network, saved the way `train.py` saves one."""
    torch.manual_seed(seed)
    model = train.Network(HIDDEN, BUCKETS)
    with torch.no_grad():
        # Both halves non-zero, so a quantiser that dropped the factoriser would be caught.
        model.transformer.weight.uniform_(-0.2, 0.2)
        model.factoriser.weight.uniform_(-0.2, 0.2)
        model.transformer_bias.uniform_(-0.2, 0.2)
        model.output.uniform_(-0.05, 0.05)
        model.output_bias.uniform_(-0.05, 0.05)
    torch.save(
        {"state": model.state_dict(), "l1": HIDDEN, "buckets": BUCKETS,
         "epoch": 1, "holdout_loss": 0.0, "positions": 0},
        path,
    )


def float_evaluations(path: Path, records: np.ndarray) -> np.ndarray:
    """What the trainer thinks these positions are worth, in centipawns."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = train.Network(HIDDEN, BUCKETS)
    model.load_state_dict(blob["state"])
    model.eval()

    index, white, black, stm, _score, pieces = dataset.unpack(records)
    counts = np.bincount(index, minlength=records.shape[0])
    offsets = np.zeros(records.shape[0], dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    bucket = np.clip((pieces - 2) // ((32 - 2) // BUCKETS + 1), 0, BUCKETS - 1)
    with torch.no_grad():
        out = model(
            torch.from_numpy(white), torch.from_numpy(black), torch.from_numpy(offsets),
            torch.from_numpy(stm), torch.from_numpy(bucket),
        )
    return np.asarray(out.numpy() * train.SCALE, dtype=np.float64)


def quantised_evaluations(npz_path: Path, records: np.ndarray) -> np.ndarray:
    """A numpy transcription of `nnue.forward`, run on the file the engine would load.

    Deliberately not a call into `nnue`: that module loads one fixed weights file at import
    and compiles it into numba literals, so it cannot be pointed at the file under test.
    """
    with np.load(npz_path) as data:
        ft_weight = data["ft_weight"].astype(np.int64)
        ft_bias = data["ft_bias"].astype(np.int64)
        out_weight = data["out_weight"].astype(np.int64)
        out_bias = data["out_bias"].astype(np.int64)
        qa, qb, scale = int(data["qa"]), int(data["qb"]), int(data["scale"])

    index, white, black, stm, _score, pieces = dataset.unpack(records)
    divisor = (32 - 2) // BUCKETS + 1
    results = []
    for row in range(records.shape[0]):
        mine = index == row
        accumulated = np.stack([
            ft_bias + ft_weight[white[mine]].sum(axis=0),
            ft_bias + ft_weight[black[mine]].sum(axis=0),
        ])
        side = int(stm[row])
        activated = np.concatenate([accumulated[side], accumulated[1 - side]])
        activated = np.clip(activated, 0, qa) ** 2
        bucket = int(np.clip((pieces[row] - 2) // divisor, 0, BUCKETS - 1))
        total = int((activated * out_weight[bucket]).sum())
        results.append((total // qa + int(out_bias[bucket])) * scale // (qa * qb))
    return np.asarray(results, dtype=np.float64)


@pytest.mark.parametrize("seed", [3, 17])
def test_the_quantised_network_agrees_with_the_checkpoint_it_came_from(
    tmp_path: Path, seed: int
) -> None:
    checkpoint, npz = tmp_path / "epoch001.pt", tmp_path / "net.npz"
    a_checkpoint(checkpoint, seed)
    finished = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "quantise.py"),
         "--checkpoint", str(checkpoint), "--out", str(npz)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert finished.returncode == 0, finished.stdout + finished.stderr

    records = packed(list(random_positions(count=60, seed=seed)))
    predicted = float_evaluations(checkpoint, records)
    shipped = quantised_evaluations(npz, records)

    difference = np.abs(predicted - shipped)
    relative = difference.mean() / predicted.std()
    correlation = float(np.corrcoef(predicted, shipped)[0, 1])

    # Both bounds sit in a gap that was measured, not guessed. Over seeds 3, 17 and 29, a
    # correct export scores a relative error of 0.11 to 0.16 and a correlation of 0.989 to
    # 0.991; dropping the factoriser gives 0.54 to 0.73 and 0.64 to 0.77; transposing the
    # two perspectives gives 0.99 to 1.53 and -0.11 to 0.34.
    #
    # A relative error of 0.13 sounds alarming and is an artefact of testing with a random
    # network: its output sum is a sum of positive activations against random-signed
    # weights, so it cancels almost completely and leaves a very narrow spread of
    # evaluations for an absolute rounding error to be compared against. A trained network
    # has structure instead of cancellation. The real one measured 6.2 cp of mean absolute
    # difference against its own checkpoint, on evaluations spanning hundreds of centipawns.
    assert correlation > 0.95, f"correlation {correlation:.4f}\n{finished.stdout}"
    assert relative < 0.30, f"relative error {relative:.4f}\n{finished.stdout}"
