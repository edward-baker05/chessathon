"""The two agreements a king-conditioned network depends on.

A net trained under one feature convention and played under another still trains to a
plausible loss and then plays badly, with nothing in any curve to say why. There are two
places that can go wrong here and neither shows up as an error.

**Offline against runtime.** `tools/dataset.py` builds the feature indices the trainer
learns from, and `nnue` builds the ones the engine plays with. They are written twice, so
they are checked against each other on real positions rather than reasoned about.

**Incremental against full.** Every move updates the accumulator in place, and a king
crossing its own bucket boundary is the one move that cannot be: every feature that
perspective reads has moved to a different block. A missed refresh leaves an accumulator
that is wrong and stays wrong for the rest of the subtree.

Both run at one bucket and at four. At one bucket the whole mechanism folds to a constant
zero, so a test that only ran there would prove nothing about the other. `nnue` fixes its
shape at import from the file it loads, so each case is a subprocess with its own network.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHECK = ROOT / "tests" / "king_bucket_check.py"

# Positions with something to say: castling on both wings, an en passant capture available,
# promotions of every kind, kings on each side of both bucket boundaries, and endings where
# the king walks. Every legal move of each is played, so the move flags are covered by
# construction rather than by a list of cases.
POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/6pk/4Kp2/7p/5P1P/3q4/3N4/2r5 b - - 0 1",
    "8/8/8/4pP2/8/8/8/4K2k w - e6 0 2",
]


def run_with(tmp_path: Path, king_buckets: int) -> dict[str, Any]:
    """Write a network of this shape, then check the runtime against it out of process."""
    weights = tmp_path / f"net-kb{king_buckets}.npz"
    subprocess.run(
        [sys.executable, "-m", "tools.random_net", "--l1", "32", "--buckets", "8",
         "--king-buckets", str(king_buckets), "--out", str(weights)],
        cwd=ROOT, check=True, capture_output=True,
    )
    finished = subprocess.run(
        [sys.executable, str(CHECK), json.dumps(POSITIONS)],
        cwd=ROOT, check=True, capture_output=True, text=True,
        env={**os.environ, "CHESSATHON_WEIGHTS": str(weights)},
    )
    parsed: dict[str, Any] = json.loads(finished.stdout.strip().splitlines()[-1])
    return parsed


@pytest.mark.parametrize("king_buckets", [1, 4])
def test_offline_features_and_incremental_updates_agree(
    tmp_path: Path, king_buckets: int
) -> None:
    result = run_with(tmp_path, king_buckets)
    assert result["king_buckets"] == king_buckets
    assert result["count"] == 0, result["failures"]
