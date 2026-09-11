"""The fresh holdout, in the form both the trainer and the scorer need.

`tools/holdout.py` writes positions with a reference score and the reference's top few moves.
Two things read that file: `tools/train.py`, which needs to score itself on it after every
epoch, and `tools/evalnet.py`, which needs to score finished networks on it. They must build
the same boards in the same order or their numbers are not comparable, so they build them
here, once.

Every position contributes its own board and then the board after each of the reference's
ranked moves. Evaluating the children is what turns a static evaluation into a move ordering,
and ordering is the half of the comparison a loss curve cannot show: this run produced a
network 6.6% better in loss that ordered candidate moves no better and drew its match.

Two conventions carried from `tools/extract.py`, so that a label here means what a training
label means. Scores are from the side to move. A mate is saturated at the same sentinel the
training data uses, and is flagged, because a saturated target is not a centipawn quantity:
a net saying +900 where the label says +12800 contributes 11,900 to a centipawn mean and
swamps it.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import chess
import numpy as np

from tools import dataset


class FreshSet(NamedTuple):
    """Everything a holdout evaluation needs, built once."""

    fens: list[str]
    """Every board to evaluate: each position, then the board after each ranked move."""
    owners: list[list[int]]
    """Per position, the indices into `fens` it owns: itself first, then its children."""
    records: np.ndarray
    """`fens` packed into training records, so a model can be run over them directly."""
    label_cp: np.ndarray
    """Reference score per position, from the side to move."""
    is_mate: np.ndarray
    """Which labels are saturated mates. Never averaged into a centipawn figure."""
    ranked_cp: list[list[int]]
    """Per position, the reference's score for each of its ranked moves."""
    rows: list[dict[str, Any]]
    """The original records, for grouping by phase, texture or cluster."""


def load(path: Path, limit: int = 0) -> FreshSet:
    loaded = json.loads(Path(path).read_text())
    rows = loaded["positions"][: limit or None]

    fens: list[str] = []
    owners: list[list[int]] = []
    ranked_cp: list[list[int]] = []
    for row in rows:
        board = chess.Board(str(row["fen"]))
        mine = [len(fens)]
        fens.append(board.fen())
        for entry in row["ranked"]:
            child = board.copy(stack=False)
            child.push(chess.Move.from_uci(str(entry["move"])))
            mine.append(len(fens))
            fens.append(child.fen())
        owners.append(mine)
        ranked_cp.append([int(entry["cp"]) for entry in row["ranked"]])

    records = np.frombuffer(
        b"".join(dataset.from_board(chess.Board(fen)) for fen in fens), dtype=np.uint8
    ).reshape(-1, dataset.RECORD).copy()

    return FreshSet(
        fens=fens,
        owners=owners,
        records=records,
        label_cp=np.array([float(row["cp"]) for row in rows]),
        is_mate=np.array([row["mate"] is not None for row in rows]),
        ranked_cp=ranked_cp,
        rows=rows,
    )


def score(fresh: FreshSet, evaluations: np.ndarray, scale: float) -> dict[str, float]:
    """Loss and move ordering for one set of evaluations, in centipawns.

    `evaluations` is one centipawn score per entry of `fresh.fens`, in that order. The
    sigmoid figure is the one to compare networks on, because it is the loss they were
    trained under; the centipawn mean is easier to read and excludes the mate rows.
    """
    predicted = np.array([evaluations[mine[0]] for mine in fresh.owners])
    quiet = ~fresh.is_mate

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values / scale))

    gap = predicted[quiet] - fresh.label_cp[quiet]
    # Error where the position is close to level. This is the number that decides games and
    # the one every capacity increase so far has failed to move: sibling moves differ by tens
    # of centipawns, so an evaluation noisier than that cannot order them. Measured against a
    # label floor of about 7cp, from re-evaluating real training records at 2M nodes.
    balanced = np.abs(fresh.label_cp) < 50
    top1 = ranked = 0
    for mine, theirs in zip(fresh.owners, fresh.ranked_cp, strict=True):
        if len(theirs) < 2:
            continue
        ranked += 1
        # The reference scores a move from the mover's side; after the move the opponent is
        # to move, so the net's score of the child is negated before the two are compared.
        ours = [-evaluations[slot] for slot in mine[1:]]
        if int(np.argmax(ours)) == int(np.argmax(theirs)):
            top1 += 1
    return {
        "sigmoid_mse": float(
            ((sigmoid(predicted) - sigmoid(fresh.label_cp)) ** 2).mean()
        ),
        "quiet_sigmoid_mse": float(
            ((sigmoid(predicted[quiet]) - sigmoid(fresh.label_cp[quiet])) ** 2).mean()
        ),
        "quiet_mae_cp": float(np.abs(gap).mean()),
        "balanced_mae_cp": float(
            np.abs(predicted[balanced] - fresh.label_cp[balanced]).mean()
        ) if balanced.any() else 0.0,
        "balanced_positions": float(balanced.sum()),
        "top1": top1 / max(ranked, 1),
        "positions": float(len(fresh.owners)),
    }
