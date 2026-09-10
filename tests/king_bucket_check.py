"""Runtime-side half of tests/test_king_buckets.py, run against one network shape.

`nnue` reads the number of king buckets out of the file it loads at import, so a test that
wants to check both shapes has to run this twice in two processes with `CHESSATHON_WEIGHTS`
pointing at a different network each time. Prints one JSON line and is never imported.
"""

import json
import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import movegen  # noqa: E402
import nnue  # noqa: E402
import position  # noqa: E402
from tools import dataset  # noqa: E402


def features_from_runtime(board: chess.Board, state: np.ndarray,
                          mail: np.ndarray) -> list[list[int]]:
    position.encode(board, state[0], mail[0])
    out: list[list[int]] = []
    for perspective in (0, 1):
        bucket = int(nnue.king_bucket(perspective, nnue.king_of(state[0], perspective)))
        rows = [
            int(nnue.feature(perspective, 0 if piece.color == chess.WHITE else 1,
                             piece.piece_type - 1, square, bucket))
            for square, piece in board.piece_map().items()
        ]
        out.append(sorted(rows))
    return out


def features_from_dataset(board: chess.Board) -> list[list[int]]:
    """The indices the trainer learns from, built out of the packed record."""
    record = np.frombuffer(dataset.from_board(board), dtype=np.uint8).reshape(1, dataset.RECORD)
    _index, white, black, _stm, _score = dataset.unpack(record, nnue.KING_BUCKETS)
    return [sorted(int(v) for v in white), sorted(int(v) for v in black)]


def encoded_moves(state: np.ndarray, mail: np.ndarray) -> dict[str, int]:
    """Every legal move of the position at slot 0, by UCI, in the engine's own encoding."""
    buffer = np.zeros(256, dtype=np.int32)
    count = int(movegen.generate(state[0], buffer, 0))
    legal: dict[str, int] = {}
    for index in range(count):
        move = int(buffer[index])
        position.make(state[0], mail[0], state[1], mail[1], np.int32(move))
        if position.legal_after(state[1], int(state[0][8])):
            legal[movegen.move_to_uci(move)] = move
    return legal


def main() -> int:
    positions = json.loads(sys.argv[1])
    state, mail = position.new_stacks()
    accumulator = nnue.new_accumulator(4)
    failures: list[list[str]] = []

    for fen in positions:
        board = chess.Board(fen)
        runtime = features_from_runtime(board, state, mail)
        offline = features_from_dataset(board)
        for name, want, got in zip(("white", "black"), runtime, offline, strict=True):
            if want != got:
                failures.append([f"{name} features disagree", fen])

        position.encode(board, state[0], mail[0])
        legal = encoded_moves(state, mail)
        by_uci = {move.uci(): move for move in board.legal_moves}
        if set(legal) != set(by_uci):
            failures.append(["move generation disagrees with python-chess", fen])
        for uci, move in legal.items():
            position.encode(board, state[0], mail[0])
            nnue.refresh(accumulator, 0, state[0], mail[0])
            position.make(state[0], mail[0], state[1], mail[1], np.int32(move))
            nnue.apply(accumulator, 0, state[0], mail[0], state[1], mail[1], np.int32(move))
            incremental = np.array(accumulator[1], copy=True)

            after = board.copy(stack=False)
            after.push(by_uci[uci])
            position.encode(after, state[2], mail[2])
            nnue.refresh(accumulator, 2, state[2], mail[2])
            if not np.array_equal(incremental, np.asarray(accumulator[2])):
                failures.append(["accumulator disagrees with a full refresh", fen, uci])

    print(json.dumps({"king_buckets": nnue.KING_BUCKETS,
                      "count": len(failures), "failures": failures[:10]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
