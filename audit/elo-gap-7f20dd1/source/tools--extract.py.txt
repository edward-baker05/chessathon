"""Turn the Lichess evaluation file into packed training positions.

Input is `lichess_db_eval.jsonl.zst`, about 21.7 GB compressed and several times that as
text: one JSON object per line, holding a FEN and a list of Stockfish evaluations at
various depths. Using it as training data is explicitly permitted: "Training data is
unrestricted, including positions annotated by an existing engine. What ships inside the
zip is what the ban covers."

Two things make this fast enough to be worth running.

It does not parse the JSON. Each line carries several principal variations of ten moves
each, so `json.loads` would spend almost all of its time building strings this tool then
throws away. Instead the three fields that matter are located with `bytes.find`.

It does not use python-chess. Constructing a `chess.Board` per position would cost more
than everything else combined. The FEN board field is unpacked directly into bitboards,
and the engine's own `attacked` decides whether the side to move is in check.

Output is a flat file of 32 byte records, described in `RECORD`. Not shipped: tools/ never
reaches the zip, which only carries root *.py and weights/.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bitboard import BOCC, KING, NFIELDS, STM, WOCC  # noqa: E402
from position import attacked, king_square  # noqa: E402
from tools.dataset import MAX_PIECES, RECORD, pack  # noqa: E402

PIECE_CODE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}

# A mate is stored as a saturating centipawn score rather than dropped. In probability
# space `sigmoid(12800 / 400)` is indistinguishable from 1, which is the right target.
MATE_SCORE = 12800

DEFAULT_SOURCE = ROOT / "data" / "lichess_db_eval.jsonl.zst"
DEFAULT_OUT = ROOT / "data" / "train.bin"


class Filters:
    """Why a position is dropped. Counted so the run can be judged rather than trusted."""

    def __init__(self) -> None:
        self.kept = 0
        self.shallow = 0
        self.in_check = 0
        self.tactical = 0
        self.extreme = 0
        self.malformed = 0

    def merge(self, other: "Filters") -> None:
        for name in ("kept", "shallow", "in_check", "tactical", "extreme", "malformed"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def report(self) -> str:
        seen = self.kept + self.shallow + self.in_check + self.tactical + self.extreme
        seen += self.malformed
        if seen == 0:
            return "nothing read"
        return (
            f"{self.kept:,} kept of {seen:,} ({self.kept / seen:.1%})\n"
            f"  dropped shallow (depth below the floor): {self.shallow:,}\n"
            f"  dropped in check:                        {self.in_check:,}\n"
            f"  dropped best move is a capture:          {self.tactical:,}\n"
            f"  dropped score out of range:              {self.extreme:,}\n"
            f"  dropped malformed:                       {self.malformed:,}"
        )


def parse_board(field: bytes) -> tuple[int, list[int], np.ndarray] | None:
    """FEN board field to an occupancy, the piece codes in square order, and bitboards.

    Squares are indexed a1 = 0, so the FEN's first rank is rank 8 and the loop counts down.
    """
    state = np.zeros(NFIELDS, dtype=np.uint64)
    occupancy = 0
    codes: list[int] = []
    rank = 7
    file = 0
    for character in field.decode("ascii"):
        if character == "/":
            if file != 8:
                return None
            rank -= 1
            file = 0
        elif character.isdigit():
            file += int(character)
        else:
            code = PIECE_CODE.get(character)
            if code is None or file > 7 or rank < 0:
                return None
            square = rank * 8 + file
            bit = np.uint64(1) << np.uint64(square)
            state[code % 6] |= bit
            state[BOCC if code >= 6 else WOCC] |= bit
            occupancy |= 1 << square
            file += 1
    if rank != 0 or file != 8:
        return None
    # The nibble order has to match the square order, so collect after the whole board is
    # known rather than as it is parsed, which walks ranks from the top.
    for square in range(64):
        if occupancy >> square & 1:
            bit = np.uint64(1) << np.uint64(square)
            black = bool(state[BOCC] & bit)
            for piece in range(6):
                if state[piece] & bit:
                    codes.append((6 if black else 0) + piece)
                    break
    return occupancy, codes, state


def find_best_eval(line: bytes) -> tuple[int, int, bytes] | None:
    """Depth, score and first principal variation move of the deepest evaluation.

    Each element of `evals` is `{"pvs":[...],"knodes":N,"depth":D}`, so the depth of an
    element follows its variations. Splitting on the opening of each element keeps them
    associated without building any of the objects.
    """
    best: tuple[int, int, bytes] | None = None
    for chunk in line.split(b'{"pvs":[')[1:]:
        depth_at = chunk.find(b'"depth":')
        if depth_at < 0:
            continue
        depth_end = chunk.find(b"}", depth_at)
        try:
            depth = int(chunk[depth_at + 8 : depth_end].rstrip(b"} "))
        except ValueError:
            continue
        if best is not None and depth <= best[0]:
            continue

        # The first variation is the engine's best line, so its score is the position's.
        cp_at = chunk.find(b'"cp":')
        mate_at = chunk.find(b'"mate":')
        if 0 <= cp_at < (mate_at if mate_at >= 0 else len(chunk)):
            end = chunk.find(b",", cp_at)
            score = int(chunk[cp_at + 5 : end])
        elif mate_at >= 0:
            end = chunk.find(b",", mate_at)
            plies = int(chunk[mate_at + 7 : end])
            score = MATE_SCORE if plies > 0 else -MATE_SCORE
        else:
            continue

        move_at = chunk.find(b'"line":"')
        if move_at < 0:
            continue
        move = chunk[move_at + 8 : move_at + 13].split(b" ")[0]
        best = (depth, score, move)
    return best


def convert(payload: tuple[list[bytes], int]) -> tuple[bytes, Filters]:
    """One batch of lines to packed records. Runs in a worker process."""
    lines, min_depth = payload
    out = bytearray()
    counts = Filters()
    state = np.zeros(NFIELDS, dtype=np.uint64)

    for line in lines:
        fen_at = line.find(b'{"fen":"')
        if fen_at < 0:
            counts.malformed += 1
            continue
        fen_end = line.find(b'"', fen_at + 8)
        fields = line[fen_at + 8 : fen_end].split(b" ")
        if len(fields) < 2:
            counts.malformed += 1
            continue

        found = find_best_eval(line)
        if found is None:
            counts.malformed += 1
            continue
        depth, score, move = found
        if depth < min_depth:
            counts.shallow += 1
            continue

        parsed = parse_board(fields[0])
        if parsed is None or len(move) < 4:
            counts.malformed += 1
            continue
        occupancy, codes, board = parsed
        if len(codes) > MAX_PIECES:
            # The file contains a few illegal positions, roughly one in fifty thousand,
            # with more pieces than a chessboard holds. Left in, they would overflow the
            # record's nibble field and silently overwrite its side to move and score.
            counts.malformed += 1
            continue
        black_to_move = fields[1] == b"b"

        # The file's scores are from White's point of view. Verified empirically over
        # 10,482 decisive, materially imbalanced positions: the sign agreed with White
        # 81.0% of the time and with the side to move 49.8%, a coin flip. Getting this
        # backwards trains a net that plays close to randomly with nothing in the loss
        # curve to say why.
        if black_to_move:
            score = -score
        if abs(score) > MATE_SCORE:
            counts.extreme += 1
            continue

        state[:] = board
        state[STM] = np.uint64(1 if black_to_move else 0)
        if state[KING] & (state[BOCC] if black_to_move else state[WOCC]) == 0:
            counts.malformed += 1
            continue
        if attacked(state, king_square(state, int(black_to_move)), int(not black_to_move)):
            counts.in_check += 1
            continue

        # A position whose best move wins material is one the quiescence search resolves
        # for itself. Training on it teaches the evaluation to double count tactics.
        source = (move[0] - 97) + (move[1] - 49) * 8
        target = (move[2] - 97) + (move[3] - 49) * 8
        if not 0 <= source < 64 or not 0 <= target < 64:
            counts.malformed += 1
            continue
        captures = bool(occupancy >> target & 1)
        if not captures and codes and (occupancy >> source & 1):
            # En passant: a pawn changing file onto an empty square.
            index = bin(occupancy & ((1 << source) - 1)).count("1")
            if codes[index] % 6 == 0 and (source & 7) != (target & 7):
                captures = True
        if captures or len(move) > 4:
            counts.tactical += 1
            continue

        out += pack(occupancy, codes, black_to_move, score)
        counts.kept += 1

    return bytes(out), counts


def read_lines(source: Path, batch: int) -> Iterator[list[bytes]]:
    """Decompressed lines, in batches, without holding the file in memory."""
    import zstandard

    with source.open("rb") as handle:
        reader = zstandard.ZstdDecompressor().stream_reader(handle)
        pending: list[bytes] = []
        remainder = b""
        while True:
            block = reader.read(1 << 22)
            if not block:
                break
            block = remainder + block
            *complete, remainder = block.split(b"\n")
            pending.extend(complete)
            if len(pending) >= batch:
                yield pending
                pending = []
        if remainder:
            pending.append(remainder)
        if pending:
            yield pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-depth", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many kept positions")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--batch", type=int, default=20_000)
    arguments = parser.parse_args()

    if not arguments.source.exists():
        parser.error(
            f"{arguments.source} does not exist. Download it with\n"
            f"  curl -L -o {arguments.source} "
            f"https://database.lichess.org/lichess_db_eval.jsonl.zst"
        )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    totals = Filters()
    started = time.perf_counter()
    lines = read_lines(arguments.source, arguments.batch)
    batches = ((batch, arguments.min_depth) for batch in lines)

    with arguments.out.open("wb") as sink, mp.Pool(arguments.workers) as pool:
        for packed, counts in pool.imap(convert, batches, chunksize=1):
            sink.write(packed)
            totals.merge(counts)
            elapsed = time.perf_counter() - started
            print(
                f"\r{totals.kept:,} kept in {elapsed:.0f}s "
                f"({totals.kept / max(elapsed, 1e-9):,.0f}/s)",
                end="",
                flush=True,
            )
            if arguments.limit and totals.kept >= arguments.limit:
                pool.terminate()
                break

    print(f"\n\n{arguments.out} ({arguments.out.stat().st_size:,} bytes)")
    print(totals.report())
    sidecar = arguments.out.with_suffix(".meta.json")
    sidecar.write_text(
        "{\n"
        f'  "source": "{arguments.source.name}",\n'
        f'  "positions": {totals.kept},\n'
        f'  "min_depth": {arguments.min_depth},\n'
        f'  "record_bytes": {RECORD},\n'
        f'  "mate_score": {MATE_SCORE}\n'
        "}\n"
    )
    print(f"{sidecar} records how this file was made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
