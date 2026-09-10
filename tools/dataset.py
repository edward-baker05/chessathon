"""The packed training record, and the one place features are derived from it.

`tools/extract.py` writes these records and `tools/train.py` reads them. Both go through
this module, and `tests/test_king_buckets.py` proves that the features it produces are
exactly the ones `nnue` builds when the engine plays, at one king bucket and at four. That
agreement is the property the whole project depends on and the one that fails most quietly:
a net trained under one feature convention and played under another still trains to a
plausible loss and then plays badly, with nothing anywhere to say why.

Not shipped: tools/ never reaches the zip.
"""

import struct
from typing import Any

import numpy as np

# One packed position, 32 bytes. 32 rather than the 27 actually used, because an aligned
# record lets a whole file be read as one numpy view with no arithmetic.
#
#   0..7    occupancy bitboard, little endian uint64
#   8..23   one nibble per occupied square in ascending square order, low nibble first,
#           holding colour * 6 + piece, piece in pawn, knight, bishop, rook, queen, king
#   24      side to move, 0 white, 1 black
#   25..26  score in centipawns from the side to move's point of view, int16
#   27..31  unused
RECORD = 32
OCC_OFFSET, NIBBLE_OFFSET, STM_OFFSET, SCORE_OFFSET = 0, 8, 24, 25
MAX_PIECES = 32


def pack(occupancy: int, codes: list[int], black_to_move: bool, score: int) -> bytes:
    """One position to one record. `codes` is in ascending square order.

    The piece count is checked rather than trusted. Only MAX_PIECES nibbles fit, and a
    thirty third piece would write over the side to move and the score instead of failing.
    The Lichess file really does contain such positions, about one in fifty thousand, with
    up to fifty one pieces on the board.
    """
    if len(codes) > MAX_PIECES:
        raise ValueError(f"{len(codes)} pieces will not fit in a {RECORD} byte record")
    record = bytearray(RECORD)
    struct.pack_into("<Q", record, OCC_OFFSET, occupancy)
    for index, code in enumerate(codes):
        if index & 1:
            record[NIBBLE_OFFSET + index // 2] |= code << 4
        else:
            record[NIBBLE_OFFSET + index // 2] = code
    record[STM_OFFSET] = 1 if black_to_move else 0
    struct.pack_into("<h", record, SCORE_OFFSET, score)
    return bytes(record)


def from_board(board: Any, score: int = 0) -> bytes:
    """One `chess.Board` to one record, in the extractor's own convention.

    `tools/extract.py` builds records straight out of the dump's FEN field and never needs
    a board object. Everything that checks the extractor against the runtime does, and
    writing the piece walk out twice is how the two quietly stop agreeing.
    """
    occupancy = 0
    codes: list[int] = []
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue
        occupancy |= 1 << square
        codes.append((0 if piece.color else 1) * 6 + piece.piece_type - 1)
    return pack(occupancy, codes, not board.turn, score)


def load(path: Any) -> np.ndarray:
    """A whole file as an (n, RECORD) uint8 view, memory mapped rather than read."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    usable = (raw.shape[0] // RECORD) * RECORD
    return raw[:usable].reshape(-1, RECORD)


Unpacked = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

# One 768-feature block per own-king bucket, and the partition that chooses the block.
# Both must agree with `nnue.king_bucket` exactly; `tests/test_king_buckets.py` checks
# every square rather than trusting the arithmetic to be written the same way twice.
SQUARE_FEATURES = 768
KING_CODE = 5  # colour * 6 + piece, with piece 5 being the king, so 5 white and 11 black.
KING_BUCKET_COUNTS = (1, 4, 8, 16)


def bucket_of(oriented_king: np.ndarray, count: int) -> np.ndarray:
    """Which block a perspective reads, from its own king's already-oriented square.

    Must agree with `nnue.king_bucket` for every square and every count, which
    `tests/test_king_buckets.py` checks rather than assumes. Orienting is the caller's job,
    because the caller is the one that knows which perspective it is building.
    """
    if count == 1:
        return np.zeros_like(oriented_king)
    if count == 4:
        return ((oriented_king >> 2) & 1) | (((oriented_king >> 5) & 1) << 1)
    if count == 8:
        return ((oriented_king & 7) >> 1) | (((oriented_king >> 5) & 1) << 2)
    if count == 16:
        return ((oriented_king & 7) >> 1) | ((((oriented_king >> 3) & 7) >> 1) << 2)
    raise ValueError(f"{count} king buckets; expected one of {KING_BUCKET_COUNTS}")


def unpack(records: np.ndarray, king_buckets: int = 1) -> Unpacked:
    """Records to sparse features, one row per piece.

    Returns `(index, white, black, stm, score)`. `index[k]` says which position piece `k`
    belongs to, so the three arrays together are a coordinate-format sparse matrix that a
    torch embedding bag consumes directly. Everything is vectorised over the whole batch,
    because a Python loop over positions is slower than the GPU step it feeds.

    With `king_buckets` at 4 every index is offset by its perspective's own-king block.
    The kings are read out of the packed record itself rather than passed in: the nibbles
    already carry both kings' piece codes and the occupancy already carries their squares.
    """
    # unpackbits with little bit order puts bit b of byte n at column 8n + b, and squares
    # are numbered from the low bit of the low byte, so the column index is the square.
    bits = np.unpackbits(records[:, OCC_OFFSET:OCC_OFFSET + 8], axis=1, bitorder="little")
    counts = bits.sum(axis=1, dtype=np.int64)

    nibbles = records[:, NIBBLE_OFFSET:NIBBLE_OFFSET + MAX_PIECES // 2]
    codes = np.empty((records.shape[0], MAX_PIECES), dtype=np.int64)
    codes[:, 0::2] = nibbles & 0x0F
    codes[:, 1::2] = nibbles >> 4

    # Both flat arrays are produced in row-major order with squares ascending inside each
    # row, which is the order the nibbles were written in, so they line up piece for piece.
    squares = (np.flatnonzero(bits) % 64).astype(np.int64)
    occupied = np.arange(MAX_PIECES)[None, :] < counts[:, None]
    flat_codes = codes[occupied]
    index = np.repeat(np.arange(records.shape[0], dtype=np.int64), counts)

    # This must match `feature` in nnue.py exactly:
    #     ((colour ^ perspective) * 6 + piece) * 64 + (square ^ (perspective * 56))
    # With code = colour * 6 + piece, white's view is code * 64 + square, and black's view
    # swaps the colour half, which is (code + 6) mod 12, and flips the rank.
    white = flat_codes * 64 + squares
    black = ((flat_codes + 6) % 12) * 64 + (squares ^ 56)

    if king_buckets > 1:
        # Each position contributes exactly one king per colour, so scattering the king
        # squares by position index gives one square per position with no loop. A record
        # missing a king would leave -1 behind and index the weights out of range, so it
        # is caught here rather than discovered as a wrong evaluation.
        kings = np.full((records.shape[0], 2), -1, dtype=np.int64)
        for colour in (0, 1):
            found = flat_codes == KING_CODE + 6 * colour
            kings[index[found], colour] = squares[found]
        if (kings < 0).any():
            raise ValueError(f"{int((kings < 0).any(axis=1).sum())} records have no king")
        # White reads the board as it stands; black reads it flipped, and `bucket_of`
        # takes an already-oriented square, so black's king is flipped before bucketing.
        white_bucket = bucket_of(kings[:, 0], king_buckets)[index]
        black_bucket = bucket_of(kings[:, 1] ^ 56, king_buckets)[index]
        white = white_bucket * SQUARE_FEATURES + white
        black = black_bucket * SQUARE_FEATURES + black

    stm = records[:, STM_OFFSET].astype(np.int64)
    raw = records[:, SCORE_OFFSET:SCORE_OFFSET + 2].copy()
    score = raw.view(np.int16).ravel().astype(np.int64)
    return index, white, black, stm, score
