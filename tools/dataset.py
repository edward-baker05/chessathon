"""The packed training record, and the one place features are derived from it.

`tools/extract.py` writes these records and `tools/train.py` reads them. Both go through
this module, and `tests/test_dataset.py` proves that the features it produces are exactly
the ones `nnue.refresh` builds when the engine plays. That agreement is the property the
whole project depends on and the one that fails most quietly: a net trained under one
feature convention and played under another still trains to a plausible loss and then
plays badly, with nothing anywhere to say why.

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

# HalfKA geometry. These deliberately restate the constants in nnue.py rather than
# importing them: nnue.py loads the shipped weights and compiles numba at import, which
# a trainer should not have to do. `tests/test_dataset.py` is what holds the two copies
# together, by comparing the features built here against the ones nnue.refresh builds.
KING_BUCKETS = 32
PIECE_SLOTS = 11
FEATURES = KING_BUCKETS * PIECE_SLOTS * 64
WHITE_KING_CODE, BLACK_KING_CODE = 5, 11


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


def load(path: Any) -> np.ndarray:
    """A whole file as an (n, RECORD) uint8 view, memory mapped rather than read."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    usable = (raw.shape[0] // RECORD) * RECORD
    return raw[:usable].reshape(-1, RECORD)


Unpacked = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


def _side_features(
    perspective: int,
    king_squares: np.ndarray,
    index: np.ndarray,
    colour: np.ndarray,
    piece: np.ndarray,
    squares: np.ndarray,
) -> np.ndarray:
    """One perspective's feature index per piece. Must match `nnue.feature` exactly.

        oriented = king ^ (perspective * 56)
        mirror   = 7 if oriented is on files a to d else 0
        bucket   = the folded king square, 0 to 31
        slot     = piece for our pieces, 5 + piece for theirs
        index    = (bucket * 11 + slot) * 64 + (square ^ (perspective * 56) ^ mirror)

    The caller drops the perspective's own king afterwards; it is the bucket, not a feature.
    """
    oriented_king = king_squares ^ (perspective * 56)
    mirror = np.where((oriented_king & 7) < 4, 7, 0)
    folded_king = oriented_king ^ mirror
    bucket = (folded_king >> 3) * 4 + (folded_king & 7) - 4
    slot = np.where((colour ^ perspective) == 0, piece, 5 + piece)
    square_index = squares ^ (perspective * 56) ^ mirror[index]
    features: np.ndarray = (bucket[index] * PIECE_SLOTS + slot) * 64 + square_index
    return features


def unpack(records: np.ndarray) -> Unpacked:
    """Records to sparse features, one row per piece.

    Returns `(index, white, black, stm, score, pieces)`. `index[k]` says which position
    piece `k` belongs to, so the three arrays together are a coordinate-format sparse
    matrix that a torch embedding bag consumes directly. Everything is vectorised over the
    whole batch, because a Python loop over positions is slower than the GPU step it feeds.

    `pieces` is the full piece count per position, returned separately because `index` no
    longer carries it: each perspective omits its own king, so `index` holds one entry
    fewer per position than there are pieces. The output bucket is chosen by the real piece
    count in `nnue.forward`, and deriving it from the feature count instead would put the
    trainer in a different bucket from the engine on every single position.
    """
    # unpackbits with little bit order puts bit b of byte n at column 8n + b, and squares
    # are numbered from the low bit of the low byte, so the column index is the square.
    bits = np.unpackbits(records[:, OCC_OFFSET:OCC_OFFSET + 8], axis=1, bitorder="little")
    pieces = bits.sum(axis=1, dtype=np.int64)

    nibbles = records[:, NIBBLE_OFFSET:NIBBLE_OFFSET + MAX_PIECES // 2]
    codes = np.empty((records.shape[0], MAX_PIECES), dtype=np.int64)
    codes[:, 0::2] = nibbles & 0x0F
    codes[:, 1::2] = nibbles >> 4

    # Both flat arrays are produced in row-major order with squares ascending inside each
    # row, which is the order the nibbles were written in, so they line up piece for piece.
    squares = (np.flatnonzero(bits) % 64).astype(np.int64)
    occupied = np.arange(MAX_PIECES)[None, :] < pieces[:, None]
    flat_codes = codes[occupied]
    index = np.repeat(np.arange(records.shape[0], dtype=np.int64), pieces)

    is_white_king = flat_codes == WHITE_KING_CODE
    is_black_king = flat_codes == BLACK_KING_CODE
    # A position with no king, or two, would silently misalign every king square after it
    # with the wrong position, because the king squares are gathered positionally.
    if int(is_white_king.sum()) != records.shape[0] or int(is_black_king.sum()) != records.shape[0]:
        raise ValueError("every position must hold exactly one king of each colour")

    colour, piece = flat_codes // 6, flat_codes % 6
    white = _side_features(0, squares[is_white_king], index, colour, piece, squares)
    black = _side_features(1, squares[is_black_king], index, colour, piece, squares)

    # Each perspective drops exactly one piece, its own king, so both sides keep the same
    # number per position and one offsets array still describes both bags.
    stm = records[:, STM_OFFSET].astype(np.int64)
    raw = records[:, SCORE_OFFSET:SCORE_OFFSET + 2].copy()
    score = raw.view(np.int16).ravel().astype(np.int64)
    return index[~is_white_king], white[~is_white_king], black[~is_black_king], stm, score, pieces
