"""Transposition table.

Storage is one uint64[BUCKETS, 8] array: four entries per row, each a key and a packed
data word, so a bucket is exactly one 64-byte cache line.

The table is passed in as an argument rather than read from a module global. numba exposes
globals to jitted code as *readonly* arrays, so a jitted function can read a global table
but cannot write to one. Everything the search mutates therefore travels as a parameter.

The table is deliberately never cleared between moves. The process lives for exactly one
game, so last move's entries describe the same game and are worth keeping. agent.py clears
it only when the incoming position does not chain onto the game it remembers.
"""

from typing import Any

import numpy as np
from numba import int32, int64, njit, uint64, void

Bits = Any
Square = Any
Flag = Any

U = np.uint64

MATE = 30000
MATE_IN_MAX = MATE - 256

BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT = 0, 1, 2, 3

# 128 MB: BUCKETS rows of 8 uint64. Comfortably inside the platform's 2 GB.
BUCKETS = 1 << 21
ENTRIES_PER_BUCKET = 4

# Score is stored biased so it survives as an unsigned 16-bit field.
SCORE_BIAS = 32768

# The packed data word, bit by bit. Sixty four, exactly, with nothing spare:
#   0..15   score, biased
#   16..31  move, in the sixteen bit form below
#   32..39  depth
#   40..41  bound
#   42..47  age
#   48..63  static evaluation, biased
SCORE_SHIFT, SCORE_BITS = 0, 16
MOVE_SHIFT, MOVE_BITS = 16, 16
DEPTH_SHIFT, DEPTH_BITS = 32, 8
BOUND_SHIFT, BOUND_BITS = 40, 2
AGE_SHIFT, AGE_BITS = 42, 6
STATIC_SHIFT, STATIC_BITS = 48, 16

MOVE_MASK = (1 << MOVE_BITS) - 1
DEPTH_MASK = (1 << DEPTH_BITS) - 1
BOUND_MASK = (1 << BOUND_BITS) - 1
# Ages are compared cyclically, so everything that reads or advances one masks with this.
AGE_MASK = (1 << AGE_BITS) - 1
STATIC_MASK = (1 << STATIC_BITS) - 1

# The generator writes a move as `from | to << 6 | promo << 12 | flag << 15`, which is
# seventeen bits, and the word above has room for sixteen. Masking the seventeenth away,
# as this did, drops the high bit of the flag: a stored castle comes back as a quiet move
# and a stored promotion as an en passant capture. Neither corrupted move ever matches
# anything the generator produces, so the entry is not wrong so much as useless, and the
# node loses both its TT move ordering and, through the internal iterative reduction, a
# ply of depth. Every node whose best move is a castle paid that, which in the opening is
# a great many of them.
#
# The seventeenth bit is redundant rather than needed. A promotion piece is one of four,
# so it is stored as `promo - 1` in two bits and restored on the way out; every other move
# has no promotion piece at all. Taking the bit from the age field instead would be a
# smaller change and a worse one: a five bit age wraps every thirty two searches, which is
# well inside one game, and an entry that has wrapped looks new again and stops being
# evicted.
#
# FLAG_PROMO is 3, named in bitboard.py. Comparing the decoded flag against the literal
# keeps this module importing nothing but numpy and numba, as the rest of it does.
_FLAG_PROMO = 3


def new_table() -> np.ndarray:
    return np.zeros((BUCKETS, ENTRIES_PER_BUCKET * 2), dtype=np.uint64)


TT = new_table()


@njit(void(uint64[:, :]), cache=False)
def tt_clear(table: Bits) -> None:
    table[:, :] = U(0)


@njit(int64(int32), cache=False, inline="always")
def pack_move(move: Bits) -> Square:
    """A generator move to its sixteen bit stored form."""
    wide = np.int64(move)
    flag = (wide >> 15) & 3
    promo = ((wide >> 12) & 7) - 1 if flag == _FLAG_PROMO else 0
    return (wide & 0xFFF) | (promo << 12) | (flag << 14)


@njit(int32(int64), cache=False, inline="always")
def unpack_move(packed: Square) -> Bits:
    """The stored form back to a move the generator would have produced."""
    flag = (packed >> 14) & 3
    promo = ((packed >> 12) & 3) + 1 if flag == _FLAG_PROMO else 0
    return np.int32((packed & 0xFFF) | (promo << 12) | (flag << 15))


@njit(uint64(int32, int32, int64, int64, int32, int64), cache=False, inline="always")
def pack_entry(
    score: Bits, move: Bits, depth: Square, bound: Square, static_eval: Bits, age: Square
) -> Bits:
    return (
        (U(np.int64(score) + SCORE_BIAS) << U(SCORE_SHIFT))
        | (U(pack_move(move) & MOVE_MASK) << U(MOVE_SHIFT))
        | (U(depth & DEPTH_MASK) << U(DEPTH_SHIFT))
        | (U(bound & BOUND_MASK) << U(BOUND_SHIFT))
        | (U(age & AGE_MASK) << U(AGE_SHIFT))
        | (U(np.int64(static_eval) + SCORE_BIAS) << U(STATIC_SHIFT))
    )


@njit(void(uint64[:, :], uint64, int64, int32, int32, int64, int64, int32, int64), cache=False)
def tt_store(
    table: Bits,
    key: Bits,
    ply: Square,
    score: Bits,
    move: Bits,
    depth: Square,
    bound: Square,
    static_eval: Bits,
    age: Square,
) -> None:
    """Store an entry, rebasing mate scores to be relative to this node.

    A mate score found at ply N means "mate in K from here". Storing it unadjusted makes it
    mean "mate in K from the root", which is the classic source of mate lines that are off
    by a few moves and of engines that shuffle instead of finishing.
    """
    adjusted = score
    if score >= MATE_IN_MAX:
        adjusted = np.int32(score + ply)
    elif score <= -MATE_IN_MAX:
        adjusted = np.int32(score - ply)

    bucket = np.int64(key & U(BUCKETS - 1))
    slot = 0
    worst = 1 << 30
    kept_move = move
    for i in range(ENTRIES_PER_BUCKET):
        stored_key = table[bucket, i * 2]
        if stored_key == U(0):
            slot = i
            break
        if stored_key == key:
            slot = i
            stored = table[bucket, i * 2 + 1]
            stored_depth = np.int64((stored >> U(DEPTH_SHIFT)) & U(DEPTH_MASK))
            stored_age = np.int64((stored >> U(AGE_SHIFT)) & U(AGE_MASK))
            stored_move = unpack_move(np.int64((stored >> U(MOVE_SHIFT)) & U(MOVE_MASK)))
            # Never lose a known good move to a search that did not find one.
            if kept_move == 0:
                kept_move = stored_move
            # Depth-preferred: within one search, shallower work never displaces deeper
            # work for the same position. Depth dominates bound type here, so an exact
            # score from a shallow search does not get to overwrite a deep one.
            if depth < stored_depth and stored_age == (age & AGE_MASK):
                return
            break
        stored = table[bucket, i * 2 + 1]
        stored_depth = np.int64((stored >> U(DEPTH_SHIFT)) & U(DEPTH_MASK))
        stored_age = np.int64((stored >> U(AGE_SHIFT)) & U(AGE_MASK))
        # Prefer to evict shallow work, and work from an older search more readily still.
        value = stored_depth - 2 * ((age - stored_age) & AGE_MASK)
        if value < worst:
            worst = value
            slot = i

    table[bucket, slot * 2] = key
    table[bucket, slot * 2 + 1] = pack_entry(adjusted, kept_move, depth, bound, static_eval, age)


@njit(cache=False)
def tt_probe(table: Bits, key: Bits, ply: Square) -> tuple[Flag, Bits, Bits, Square, Square, Bits]:
    """Look up an entry, undoing the mate-score rebasing done on store."""
    bucket = np.int64(key & U(BUCKETS - 1))
    for i in range(ENTRIES_PER_BUCKET):
        if table[bucket, i * 2] != key:
            continue
        stored = table[bucket, i * 2 + 1]
        if stored == U(0):
            continue
        score = np.int32(np.int64((stored >> U(SCORE_SHIFT)) & U(0xFFFF)) - SCORE_BIAS)
        move = unpack_move(np.int64((stored >> U(MOVE_SHIFT)) & U(MOVE_MASK)))
        depth = np.int64((stored >> U(DEPTH_SHIFT)) & U(DEPTH_MASK))
        bound = np.int64((stored >> U(BOUND_SHIFT)) & U(BOUND_MASK))
        static_eval = np.int32(np.int64((stored >> U(STATIC_SHIFT)) & U(STATIC_MASK)) - SCORE_BIAS)
        if score >= MATE_IN_MAX:
            score = np.int32(score - ply)
        elif score <= -MATE_IN_MAX:
            score = np.int32(score + ply)
        return True, score, move, depth, bound, static_eval
    return False, np.int32(0), np.int32(0), np.int64(0), np.int64(BOUND_NONE), np.int32(0)


# Warm every jitted function with the argument types the real calls use.
tt_clear(TT)
tt_store(TT, U(1), 0, int32(0), int32(0), 1, BOUND_EXACT, int32(0), 1)
tt_probe(TT, U(1), 0)
tt_clear(TT)
