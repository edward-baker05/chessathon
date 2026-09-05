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


def new_table() -> np.ndarray:
    return np.zeros((BUCKETS, ENTRIES_PER_BUCKET * 2), dtype=np.uint64)


TT = new_table()


@njit(void(uint64[:, :]), cache=False)
def tt_clear(table: Bits) -> None:
    table[:, :] = U(0)


@njit(uint64(int32, int32, int64, int64, int32, int64), cache=False, inline="always")
def pack_entry(
    score: Bits, move: Bits, depth: Square, bound: Square, static_eval: Bits, age: Square
) -> Bits:
    return (
        U(np.int64(score) + SCORE_BIAS)
        | (U(np.int64(move) & 0xFFFF) << U(16))
        | (U(depth & 0xFF) << U(32))
        | (U((bound & 3) | ((age & 63) << 2)) << U(40))
        | (U(np.int64(static_eval) + SCORE_BIAS) << U(48))
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
            stored_depth = np.int64((stored >> U(32)) & U(0xFF))
            stored_age = np.int64((stored >> U(42)) & U(63))
            stored_move = np.int32(np.int64((stored >> U(16)) & U(0xFFFF)))
            # Never lose a known good move to a search that did not find one.
            if kept_move == 0:
                kept_move = stored_move
            # Depth-preferred: within one search, shallower work never displaces deeper
            # work for the same position. Depth dominates bound type here, so an exact
            # score from a shallow search does not get to overwrite a deep one.
            if depth < stored_depth and stored_age == (age & 63):
                return
            break
        stored = table[bucket, i * 2 + 1]
        stored_depth = np.int64((stored >> U(32)) & U(0xFF))
        stored_age = np.int64((stored >> U(42)) & U(63))
        # Prefer to evict shallow work, and work from an older search more readily still.
        value = stored_depth - 2 * ((age - stored_age) & 63)
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
        score = np.int32(np.int64(stored & U(0xFFFF)) - SCORE_BIAS)
        move = np.int32(np.int64((stored >> U(16)) & U(0xFFFF)))
        depth = np.int64((stored >> U(32)) & U(0xFF))
        bound = np.int64((stored >> U(40)) & U(3))
        static_eval = np.int32(np.int64((stored >> U(48)) & U(0xFFFF)) - SCORE_BIAS)
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
