"""Bit primitives, attack tables and Zobrist keys. No position logic lives here.

Every table is built at import so numba compilation and table generation land inside the
platform's 90 second init budget rather than on the game clock.

Nothing here uses numba's cache=True. numba bakes the *contents* of global numpy arrays
into a cached binary with no warning, so a cached build would silently serve stale magics
or Zobrist keys the first time table generation changed.
"""

from typing import Any

import numpy as np
from numba import boolean, int64, njit, uint64

# The numba signature on each decorator below is the authoritative type contract, and it is
# checked at compile time. The Python annotations are deliberately Any: mypy cannot model
# the int and np.uint64 interconversion numba performs at the call boundary, and restating
# the contract in a second notation only creates a second place for it to be wrong.
Bits = Any
Square = Any

U = np.uint64
ONE = U(1)
ZERO = U(0)
FULL = U(0xFFFFFFFFFFFFFFFF)

# Indices into the position state vector. Six piece bitboards, two colour occupancies,
# then the scalars. 13 to 15 are spare, reserved for an NNUE accumulator handle later.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
WOCC, BOCC, STM, CASTLE, EP, HALF, KEY = 6, 7, 8, 9, 10, 11, 12
NFIELDS = 16

# Castling rights bits.
WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE = 1, 2, 4, 8
ALL_CASTLING = 15

DEBRUIJN = U(0x03F79D71B4CB0A89)

# Derived from the constant rather than transcribed. A transcribed table cannot be
# eyeballed and is wrong on only some squares, which is the worst way to be wrong.
DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
# The multiply is meant to overflow: wrapping at 64 bits is exactly what makes a de Bruijn
# sequence work, so numpy's overflow warning is noise rather than a signal here.
with np.errstate(over="ignore"):
    for _i in range(64):
        DEBRUIJN_INDEX[int(((ONE << U(_i)) * DEBRUIJN) >> U(58))] = _i


@njit(int64(uint64), cache=False, inline="always")
def popcount(b: Bits) -> Bits:
    n = 0
    while b:
        b &= b - ONE  # subtracting a uint64 literal keeps numba from widening to float
        n += 1
    return n


@njit(int64(uint64), cache=False, inline="always")
def lsb(b: Bits) -> Square:
    """Index of the lowest set bit. Undefined for zero, so callers must guard."""
    return DEBRUIJN_INDEX[int(((b & (~b + ONE)) * DEBRUIJN) >> U(58))]


@njit(uint64(int64, int64, int64, uint64), cache=False)
def ray_attacks(sq: Square, dr: Square, df: Square, occ: Bits) -> Bits:
    """Squares reachable along one direction, including the first blocker."""
    att = ZERO
    r = sq // 8 + dr
    f = sq % 8 + df
    while r >= 0 and r < 8 and f >= 0 and f < 8:
        s = r * 8 + f
        att |= ONE << U(s)
        if occ & (ONE << U(s)):
            break
        r += dr
        f += df
    return att


@njit(uint64(int64, int64, int64), cache=False)
def ray_mask(sq: Square, dr: Square, df: Square) -> Bits:
    """Relevant occupancy along one direction: everything but the final square.

    A piece on the edge cannot block anything beyond itself, so excluding it halves the
    table size without changing any answer.
    """
    m = ZERO
    r = sq // 8 + dr
    f = sq % 8 + df
    while True:
        if r < 0 or r > 7 or f < 0 or f > 7:
            break
        nr = r + dr
        nf = f + df
        if nr < 0 or nr > 7 or nf < 0 or nf > 7:
            break
        m |= ONE << U(r * 8 + f)
        r = nr
        f = nf
    return m


@njit(uint64(int64, uint64, boolean), cache=False)
def slider_attacks(sq: Square, occ: Bits, is_rook: bool) -> Bits:
    if is_rook:
        return (
            ray_attacks(sq, 1, 0, occ)
            | ray_attacks(sq, -1, 0, occ)
            | ray_attacks(sq, 0, 1, occ)
            | ray_attacks(sq, 0, -1, occ)
        )
    return (
        ray_attacks(sq, 1, 1, occ)
        | ray_attacks(sq, 1, -1, occ)
        | ray_attacks(sq, -1, 1, occ)
        | ray_attacks(sq, -1, -1, occ)
    )


@njit(uint64(int64, boolean), cache=False)
def slider_mask(sq: Square, is_rook: bool) -> Bits:
    if is_rook:
        return ray_mask(sq, 1, 0) | ray_mask(sq, -1, 0) | ray_mask(sq, 0, 1) | ray_mask(sq, 0, -1)
    return ray_mask(sq, 1, 1) | ray_mask(sq, 1, -1) | ray_mask(sq, -1, 1) | ray_mask(sq, -1, -1)


@njit(uint64(uint64[:]), cache=False, inline="always")
def splitmix(state: np.ndarray) -> Bits:
    """Seeded PRNG, so the magics we find are the same on every machine and every run."""
    state[0] += U(0x9E3779B97F4A7C15)
    z = state[0]
    z = (z ^ (z >> U(30))) * U(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> U(27))) * U(0x94D049BB133111EB)
    return z ^ (z >> U(31))


@njit(cache=False)
def build_magics(is_rook: bool, seed: int) -> tuple[np.ndarray, ...]:
    """Fancy magic bitboards, with the magic constants found here rather than copied in.

    For each square: enumerate every occupancy subset of the relevant mask, compute the
    true attack set for each, then look for a multiplier that maps all of them into a
    table without a contradictory collision.
    """
    rng = np.empty(1, dtype=np.uint64)
    rng[0] = U(seed)
    magic = np.zeros(64, dtype=np.uint64)
    mask = np.zeros(64, dtype=np.uint64)
    shift = np.zeros(64, dtype=np.int64)
    offset = np.zeros(64, dtype=np.int64)

    total = 0
    for sq in range(64):
        mask[sq] = slider_mask(sq, is_rook)
        bits = popcount(mask[sq])
        shift[sq] = 64 - bits
        offset[sq] = total
        total += 1 << bits

    table = np.zeros(total, dtype=np.uint64)

    for sq in range(64):
        m = mask[sq]
        bits = 64 - shift[sq]
        n = 1 << bits
        occs = np.zeros(n, dtype=np.uint64)
        atts = np.zeros(n, dtype=np.uint64)
        # Carry-rippler walk over every subset of the mask.
        sub = ZERO
        for i in range(n):
            occs[i] = sub
            atts[i] = slider_attacks(sq, sub, is_rook)
            sub = (sub - m) & m

        used = np.zeros(n, dtype=np.uint64)
        seen = np.zeros(n, dtype=np.bool_)
        while True:
            candidate = splitmix(rng) & splitmix(rng) & splitmix(rng)
            # A magic needs enough high bits set to spread the index across the table.
            if popcount((m * candidate) & U(0xFF00000000000000)) < 6:
                continue
            for i in range(n):
                seen[i] = False
            ok = True
            for i in range(n):
                idx = int((occs[i] * candidate) >> U(shift[sq]))
                if not seen[idx]:
                    seen[idx] = True
                    used[idx] = atts[i]
                elif used[idx] != atts[i]:
                    ok = False
                    break
            if ok:
                magic[sq] = candidate
                for i in range(n):
                    table[offset[sq] + i] = used[i]
                break

    return magic, mask, shift, offset, table


@njit(cache=False)
def build_leapers() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knight = np.zeros(64, dtype=np.uint64)
    king = np.zeros(64, dtype=np.uint64)
    pawn = np.zeros((2, 64), dtype=np.uint64)
    for sq in range(64):
        r = sq // 8
        f = sq % 8
        for dr, df in ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)):
            nr = r + dr
            nf = f + df
            if 0 <= nr <= 7 and 0 <= nf <= 7:
                knight[sq] |= ONE << U(nr * 8 + nf)
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                nr = r + dr
                nf = f + df
                if 0 <= nr <= 7 and 0 <= nf <= 7:
                    king[sq] |= ONE << U(nr * 8 + nf)
        for df in (-1, 1):
            nf = f + df
            if 0 <= nf <= 7:
                if r < 7:
                    pawn[0, sq] |= ONE << U((r + 1) * 8 + nf)
                if r > 0:
                    pawn[1, sq] |= ONE << U((r - 1) * 8 + nf)
    return knight, king, pawn


KNIGHT_ATT, KING_ATT, PAWN_ATT = build_leapers()
RMAGIC, RMASK, RSHIFT, ROFF, RTABLE = build_magics(True, 0x1234567)
BMAGIC, BMASK, BSHIFT, BOFF, BTABLE = build_magics(False, 0x89ABCDEF)


@njit(uint64(int64, uint64), cache=False, inline="always")
def rook_attacks(sq: Square, occ: Bits) -> Bits:
    return RTABLE[ROFF[sq] + int(((occ & RMASK[sq]) * RMAGIC[sq]) >> U(RSHIFT[sq]))]


@njit(uint64(int64, uint64), cache=False, inline="always")
def bishop_attacks(sq: Square, occ: Bits) -> Bits:
    return BTABLE[BOFF[sq] + int(((occ & BMASK[sq]) * BMAGIC[sq]) >> U(BSHIFT[sq]))]


@njit(uint64(int64, uint64), cache=False, inline="always")
def queen_attacks(sq: Square, occ: Bits) -> Bits:
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ)


# Castling rights survive a move unless its origin or destination is a king or rook home
# square. Masking on the destination is what handles a rook being captured where it stands.
CASTLE_MASK = np.full(64, ALL_CASTLING, dtype=np.int64)
CASTLE_MASK[0] = ALL_CASTLING ^ WHITE_QUEENSIDE  # a1
CASTLE_MASK[4] = ALL_CASTLING ^ (WHITE_KINGSIDE | WHITE_QUEENSIDE)  # e1
CASTLE_MASK[7] = ALL_CASTLING ^ WHITE_KINGSIDE  # h1
CASTLE_MASK[56] = ALL_CASTLING ^ BLACK_QUEENSIDE  # a8
CASTLE_MASK[60] = ALL_CASTLING ^ (BLACK_KINGSIDE | BLACK_QUEENSIDE)  # e8
CASTLE_MASK[63] = ALL_CASTLING ^ BLACK_KINGSIDE  # h8

_rng = np.random.default_rng(0x5EED)
Z_PIECE = _rng.integers(1, 2**64, size=(2, 6, 64), dtype=np.uint64)
Z_STM = _rng.integers(1, 2**64, size=1, dtype=np.uint64)
Z_CASTLE = _rng.integers(1, 2**64, size=16, dtype=np.uint64)
Z_EP = _rng.integers(1, 2**64, size=8, dtype=np.uint64)

# Warm every jitted function once, with the argument types the real calls use. numba
# compiles per signature, so this has to match or the compile lands on the clock instead.
popcount(U(0xFF))
lsb(U(1))
rook_attacks(0, U(0))
bishop_attacks(0, U(0))
queen_attacks(0, U(0))
