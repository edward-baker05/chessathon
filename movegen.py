"""Pseudo-legal move generation and the move codec.

Moves are generated pseudo-legally and filtered afterwards by making them and testing
whether the mover left its king attacked. That is simpler and far less error-prone than
fully legal generation with pin masks, and the probe behind this design showed the cost is
affordable at over 10 Mnps.

A move is one int32: from | to << 6 | promo << 12 | flag << 15.
"""

from typing import Any

import chess
import numpy as np
from numba import int8, int32, int64, njit, uint64

import position
from bitboard import (
    BISHOP,
    BLACK_KINGSIDE,
    BLACK_QUEENSIDE,
    BOCC,
    CASTLE,
    EP,
    FLAG_CASTLE,
    FLAG_EP,
    FLAG_PROMO,
    KING,
    KING_ATT,
    KNIGHT,
    KNIGHT_ATT,
    ONE,
    PAWN,
    PAWN_ATT,
    QUEEN,
    ROOK,
    STM,
    WHITE_KINGSIDE,
    WHITE_QUEENSIDE,
    WOCC,
    ZERO,
    U,
    bishop_attacks,
    lsb,
    rook_attacks,
)
from position import attacked, legal_after, make

Bits = Any
Square = Any
Move = Any

MAX_MOVES = 256

# Promotion piece indices as stored in a move, 1..4.
PROMO_PIECES = "nbrq"

RANK_2 = U(0x000000000000FF00)
RANK_7 = U(0x00FF000000000000)

# Squares that must be empty to castle, and the square the king crosses.
WK_EMPTY, WQ_EMPTY = U(0x0000000000000060), U(0x000000000000000E)
BK_EMPTY, BQ_EMPTY = U(0x6000000000000000), U(0x0E00000000000000)


@njit(int64(int32), cache=False, inline="always")
def move_from(m: Move) -> Square:
    return m & 63


@njit(int64(int32), cache=False, inline="always")
def move_to(m: Move) -> Square:
    return (m >> 6) & 63


@njit(int64(int32), cache=False, inline="always")
def move_promo(m: Move) -> Square:
    return (m >> 12) & 7


@njit(int64(int32), cache=False, inline="always")
def move_flag(m: Move) -> Square:
    return (m >> 15) & 3


def pack(frm: int, to: int, promo: int = 0, flag: int = 0) -> int:
    """Plain-Python move packing, for tests and the agent boundary."""
    return frm | (to << 6) | (promo << 12) | (flag << 15)


def move_to_uci(m: int) -> str:
    """Plain-Python move rendering, for tests and the agent boundary."""
    text = chess.square_name(m & 63) + chess.square_name((m >> 6) & 63)
    if ((m >> 15) & 3) == FLAG_PROMO:
        text += PROMO_PIECES[((m >> 12) & 7) - 1]
    return text


@njit(int64(uint64[:], int32[:], int64), cache=False)
def generate(state: Bits, moves: Bits, base: Square) -> Square:
    """Every pseudo-legal move, written from `base`. Returns one past the last index."""
    n = base
    black = int64(state[STM])
    us = state[BOCC] if black else state[WOCC]
    them = state[WOCC] if black else state[BOCC]
    occ = us | them
    empty = ~occ

    pawns = state[PAWN] & us
    forward = -8 if black else 8
    start_rank = RANK_7 if black else RANK_2
    last_rank_low, last_rank_high = (0, 8) if black else (56, 64)

    bits = pawns
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        to = frm + forward
        if 0 <= to < 64 and (empty & (ONE << U(to))):
            if last_rank_low <= to < last_rank_high:
                for piece in range(1, 5):
                    moves[n] = frm | (to << 6) | (piece << 12) | (FLAG_PROMO << 15)
                    n += 1
            else:
                moves[n] = frm | (to << 6)
                n += 1
                double = frm + 2 * forward
                if (ONE << U(frm)) & start_rank and (empty & (ONE << U(double))):
                    moves[n] = frm | (double << 6)
                    n += 1
        captures = PAWN_ATT[black, frm] & them
        while captures:
            target = lsb(captures)
            captures &= captures - ONE
            if last_rank_low <= target < last_rank_high:
                for piece in range(1, 5):
                    moves[n] = frm | (target << 6) | (piece << 12) | (FLAG_PROMO << 15)
                    n += 1
            else:
                moves[n] = frm | (target << 6)
                n += 1
        if state[EP] != ZERO:
            ep_square = int64(state[EP]) - 1
            if PAWN_ATT[black, frm] & (ONE << U(ep_square)):
                moves[n] = frm | (ep_square << 6) | (FLAG_EP << 15)
                n += 1

    bits = state[KNIGHT] & us
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        targets = KNIGHT_ATT[frm] & ~us
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[n] = frm | (to << 6)
            n += 1

    bits = (state[BISHOP] | state[QUEEN]) & us
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        targets = bishop_attacks(frm, occ) & ~us
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[n] = frm | (to << 6)
            n += 1

    bits = (state[ROOK] | state[QUEEN]) & us
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        targets = rook_attacks(frm, occ) & ~us
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[n] = frm | (to << 6)
            n += 1

    ksq = lsb(state[KING] & us)
    targets = KING_ATT[ksq] & ~us
    while targets:
        to = lsb(targets)
        targets &= targets - ONE
        moves[n] = ksq | (to << 6)
        n += 1

    # Castling. The destination square is not checked here: the legality filter that runs
    # after make() catches a king landing in check.
    rights = int64(state[CASTLE])
    if black:
        if (
            (rights & BLACK_KINGSIDE)
            and not (occ & BK_EMPTY)
            and not attacked(state, 60, 0)
            and not attacked(state, 61, 0)
        ):
            moves[n] = 60 | (62 << 6) | (FLAG_CASTLE << 15)
            n += 1
        if (
            (rights & BLACK_QUEENSIDE)
            and not (occ & BQ_EMPTY)
            and not attacked(state, 60, 0)
            and not attacked(state, 59, 0)
        ):
            moves[n] = 60 | (58 << 6) | (FLAG_CASTLE << 15)
            n += 1
    else:
        if (
            (rights & WHITE_KINGSIDE)
            and not (occ & WK_EMPTY)
            and not attacked(state, 4, 1)
            and not attacked(state, 5, 1)
        ):
            moves[n] = 4 | (6 << 6) | (FLAG_CASTLE << 15)
            n += 1
        if (
            (rights & WHITE_QUEENSIDE)
            and not (occ & WQ_EMPTY)
            and not attacked(state, 4, 1)
            and not attacked(state, 3, 1)
        ):
            moves[n] = 4 | (2 << 6) | (FLAG_CASTLE << 15)
            n += 1
    return n


@njit(int64(uint64[:], int32[:], int64), cache=False)
def generate_captures(state: Bits, moves: Bits, base: Square) -> Square:
    """Captures and queen promotions only, for quiescence search."""
    n = base
    black = int64(state[STM])
    us = state[BOCC] if black else state[WOCC]
    them = state[WOCC] if black else state[BOCC]
    occ = us | them
    empty = ~occ

    forward = -8 if black else 8
    last_rank_low, last_rank_high = (0, 8) if black else (56, 64)

    bits = state[PAWN] & us
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        to = frm + forward
        # A quiet push is only worth searching here when it promotes.
        if 0 <= to < 64 and last_rank_low <= to < last_rank_high and (empty & (ONE << U(to))):
            moves[n] = frm | (to << 6) | (4 << 12) | (FLAG_PROMO << 15)
            n += 1
        captures = PAWN_ATT[black, frm] & them
        while captures:
            target = lsb(captures)
            captures &= captures - ONE
            if last_rank_low <= target < last_rank_high:
                moves[n] = frm | (target << 6) | (4 << 12) | (FLAG_PROMO << 15)
                n += 1
            else:
                moves[n] = frm | (target << 6)
                n += 1
        if state[EP] != ZERO:
            ep_square = int64(state[EP]) - 1
            if PAWN_ATT[black, frm] & (ONE << U(ep_square)):
                moves[n] = frm | (ep_square << 6) | (FLAG_EP << 15)
                n += 1

    bits = state[KNIGHT] & us
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        targets = KNIGHT_ATT[frm] & them
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[n] = frm | (to << 6)
            n += 1

    bits = (state[BISHOP] | state[QUEEN]) & us
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        targets = bishop_attacks(frm, occ) & them
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[n] = frm | (to << 6)
            n += 1

    bits = (state[ROOK] | state[QUEEN]) & us
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        targets = rook_attacks(frm, occ) & them
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[n] = frm | (to << 6)
            n += 1

    ksq = lsb(state[KING] & us)
    targets = KING_ATT[ksq] & them
    while targets:
        to = lsb(targets)
        targets &= targets - ONE
        moves[n] = ksq | (to << 6)
        n += 1

    return n


@njit(int64(uint64[:, :], int8[:, :], int32[:], int64, int64), cache=False)
def perft(state: Bits, mail: Bits, moves: Bits, ply: Square, depth: Square) -> Square:
    """Count legal move sequences. The correctness gate for the whole engine.

    Lives here rather than in position.py because it needs generate(), and movegen.py
    already imports position.py. Putting it the other way round is a circular import.
    """
    if depth == 0:
        return 1
    base = ply * MAX_MOVES
    count = generate(state[ply], moves, base)
    mover_black = int64(state[ply][STM])
    total = 0
    for i in range(base, count):
        make(state[ply], mail[ply], state[ply + 1], mail[ply + 1], moves[i])
        if legal_after(state[ply + 1], mover_black):
            total += perft(state, mail, moves, ply + 1, depth - 1)
    return total


# Warm every jitted function with the argument types the real calls use.
_state, _mailbox = position.new_stacks()
position.encode(chess.Board(), _state[0], _mailbox[0])
_moves = np.zeros(MAX_MOVES, dtype=np.int32)
generate(_state[0], _moves, 0)
generate_captures(_state[0], _moves, 0)
move_from(np.int32(0))
move_to(np.int32(0))
move_promo(np.int32(0))
move_flag(np.int32(0))
_perft_moves = np.zeros(64 * MAX_MOVES, dtype=np.int32)
perft(_state, _mailbox, _perft_moves, 0, 1)
