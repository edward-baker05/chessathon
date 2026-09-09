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
from numba import boolean, int8, int32, int64, njit, uint64

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
from position import attacked, attackers_to, legal_after, make

Bits = Any
Square = Any
Move = Any

MAX_MOVES = 256

# Promotion piece indices as stored in a move, 1..4.
PROMO_PIECES = "nbrq"

RANK_2 = U(0x000000000000FF00)
RANK_7 = U(0x00FF000000000000)
FILE_A = U(0x0101010101010101)
FILE_H = U(0x8080808080808080)

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


@njit(boolean(uint64[:]), cache=False)
def has_free_move(state: Bits) -> Any:
    """True only when the side to move certainly has a legal move. Never makes a move.

    Assumes the side to move is not in check, which the callers all establish first.
    Under that assumption a piece that is neither the king nor pinned to it can always
    move: the king is not attacked now, and a piece standing on no pin line cannot expose
    it. So one such piece with anywhere at all to go settles the question. En passant is
    the one capture that can expose a king from off the pin lines, and is left out here.

    A pinned piece stands between its king and a slider, so it stands on a rank, file or
    diagonal through the king. Everything off those lines is unpinned, and that is the
    test used here: two occupancy-free slider lookups, rather than the sniper walk that
    `pinned_pieces` needs to say exactly which pieces are pinned. It excludes some pieces
    that are not pinned either, which costs nothing but completeness.

    False means "not certain", never "no move": the king, the pinned pieces and everything
    else standing on a king line have moves of their own that this does not look at. The
    caller falls back on `has_legal_move`, which is exact. That is what makes a piece-count
    gate unnecessary: this is the cheap path, and the exact test is what decides.
    """
    black = int64(state[STM])
    us = state[BOCC] if black else state[WOCC]
    them = state[WOCC] if black else state[BOCC]
    occ = us | them
    ksq = lsb(state[KING] & us)
    king_lines = rook_attacks(ksq, ZERO) | bishop_attacks(ksq, ZERO)
    # Without a slider looking down one of those lines nothing is pinned at all, and the
    # whole side counts as free. That is the usual case in an endgame, and it is what
    # keeps the loops below short there.
    if king_lines & them & (state[BISHOP] | state[ROOK] | state[QUEEN]):
        free = us & ~king_lines & ~state[KING]
    else:
        free = us & ~state[KING]

    # Pawns first, and all of them at once: a push is a shift and a mask, and in most
    # positions some pawn has a square in front of it, which answers the question before
    # anything has been looked up.
    pawns = state[PAWN] & free
    if black:
        if (pawns >> U(8)) & ~occ:
            return True
        if (((pawns & ~FILE_A) >> U(9)) | ((pawns & ~FILE_H) >> U(7))) & them:
            return True
    else:
        if (pawns << U(8)) & ~occ:
            return True
        if (((pawns & ~FILE_H) << U(9)) | ((pawns & ~FILE_A) << U(7))) & them:
            return True

    bits = state[KNIGHT] & free
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        if KNIGHT_ATT[frm] & ~us:
            return True

    bits = (state[BISHOP] | state[QUEEN]) & free
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        if bishop_attacks(frm, occ) & ~us:
            return True

    bits = (state[ROOK] | state[QUEEN]) & free
    while bits:
        frm = lsb(bits)
        bits &= bits - ONE
        if rook_attacks(frm, occ) & ~us:
            return True

    # The king last, because it is the one piece whose moves have to be checked square by
    # square. It is also the piece that answers the question in an endgame with a handful
    # of men, where every other piece is blocked or standing on a king line: without this
    # a king and pawn ending falls through to full generation at nearly every leaf.
    #
    # The king is lifted out of the occupancy first. A slider otherwise stops on the
    # square the king stands on, and the square behind it reads as safe when stepping
    # there walks straight down the same line.
    lifted = occ ^ (ONE << U(ksq))
    targets = KING_ATT[ksq] & ~occ
    while targets:
        to = lsb(targets)
        targets &= targets - ONE
        if not (attackers_to(state, to, lifted) & them):
            return True

    return False


@njit(boolean(uint64[:], int8[:], uint64[:], int8[:], int32[:]), cache=False)
def has_legal_move(state: Bits, mail: Bits, dst_state: Bits, dst_mail: Bits,
                   moves: Bits) -> Any:
    """Does the side to move have any legal move at all?

    Stalemate and checkmate are the same question asked either side of `in_check`, and
    neither can be answered from a capture list. Returns on the first legal move, which
    in a position that is not close to terminal is almost always the first generated one.
    """
    count = generate(state, moves, 0)
    mover_black = int64(state[STM])
    for i in range(count):
        make(state, mail, dst_state, dst_mail, moves[i])
        if legal_after(dst_state, mover_black):
            return True
    return False


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
has_free_move(_state[0])
has_legal_move(_state[0], _mailbox[0], _state[1], _mailbox[1], _moves)
_perft_moves = np.zeros(64 * MAX_MOVES, dtype=np.int32)
perft(_state, _mailbox, _perft_moves, 0, 1)
