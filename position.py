"""Position state, encoding and attack detection.

The state of one position is a row of `NFIELDS` uint64 plus a 64-entry int8 mailbox. Both
live in preallocated per-ply stacks so the search never allocates. Colour is read from the
occupancy bitboards, so the mailbox only has to carry the piece type.

En passant is stored as square + 1, with 0 meaning none. The state vector is unsigned, so
a -1 sentinel would compare as a huge positive number.
"""

from typing import Any

import chess
import numpy as np
from numba import boolean, int8, int32, int64, njit, uint64, void

from bitboard import (
    BISHOP,
    BLACK_KINGSIDE,
    BLACK_QUEENSIDE,
    BOCC,
    CASTLE,
    CASTLE_MASK,
    EP,
    FLAG_CASTLE,
    FLAG_EP,
    FLAG_PROMO,
    FULLMOVE,
    HALF,
    KEY,
    KING,
    KING_ATT,
    KNIGHT,
    KNIGHT_ATT,
    NFIELDS,
    ONE,
    PAWN,
    PAWN_ATT,
    QUEEN,
    ROOK,
    STM,
    WHITE_KINGSIDE,
    WHITE_QUEENSIDE,
    WOCC,
    Z_CASTLE,
    Z_EP,
    Z_PIECE,
    Z_STM,
    ZERO,
    U,
    bishop_attacks,
    lsb,
    rook_attacks,
)

# As in bitboard.py, the numba decorator signature is the authoritative contract. Flag is
# used where numba returns its own boolean rather than a Python bool.
Bits = Any
Square = Any
Flag = Any

STACK_PLIES = 256

# python-chess piece types are 1..6; ours are 0..5. Kept as a named constant so the
# off-by-one is stated once rather than scattered.
PIECE_OFFSET = 1


def new_stacks() -> tuple[np.ndarray, np.ndarray]:
    """Preallocated per-ply state and mailbox stacks."""
    return (
        np.zeros((STACK_PLIES, NFIELDS), dtype=np.uint64),
        np.zeros((STACK_PLIES, 64), dtype=np.int8),
    )


@njit(boolean(uint64[:], int64), cache=False)
def ep_is_capturable(state: Bits, ep_square: Square) -> Flag:
    """True when an enemy pawn could actually make the en passant capture.

    Hashing the en passant square unconditionally makes otherwise identical positions hash
    differently, which quietly destroys transposition table hit rates.
    """
    black_to_move = state[STM] != ZERO
    us = state[BOCC] if black_to_move else state[WOCC]
    # A pawn of ours can capture onto ep_square exactly when it stands on a square that
    # attacks it, which is what the opposite colour's attack table from ep_square gives.
    attackers = PAWN_ATT[1 if black_to_move else 0, ep_square]
    return (attackers & state[PAWN] & us) != ZERO


@njit(uint64(uint64[:]), cache=False)
def full_key(state: Bits) -> Bits:
    """Recompute the Zobrist key from scratch. The reference for incremental updates."""
    key = ZERO
    for piece in range(6):
        for colour in range(2):
            bits = state[piece] & (state[BOCC] if colour else state[WOCC])
            while bits:
                square = lsb(bits)
                bits &= bits - ONE
                key ^= Z_PIECE[colour, piece, square]
    if state[STM] != ZERO:
        key ^= Z_STM[0]
    key ^= Z_CASTLE[int(state[CASTLE])]
    if state[EP] != ZERO:
        ep_square = int(state[EP]) - 1
        if ep_is_capturable(state, ep_square):
            key ^= Z_EP[ep_square % 8]
    return key


@njit(int64(uint64[:], int64), cache=False, inline="always")
def king_square(state: Bits, black: Square) -> Square:
    return lsb(state[KING] & (state[BOCC] if black else state[WOCC]))


@njit(boolean(uint64[:], int64, int64), cache=False)
def attacked(state: Bits, sq: Square, by_black: Square) -> Flag:
    """Is `sq` attacked by the given colour? Colour comes from the occupancy bitboards."""
    occ = state[WOCC] | state[BOCC]
    them = state[BOCC] if by_black else state[WOCC]
    # PAWN_ATT[0, sq] is where a white pawn on sq attacks, which is exactly the set of
    # squares a black pawn would have to stand on to attack sq. The index is inverted
    # relative to intuition, and getting it backwards is a rarely triggered bug.
    if PAWN_ATT[0 if by_black else 1, sq] & state[PAWN] & them:
        return True
    if KNIGHT_ATT[sq] & state[KNIGHT] & them:
        return True
    if KING_ATT[sq] & state[KING] & them:
        return True
    if bishop_attacks(sq, occ) & (state[BISHOP] | state[QUEEN]) & them:
        return True
    return (rook_attacks(sq, occ) & (state[ROOK] | state[QUEEN]) & them) != ZERO


@njit(boolean(uint64[:]), cache=False)
def in_check(state: Bits) -> Flag:
    black = int64(state[STM])
    return attacked(state, king_square(state, black), 1 - black)


def encode(board: chess.Board, state: np.ndarray, mailbox: np.ndarray) -> None:
    """Write a python-chess board into one state row and one mailbox row.

    Plain Python: this runs once per move at the boundary, not inside the search.
    """
    state[PAWN] = board.pawns
    state[KNIGHT] = board.knights
    state[BISHOP] = board.bishops
    state[ROOK] = board.rooks
    state[QUEEN] = board.queens
    state[KING] = board.kings
    state[WOCC] = board.occupied_co[chess.WHITE]
    state[BOCC] = board.occupied_co[chess.BLACK]
    state[STM] = 0 if board.turn else 1

    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= WHITE_KINGSIDE
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= WHITE_QUEENSIDE
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= BLACK_KINGSIDE
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= BLACK_QUEENSIDE
    state[CASTLE] = rights

    state[EP] = (board.ep_square + 1) if board.ep_square is not None else 0
    state[HALF] = board.halfmove_clock
    state[FULLMOVE] = board.fullmove_number

    mailbox[:] = -1
    for square, piece in board.piece_map().items():
        mailbox[square] = piece.piece_type - PIECE_OFFSET

    state[KEY] = full_key(state)


def decode_fen(state: np.ndarray, mailbox: np.ndarray) -> str:
    """Render a state row back to a FEN. Used by tests and when debugging a game."""
    white = int(state[WOCC])
    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            square = rank * 8 + file
            piece = int(mailbox[square])
            if piece < 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            symbol = chess.piece_symbol(piece + PIECE_OFFSET)
            row += symbol.upper() if white & (1 << square) else symbol
        if empty:
            row += str(empty)
        rows.append(row)

    rights = int(state[CASTLE])
    castling = ""
    if rights & WHITE_KINGSIDE:
        castling += "K"
    if rights & WHITE_QUEENSIDE:
        castling += "Q"
    if rights & BLACK_KINGSIDE:
        castling += "k"
    if rights & BLACK_QUEENSIDE:
        castling += "q"

    ep = int(state[EP])
    return (
        f"{'/'.join(rows)} {'b' if state[STM] else 'w'} {castling or '-'} "
        f"{chess.square_name(ep - 1) if ep else '-'} "
        f"{int(state[HALF])} {int(state[FULLMOVE])}"
    )


@njit(void(uint64[:], int8[:], uint64[:], int8[:], int32), cache=False)
def make(
    src_state: Bits, src_mail: Bits, dst_state: Bits, dst_mail: Bits, move: Bits
) -> None:
    """Copy-make. Writes the position after `move` into the destination row.

    Copy-make rather than unmake: the probe behind this design reached 11.7 Mnps while
    copying state, so roughly 190 bytes per node is not the bottleneck, and it removes an
    entire class of state-restoration bugs.
    """
    for i in range(NFIELDS):
        dst_state[i] = src_state[i]
    for i in range(64):
        dst_mail[i] = src_mail[i]

    frm = int64(move & 63)
    to = int64((move >> 6) & 63)
    promo = int64((move >> 12) & 7)
    flag = int64((move >> 15) & 3)

    black = int64(src_state[STM])
    us = BOCC if black else WOCC
    them = WOCC if black else BOCC
    from_bit = ONE << U(frm)
    to_bit = ONE << U(to)
    moved = int64(src_mail[frm])

    dst_state[HALF] = src_state[HALF] + ONE

    if flag == FLAG_EP:
        # The captured pawn sits beside the target square, not behind it: a black pawn
        # capturing onto e3 removes the white pawn on e4. Getting this sign backwards was
        # the single bug behind every perft mismatch in the design probe, and it only
        # shows up in positions with a horizontal pin.
        capture_square = to + 8 if black else to - 8
        capture_bit = ONE << U(capture_square)
        dst_state[PAWN] &= ~capture_bit
        dst_state[them] &= ~capture_bit
        dst_mail[capture_square] = -1
        dst_state[PAWN] = (dst_state[PAWN] & ~from_bit) | to_bit
        dst_state[us] = (dst_state[us] & ~from_bit) | to_bit
        dst_mail[frm] = -1
        dst_mail[to] = PAWN
        dst_state[HALF] = ZERO
    else:
        captured = int64(src_mail[to])
        if captured >= 0:
            dst_state[captured] &= ~to_bit
            dst_state[them] &= ~to_bit
            dst_state[HALF] = ZERO
        if moved == PAWN:
            dst_state[HALF] = ZERO

        dst_state[moved] &= ~from_bit
        dst_state[us] = (dst_state[us] & ~from_bit) | to_bit
        dst_mail[frm] = -1
        if flag == FLAG_PROMO:
            dst_state[promo] |= to_bit
            dst_mail[to] = int8(promo)
        else:
            dst_state[moved] |= to_bit
            dst_mail[to] = int8(moved)

        if flag == FLAG_CASTLE:
            if to == 6:
                rook_from, rook_to = 7, 5
            elif to == 2:
                rook_from, rook_to = 0, 3
            elif to == 62:
                rook_from, rook_to = 63, 61
            else:
                rook_from, rook_to = 56, 59
            rook_from_bit = ONE << U(rook_from)
            rook_to_bit = ONE << U(rook_to)
            dst_state[ROOK] = (dst_state[ROOK] & ~rook_from_bit) | rook_to_bit
            dst_state[us] = (dst_state[us] & ~rook_from_bit) | rook_to_bit
            dst_mail[rook_from] = -1
            dst_mail[rook_to] = ROOK

    if moved == PAWN and (to - frm == 16 or frm - to == 16):
        dst_state[EP] = U((frm + to) // 2 + 1)
    else:
        dst_state[EP] = ZERO

    dst_state[CASTLE] = U(int64(src_state[CASTLE]) & CASTLE_MASK[frm] & CASTLE_MASK[to])
    dst_state[STM] = U(1 - black)
    if black:
        dst_state[FULLMOVE] = src_state[FULLMOVE] + ONE
    dst_state[KEY] = full_key(dst_state)


@njit(boolean(uint64[:], int64), cache=False, inline="always")
def legal_after(dst_state: Bits, mover_black: Square) -> Flag:
    """Did the side that just moved leave its own king attacked?"""
    return not attacked(dst_state, king_square(dst_state, mover_black), 1 - mover_black)


# Warm every jitted function with the argument types the real calls use.
_state, _mailbox = new_stacks()
encode(chess.Board(), _state[0], _mailbox[0])
ep_is_capturable(_state[0], 0)
full_key(_state[0])
king_square(_state[0], 0)
attacked(_state[0], 0, 0)
in_check(_state[0])
_warm_move = np.int32((8) | (16 << 6))
make(_state[0], _mailbox[0], _state[1], _mailbox[1], _warm_move)
legal_after(_state[1], 0)
