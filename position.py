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
from numba import boolean, int64, njit, uint64

from bitboard import (
    BISHOP,
    BLACK_KINGSIDE,
    BLACK_QUEENSIDE,
    BOCC,
    CASTLE,
    EP,
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


# Warm every jitted function with the argument types the real calls use.
_state, _mailbox = new_stacks()
encode(chess.Board(), _state[0], _mailbox[0])
ep_is_capturable(_state[0], 0)
full_key(_state[0])
king_square(_state[0], 0)
attacked(_state[0], 0, 0)
in_check(_state[0])
