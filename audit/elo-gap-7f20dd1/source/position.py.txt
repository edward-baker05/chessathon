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
    BETWEEN,
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
    LINE,
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
    popcount,
    rook_attacks,
)

# As in bitboard.py, the numba decorator signature is the authoritative contract. Flag is
# used where numba returns its own boolean rather than a Python bool.
Bits = Any
Square = Any
Flag = Any

STACK_PLIES = 256

# Values SEE uses. Deliberately separate from the evaluation's own piece values: SEE is
# about whether an exchange wins material, not about how the position should be judged.
SEE_VALUE = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int32)

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
    them = state[WOCC] if black_to_move else state[BOCC]
    # Same inverted index as in attacked(): the squares a BLACK pawn could capture from
    # are the squares a WHITE pawn on ep_square would attack, so black uses PAWN_ATT[0].
    attackers = PAWN_ATT[0 if black_to_move else 1, ep_square] & state[PAWN] & us
    if attackers == ZERO:
        return False

    # Geometry is not enough. An en passant capture is the only move that vacates two
    # squares at once, so it can expose its own king along a rank or a diagonal that
    # neither square alone was blocking. A right that cannot legally be exercised is not
    # a right, and hashing it makes two identical positions hash differently.
    ksq = lsb(state[KING] & us)
    captured = ONE << U(ep_square + 8 if black_to_move else ep_square - 8)
    target = ONE << U(ep_square)
    occupancy = (state[WOCC] | state[BOCC]) & ~captured
    straight = them & (state[ROOK] | state[QUEEN])
    diagonal = them & (state[BISHOP] | state[QUEEN])
    while attackers != ZERO:
        after = (occupancy & ~(ONE << U(lsb(attackers)))) | target
        attackers &= attackers - ONE
        if (rook_attacks(ksq, after) & straight) != ZERO:
            continue
        if (bishop_attacks(ksq, after) & diagonal) != ZERO:
            continue
        return True
    return False


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
    us_colour = black
    them_colour = 1 - black
    from_bit = ONE << U(frm)
    to_bit = ONE << U(to)
    moved = int64(src_mail[frm])

    dst_state[HALF] = src_state[HALF] + ONE

    # Incremental Zobrist. A full recompute here costs a scan of all twelve bitboards per
    # node, which measured at roughly a third of the total node cost.
    key = src_state[KEY]
    key ^= Z_STM[0]
    if src_state[EP] != ZERO:
        old_ep = int64(src_state[EP]) - 1
        if ep_is_capturable(src_state, old_ep):
            key ^= Z_EP[old_ep % 8]

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
        key ^= Z_PIECE[them_colour, PAWN, capture_square]
        key ^= Z_PIECE[us_colour, PAWN, frm]
        key ^= Z_PIECE[us_colour, PAWN, to]
    else:
        captured = int64(src_mail[to])
        if captured >= 0:
            dst_state[captured] &= ~to_bit
            dst_state[them] &= ~to_bit
            dst_state[HALF] = ZERO
            key ^= Z_PIECE[them_colour, captured, to]
        if moved == PAWN:
            dst_state[HALF] = ZERO

        dst_state[moved] &= ~from_bit
        dst_state[us] = (dst_state[us] & ~from_bit) | to_bit
        dst_mail[frm] = -1
        key ^= Z_PIECE[us_colour, moved, frm]
        if flag == FLAG_PROMO:
            dst_state[promo] |= to_bit
            dst_mail[to] = int8(promo)
            key ^= Z_PIECE[us_colour, promo, to]
        else:
            dst_state[moved] |= to_bit
            dst_mail[to] = int8(moved)
            key ^= Z_PIECE[us_colour, moved, to]

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
            key ^= Z_PIECE[us_colour, ROOK, rook_from]
            key ^= Z_PIECE[us_colour, ROOK, rook_to]

    if moved == PAWN and (to - frm == 16 or frm - to == 16):
        dst_state[EP] = U((frm + to) // 2 + 1)
    else:
        dst_state[EP] = ZERO

    old_rights = int64(src_state[CASTLE])
    new_rights = old_rights & CASTLE_MASK[frm] & CASTLE_MASK[to]
    dst_state[CASTLE] = U(new_rights)
    key ^= Z_CASTLE[old_rights] ^ Z_CASTLE[new_rights]

    dst_state[STM] = U(1 - black)
    if black:
        dst_state[FULLMOVE] = src_state[FULLMOVE] + ONE

    # The new en passant term needs the finished position: ep_is_capturable reads the
    # updated pawns and the flipped side to move.
    if dst_state[EP] != ZERO:
        new_ep = int64(dst_state[EP]) - 1
        if ep_is_capturable(dst_state, new_ep):
            key ^= Z_EP[new_ep % 8]
    dst_state[KEY] = key


@njit(void(uint64[:], int8[:], uint64[:], int8[:]), cache=False)
def make_null(src_state: Bits, src_mail: Bits, dst_state: Bits, dst_mail: Bits) -> None:
    """Pass the move to the opponent. Used by null-move pruning.

    The halfmove clock is reset rather than carried. A null move breaks the alternating
    parity that repetition scanning relies on, and zeroing the clock bounds that scan to
    nothing inside the null subtree. The cost is that a fifty-move draw deep inside a null
    subtree can go unnoticed, which is conservative: it never invents a draw.
    """
    for i in range(NFIELDS):
        dst_state[i] = src_state[i]
    for i in range(64):
        dst_mail[i] = src_mail[i]

    key = src_state[KEY] ^ Z_STM[0]
    if src_state[EP] != ZERO:
        old_ep = int64(src_state[EP]) - 1
        if ep_is_capturable(src_state, old_ep):
            key ^= Z_EP[old_ep % 8]
    dst_state[EP] = ZERO
    dst_state[HALF] = ZERO
    dst_state[STM] = U(1 - int64(src_state[STM]))
    dst_state[KEY] = key


@njit(boolean(uint64[:], int64), cache=False, inline="always")
def legal_after(dst_state: Bits, mover_black: Square) -> Flag:
    """Did the side that just moved leave its own king attacked?"""
    return not attacked(dst_state, king_square(dst_state, mover_black), 1 - mover_black)


@njit(uint64(uint64[:], int64, uint64), cache=False)
def attackers_to(state: Bits, sq: Square, occ: Bits) -> Bits:
    """Every piece of either colour attacking `sq`, given an occupancy.

    Taking `occ` as an argument rather than reading it from the state is what makes x-ray
    detection work: SEE clears each consumed piece from `occ` and calls this again, which
    reveals sliders that were standing behind it.
    """
    return (
        (PAWN_ATT[0, sq] & state[PAWN] & state[BOCC])
        | (PAWN_ATT[1, sq] & state[PAWN] & state[WOCC])
        | (KNIGHT_ATT[sq] & state[KNIGHT])
        | (KING_ATT[sq] & state[KING])
        | (bishop_attacks(sq, occ) & (state[BISHOP] | state[QUEEN]))
        | (rook_attacks(sq, occ) & (state[ROOK] | state[QUEEN]))
    ) & occ


@njit(uint64(uint64[:], int64, uint64), cache=False)
def king_snipers(state: Bits, black: Square, exclude: Bits) -> Bits:
    """Enemy sliders that would see this king on an empty board, `exclude` left out.

    This is the half of a pin that cannot change while an exchange runs: the king stays
    where it is and no piece changes type or colour, so only whether a line is blocked
    varies from one capture to the next. Splitting it out is what lets `see` cache the
    expensive half and still count blockers against the occupancy it actually has.
    """
    us = state[BOCC] if black else state[WOCC]
    them = (state[WOCC] if black else state[BOCC]) & ~exclude
    ksq = lsb(state[KING] & us)
    return (rook_attacks(ksq, ZERO) & them & (state[ROOK] | state[QUEEN])) | (
        bishop_attacks(ksq, ZERO) & them & (state[BISHOP] | state[QUEEN])
    )


@njit(uint64(uint64[:], int64), cache=False)
def pinned_pieces(state: Bits, black: Square) -> Bits:
    """One side's pieces that stand alone between their king and an enemy slider.

    Each of them may still move, but only along the line it shares with its king, so a
    recapture on any square off that line is not available however the geometry looks.
    """
    # When there are no snipers there is no pin to look for, and the call costs two
    # lookups and stops.
    snipers = king_snipers(state, black, ZERO)
    if snipers == ZERO:
        return ZERO

    us = state[BOCC] if black else state[WOCC]
    ksq = lsb(state[KING] & us)
    occupancy = state[WOCC] | state[BOCC]
    pinned = ZERO
    while snipers != ZERO:
        blockers = BETWEEN[ksq, lsb(snipers)] & occupancy
        snipers &= snipers - ONE
        if popcount(blockers) == 1:
            pinned |= blockers
    return pinned & us


@njit(int32(uint64[:], int8[:], int32), cache=False)
def see(state: Bits, mail: Bits, move: Bits) -> Bits:
    """Static exchange evaluation: the material outcome of the capture sequence on `to`.

    The swap algorithm. Build the list of gains assuming both sides always recapture with
    their least valuable attacker, then walk it backwards applying the option not to
    continue, which is what turns a raw sequence into a value either side would accept.
    """
    frm = int64(move & 63)
    to = int64((move >> 6) & 63)
    flag = int64((move >> 15) & 3)

    # Castling never captures, and en passant is a pawn for a pawn on a square SEE would
    # have to special-case. Neither is worth the complexity here.
    if flag == FLAG_CASTLE:
        return int32(0)

    captured = int64(mail[to])
    gain = np.zeros(32, dtype=np.int32)
    gain[0] = SEE_VALUE[captured] if captured >= 0 else 0
    if flag == FLAG_EP:
        gain[0] = SEE_VALUE[PAWN]

    # `to` is set unconditionally because the mover now stands there. For a capture it was
    # occupied already; for a quiet move it was not, and leaving it clear lets a slider
    # trace straight through the square the moving piece is standing on, which is how the
    # pin test below would miss a line the move itself has just blocked.
    occ = ((state[WOCC] | state[BOCC]) & ~(ONE << U(frm))) | (ONE << U(to))
    if flag == FLAG_EP:
        occ &= ~(ONE << U(to + 8 if state[STM] else to - 8))

    on_square = int64(mail[frm])
    if flag == FLAG_PROMO:
        # The pawn is gone: what stands on `to` and what the opponent can win back is the
        # promoted piece, and the promotion itself is worth the difference in material.
        # Valuing the mover as a pawn scores an uncontested queening at zero.
        on_square = int64((move >> 12) & 7)
        gain[0] += SEE_VALUE[on_square] - SEE_VALUE[PAWN]

    side_black = int64(state[STM])
    attacks = attackers_to(state, to, occ)

    # Which enemy sliders see each king is fixed for the whole exchange, so it is computed
    # at most once per side and only for the sides that are actually asked. Whether any of
    # them is pinning is not fixed, and is counted below against the occupancy the exchange
    # has reached. Reading a finished pin set once, as this did, is wrong in both
    # directions: the piece that left `frm` may have been the pinner, so a pin can vanish,
    # and a blocker may be consumed as a recapture, so a pin can appear. The first way
    # round is the damaging one, because it hides a legal recapture and turns a losing
    # capture into a winning one.
    #
    # The square `to` is left out of both sniper sets. Whatever stands there is the piece
    # being captured, and capturing the pinner is always legal: the capturer ends up on the
    # pinner's own square, still on the line, still blocking it.
    target = ONE << U(to)
    snipers_white = ZERO
    snipers_black = ZERO
    known_white = False
    known_black = False

    depth = 0
    while True:
        side_black = 1 - side_black
        side_pieces = state[BOCC] if side_black else state[WOCC]
        mine = attacks & side_pieces & occ
        if mine == ZERO:
            break

        if side_black:
            if not known_black:
                snipers_black = king_snipers(state, 1, target)
                known_black = True
            snipers = snipers_black
        else:
            if not known_white:
                snipers_white = king_snipers(state, 0, target)
                known_white = True
            snipers = snipers_white
        # A king that has already recaptured no longer stands on its own square, so the
        # geometry cached above no longer describes it and the filter is skipped.
        king_bits = state[KING] & side_pieces & occ
        if snipers != ZERO and king_bits != ZERO:
            ksq = lsb(king_bits)
            rest = snipers
            while rest != ZERO:
                sniper = lsb(rest)
                rest &= rest - ONE
                # A pinner that has been captured, or that was the piece which moved to
                # `to` in the first place, is no longer pinning anything.
                if (occ >> U(sniper)) & ONE == ZERO:
                    continue
                blocked = BETWEEN[ksq, sniper] & occ
                if popcount(blocked) != 1:
                    continue
                square = lsb(blocked)
                # A pinned piece may capture on `to` only if `to` lies on the line it
                # already shares with its king.
                if (LINE[ksq, square] >> U(to)) & ONE == ZERO:
                    mine &= ~blocked
            if mine == ZERO:
                break

        # Recapture with the least valuable attacker available.
        piece = -1
        for candidate in range(6):
            if mine & state[candidate]:
                piece = candidate
                break
        if piece < 0:
            break

        square = lsb(mine & state[piece])
        # A king may only recapture onto a square the other side has stopped attacking,
        # and a pinned defender still controls its squares, so this cannot be folded into
        # the pin filter above. The attacker set is re-derived without the king because a
        # slider standing behind it attacks through the square it is leaving.
        if piece == KING:
            vacated = occ & ~(ONE << U(square))
            enemy = state[WOCC] if side_black else state[BOCC]
            if (attackers_to(state, to, vacated) & enemy) != ZERO:
                break

        depth += 1
        if depth >= 31:
            break
        # A recapturing pawn that lands on the last rank queens, exactly as the opening
        # move does. Only a pawn of the right colour can reach either back rank at all, so
        # the rank alone settles it. Left out, the exchange keeps calling the new queen a
        # pawn: it under-counts the material by the promotion and then offers the opponent
        # a pawn to win back instead of a queen.
        promotes = piece == PAWN and (to >> 3 == 7 or to >> 3 == 0)
        gain[depth] = SEE_VALUE[on_square] - gain[depth - 1]
        if promotes:
            gain[depth] += SEE_VALUE[QUEEN] - SEE_VALUE[PAWN]

        occ &= ~(ONE << U(square))
        on_square = QUEEN if promotes else piece
        # Re-derive attackers so sliders behind the piece just consumed are included.
        attacks = attackers_to(state, to, occ)

    # Walk back: at each point the side to move could simply decline the exchange.
    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1
    return int32(gain[0])


@njit(boolean(uint64[:], int64), cache=False, inline="always")
def has_non_pawn_material(state: Bits, black: Square) -> Flag:
    """Null move is unsafe without this: a side with only pawns can be in zugzwang."""
    side = state[BOCC] if black else state[WOCC]
    return (side & (state[KNIGHT] | state[BISHOP] | state[ROOK] | state[QUEEN])) != ZERO


# The light squares. A bishop can never leave the colour it stands on, so bishops that
# share one of these two halves can never attack each other or force anything.
LIGHT_SQUARES = U(0x55AA55AA55AA55AA)


@njit(boolean(uint64[:]), cache=False)
def insufficient_material(state: Bits) -> Flag:
    """Positions in which neither side can construct a mate. Draws under FIDE rules."""
    if state[PAWN] | state[ROOK] | state[QUEEN]:
        return False
    minors = state[KNIGHT] | state[BISHOP]
    if popcount(minors) <= 1:
        return True
    # Any number of bishops, on either side, all standing on one colour. Without a knight
    # to change the colour of the mating net this is dead however many there are.
    if state[KNIGHT] == ZERO:
        bishops = state[BISHOP]
        return (bishops & LIGHT_SQUARES) == ZERO or (bishops & ~LIGHT_SQUARES) == ZERO
    return False


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
make_null(_state[0], _mailbox[0], _state[1], _mailbox[1])
attackers_to(_state[0], 0, _state[0][WOCC] | _state[0][BOCC])
king_snipers(_state[0], 0, ZERO)
pinned_pieces(_state[0], 0)
see(_state[0], _mailbox[0], _warm_move)
has_non_pawn_material(_state[0], 0)
insufficient_material(_state[0])
