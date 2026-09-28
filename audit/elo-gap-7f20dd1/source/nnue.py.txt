"""Quantised NNUE evaluation: weights, features, accumulator and forward pass.

The network is `768 -> L1x2 -> 1` with squared clipped ReLU and eight output buckets, the
shape modern CPU engines use. It is deliberately not the classic Stockfish shape with a
deep output stack: a `2*L1 -> 16 -> 32 -> 1` stack measured at 1386 ns per node here
against 87 ns for a single output row, which would more than double the cost of a node.

Three numba facts govern the structure of this file. All three were measured, and each one
cost a factor of between six and thirty six when it was got wrong.

1. numba only vectorises loops over C-contiguous arrays. The same update loop typed
   `int16[:, :]` ran at 261 ns and typed `int16[::1]` at 22 ns.

2. A hot loop must live in its own jitted function with a **fully explicit signature**.
   The identical arithmetic written inline in a function that also slices its arguments
   compiled to scalar code and ran at 522 ns; split out behind an explicit
   `int32(ROW, ROW, RO_ROW)` signature it vectorises and runs at 87 ns. Lazy compilation,
   `inline="always"` and hand-inlining all measured slow. This is why every loop below is
   a small function with a spelled-out signature, and why the wrappers that slice do no
   arithmetic. It looks like indirection. It is a 6x speedup, and `tests/test_nnue.py`
   guards it.

3. Rows sliced from the module-level weight arrays are `readonly` to numba, which is a
   different type from a writable row. A signature that does not say so fails to compile,
   or worse, forces a lazily typed fallback. Hence `RO_ROW`.

`np.dot` is unavailable: numba's linear algebra needs scipy, which the platform does not
ship. numpy slice expressions inside njit allocate and measured 18x slower than a loop.

The output sum accumulates in int32. int64 measured at 516 ns against 81 ns, because the
vector unit carries two int64 lanes against eight int16. int32 can overflow in principle,
so `tools/quantise.py` proves it cannot for the specific weights being shipped and refuses
to write a file it cannot prove safe.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numba import int8, int16, int32, int64, njit, uint64, void
from numba.core import types

from bitboard import BOCC, ONE, PAWN, ROOK, STM, WOCC, U, popcount

Bits = Any
Square = Any

# A writable accumulator row, and a readonly row sliced out of a weight array. numba
# treats these as distinct types and will not vectorise a loop it cannot type exactly.
# numba's type constructors carry no annotations, so mypy cannot check this call.
ROW: Any = types.Array(types.int16, 1, "C")  # type: ignore[no-untyped-call]
RO_ROW: Any = types.Array(types.int16, 1, "C", readonly=True)  # type: ignore[no-untyped-call]

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "net.npz"

# Piece-square-colour features, six piece types by two colours by sixty four squares.
NUM_FEATURES = 768
# One extra all-zero row past the real features. Pointing an unused slot at it lets every
# move share one "subtract two, add two" update instead of branching per move flag.
ZERO_FEATURE = NUM_FEATURES

# An evaluation is clamped to this. The search reserves scores near MATE for real mates,
# and a net that produced one would be read as a forced win that does not exist.
EVAL_LIMIT = 10000


def _load(path: Path) -> dict[str, np.ndarray]:
    """Read the shipped network, or fail loudly.

    There is deliberately no fallback to the material evaluation. A silent fallback would
    pass validation and then play an entire rated round hundreds of Elo weak, and would
    look like an unexplained rating drop rather than a broken upload. Raising here puts
    the reason in the validation log before the build ever plays a rated game.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The agent ships its network in weights/net.npz; "
            f"build one with tools/quantise.py"
        )
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    required = ("ft_weight", "ft_bias", "out_weight", "out_bias", "qa", "qb", "scale")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}")
    if arrays["ft_weight"].shape[0] != NUM_FEATURES:
        raise ValueError(
            f"{path} has {arrays['ft_weight'].shape[0]} input features, expected {NUM_FEATURES}"
        )
    hidden = int(arrays["ft_weight"].shape[1])
    if arrays["ft_bias"].shape != (hidden,):
        raise ValueError(f"{path} feature bias is {arrays['ft_bias'].shape}, expected ({hidden},)")
    if arrays["out_weight"].shape[1] != 2 * hidden:
        raise ValueError(
            f"{path} output weight is {arrays['out_weight'].shape}, "
            f"expected (buckets, {2 * hidden})"
        )
    if arrays["out_weight"].shape[0] != arrays["out_bias"].shape[0]:
        raise ValueError(f"{path} has mismatched output weight and bias bucket counts")
    return arrays


_NET = _load(WEIGHTS_PATH)

# These are read before the jitted functions below are defined, so numba compiles them as
# literals and every loop bound is known at compile time. That is what lets LLVM unroll
# and vectorise the inner loops.
L1: int = int(_NET["ft_bias"].shape[0])
BUCKETS: int = int(_NET["out_bias"].shape[0])
QA: int = int(_NET["qa"])
QB: int = int(_NET["qb"])
SCALE: int = int(_NET["scale"])

# The zero row is appended here rather than saved in the file, so the file holds only real
# weights and the trainer never has to know about this trick.
FT_WEIGHT = np.ascontiguousarray(
    np.vstack([_NET["ft_weight"], np.zeros((1, L1), dtype=np.int16)]), dtype=np.int16
)
FT_BIAS = np.ascontiguousarray(_NET["ft_bias"], dtype=np.int16)
OUT_WEIGHT = np.ascontiguousarray(_NET["out_weight"], dtype=np.int16)
OUT_BIAS = np.ascontiguousarray(_NET["out_bias"], dtype=np.int32)

# Buckets are chosen by piece count. Two kings are always on the board, so the count runs
# 2 to 32 and this maps it onto 0 to BUCKETS-1.
BUCKET_DIVISOR = (32 - 2) // BUCKETS + 1


# --------------------------------------------------------------------------------------
# The hot loops. Each one is its own function with a fully explicit signature, for the
# reason given at the top of this file. Do not inline these into their callers.
# --------------------------------------------------------------------------------------


@njit(void(ROW, RO_ROW), cache=False)
def _set(row: Bits, bias: Bits) -> None:
    for i in range(L1):
        row[i] = bias[i]


@njit(void(ROW, RO_ROW), cache=False)
def _add(row: Bits, weights: Bits) -> None:
    for i in range(L1):
        row[i] += weights[i]


@njit(void(ROW, ROW), cache=False)
def _copy_row(dst: Bits, src: Bits) -> None:
    for i in range(L1):
        dst[i] = src[i]


@njit(void(ROW, ROW, RO_ROW, RO_ROW), cache=False)
def _move_one(dst: Bits, src: Bits, sub: Bits, add: Bits) -> None:
    """A quiet move: one feature leaves, one arrives. The majority of updates."""
    for i in range(L1):
        dst[i] = src[i] - sub[i] + add[i]


@njit(void(ROW, ROW, RO_ROW, RO_ROW, RO_ROW, RO_ROW), cache=False)
def _move_two(dst: Bits, src: Bits, sub0: Bits, sub1: Bits, add0: Bits, add1: Bits) -> None:
    """A capture, castle or promotion: up to two features leave and two arrive."""
    for i in range(L1):
        dst[i] = src[i] - sub0[i] - sub1[i] + add0[i] + add1[i]


@njit(int32(ROW, ROW, RO_ROW), cache=False)
def _dot(us: Bits, them: Bits, weights: Bits) -> Bits:
    """Squared clipped ReLU against one output row, side to move first.

    int32 rather than int64 accumulation: int64 halves the vector width and measured six
    times slower. `tools/quantise.py` proves the shipped weights cannot overflow this.
    """
    total = np.int32(0)
    for i in range(L1):
        v = np.int32(min(max(us[i], np.int16(0)), np.int16(QA)))
        total += v * v * weights[i]
    for i in range(L1):
        v = np.int32(min(max(them[i], np.int16(0)), np.int16(QA)))
        total += v * v * weights[L1 + i]
    return total


# --------------------------------------------------------------------------------------
# The wrappers. These slice and decode; they contain no loop over L1.
# --------------------------------------------------------------------------------------


@njit(int32(int64, int64, int64, int64), cache=False, inline="always")
def feature(perspective: Square, colour: Square, piece: Square, square: Square) -> Bits:
    """Index of one piece-on-square feature, as seen from one side.

    `perspective` is 0 for white and 1 for black. From black's side the board is flipped
    vertically and the colours are swapped, so that "my pawn on my second rank" is the
    same feature for both players and the two halves of the network share weights.
    """
    return np.int32(((colour ^ perspective) * 6 + piece) * 64 + (square ^ (perspective * 56)))


@njit(void(int16[:, :, ::1], int64, uint64[:], int8[:]), cache=False)
def refresh(acc: Bits, ply: Square, state: Bits, mail: Bits) -> None:
    """Rebuild both accumulators from the board. Called once per search, at the root."""
    _set(acc[ply, 0], FT_BIAS)
    _set(acc[ply, 1], FT_BIAS)
    black_occupancy = state[BOCC]
    for square in range(64):
        piece = np.int64(mail[square])
        if piece < 0:
            continue
        colour = np.int64(1) if ((black_occupancy >> U(square)) & ONE) != 0 else np.int64(0)
        _add(acc[ply, 0], FT_WEIGHT[feature(0, colour, piece, square)])
        _add(acc[ply, 1], FT_WEIGHT[feature(1, colour, piece, square)])


@njit(void(int16[:, :, ::1], int64), cache=False)
def copy(acc: Bits, ply: Square) -> None:
    """Carry the accumulator across a null move, which changes no piece."""
    _copy_row(acc[ply + 1, 0], acc[ply, 0])
    _copy_row(acc[ply + 1, 1], acc[ply, 1])


@njit(void(int16[:, :, ::1], int64, uint64[:], int8[:], int32), cache=False)
def apply(acc: Bits, ply: Square, state: Bits, mail: Bits, move: Bits) -> None:
    """Write ply+1 from ply for one move, without rebuilding from the board.

    `state` and `mail` are the position *before* the move. Everything the update needs is
    derivable from them, which is why this can sit in the search after the legality check
    rather than inside `position.make`: `position.py` stays free of evaluation concerns,
    and pseudo-legal moves that turn out illegal cost nothing.

    At most two features leave and two arrive. A capture removes the mover's old square
    and the victim; castling removes the king's and the rook's old squares and adds both
    new ones; a promotion adds a different piece from the one that left. Unused slots
    point at the all-zero row so the arithmetic stays uniform.
    """
    frm = np.int64(move & 63)
    to = np.int64((move >> 6) & 63)
    promo = np.int64((move >> 12) & 7)
    flag = np.int64((move >> 15) & 3)

    black = np.int64(state[STM])
    us = black
    them = 1 - black
    moved = np.int64(mail[frm])
    # FLAG_PROMO is 3, FLAG_CASTLE is 2 and FLAG_EP is 1, named in bitboard.py. Comparing
    # the decoded flag directly keeps this module free of a position.py import.
    arriving = promo if flag == 3 else moved

    simple = flag == 0 and mail[to] < 0
    for p in range(2):
        sub0 = feature(p, us, moved, frm)
        add0 = feature(p, us, arriving, to)
        if simple:
            _move_one(acc[ply + 1, p], acc[ply, p], FT_WEIGHT[sub0], FT_WEIGHT[add0])
            continue

        sub1 = np.int32(ZERO_FEATURE)
        add1 = np.int32(ZERO_FEATURE)
        if flag == 1:
            # En passant. The captured pawn sits beside the target square, not behind it.
            sub1 = feature(p, them, PAWN, to + 8 if black else to - 8)
        elif flag == 2:
            # Castling. The king's move is already in slot zero, so this is the rook.
            if to == 6:
                rook_from, rook_to = np.int64(7), np.int64(5)
            elif to == 2:
                rook_from, rook_to = np.int64(0), np.int64(3)
            elif to == 62:
                rook_from, rook_to = np.int64(63), np.int64(61)
            else:
                rook_from, rook_to = np.int64(56), np.int64(59)
            sub1 = feature(p, us, ROOK, rook_from)
            add1 = feature(p, us, ROOK, rook_to)
        else:
            captured = np.int64(mail[to])
            if captured >= 0:
                sub1 = feature(p, them, captured, to)
        _move_two(
            acc[ply + 1, p],
            acc[ply, p],
            FT_WEIGHT[sub0],
            FT_WEIGHT[sub1],
            FT_WEIGHT[add0],
            FT_WEIGHT[add1],
        )


@njit(int32(int16[:, :, ::1], int64, uint64[:]), cache=False)
def forward(acc: Bits, ply: Square, state: Bits) -> Bits:
    """Evaluate the accumulated position in centipawns, from the side to move's view."""
    stm = np.int64(state[STM])
    bucket = (popcount(state[WOCC] | state[BOCC]) - 2) // BUCKET_DIVISOR
    total = _dot(acc[ply, stm], acc[ply, 1 - stm], OUT_WEIGHT[bucket])
    # Three scalar operations, so int64 costs nothing here and removes any doubt about the
    # final scaling overflowing. SCReLU squares the input scale, hence the extra divide.
    scaled = (np.int64(total) // QA + np.int64(OUT_BIAS[bucket])) * SCALE // (QA * QB)
    if scaled > EVAL_LIMIT:
        return np.int32(EVAL_LIMIT)
    if scaled < -EVAL_LIMIT:
        return np.int32(-EVAL_LIMIT)
    return np.int32(scaled)


def new_accumulator(plies: int) -> np.ndarray:
    """A per-ply accumulator stack. C-contiguous, so `acc[ply, p]` vectorises."""
    return np.zeros((plies, 2, L1), dtype=np.int16)


# Warm every jitted function at import, with the argument types the real calls use, so
# compilation lands inside the platform's 90 second init budget rather than on the clock.
# A real position rather than an empty board: `apply` reads the mover off the mailbox, and
# warming it on an empty square would index the weights with a piece type of -1.
_acc = new_accumulator(4)
_state = np.zeros(16, dtype=np.uint64)
_mail = np.full(64, -1, dtype=np.int8)
_mail[8] = PAWN
_mail[16] = PAWN
_mail[63] = ROOK
_state[PAWN] = (np.uint64(1) << np.uint64(8)) | (np.uint64(1) << np.uint64(16))
_state[ROOK] = np.uint64(1) << np.uint64(63)
_state[WOCC] = _state[PAWN]
_state[BOCC] = _state[ROOK]
refresh(_acc, 0, _state, _mail)
copy(_acc, 0)
forward(_acc, 0, _state)
for _flag in (0, 1, 2, 3):
    # Both branches of `apply`: a quiet move onto an empty square, and every flagged form.
    apply(_acc, 0, _state, _mail, np.int32(8 | (24 << 6) | (4 << 12) | (_flag << 15)))
apply(_acc, 0, _state, _mail, np.int32(8 | (16 << 6)))
