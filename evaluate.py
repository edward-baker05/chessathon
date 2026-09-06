"""Evaluation. Centipawns, positive means the side to move is better.

The search calls this at leaf nodes from jitted code, so it has to stay njit-compatible.
The contract is now:

    evaluate(acc, ply, state) -> int32

`acc` is the per-ply accumulator stack from `Work`, `ply` the current ply, and `state` one
row of the position state vector as laid out in position.py. The evaluation reads the
accumulator rather than the board, because the accumulator was already brought up to date
incrementally when the move into this node was made. See nnue.py.

torch and onnxruntime must not appear here. Their per-call overhead is 20 to 50 us, which
is larger than the entire budget for a node, so calling one per leaf would make the engine
slower than it was before it had a real search at all. The network is trained offline in
torch, exported to a quantised .npz, and evaluated here in numba over an incrementally
updated int16 accumulator.

Checkmate and stalemate are not detected here. The search knows when a node has no legal
moves and scores it, so an evaluation that looked for mate would be both slower and wrong
at a node the search has already handled.
"""

from typing import Any

import numpy as np
from numba import int8, int32, njit, uint64

from bitboard import BOCC, STM, WOCC, popcount
from nnue import forward as evaluate

Bits = Any

__all__ = ["PIECE_VALUE", "evaluate", "material_eval"]

# Pawn, knight, bishop, rook, queen. The king is not counted: both sides always have one.
PIECE_VALUE = np.array([100, 320, 330, 500, 900], dtype=np.int32)


@njit(int32(uint64[:], int8[:]), cache=False)
def material_eval(state: Bits, mailbox: Bits) -> Bits:
    """Material only. What the engine used before it had a network.

    Retained because it is the reference the network is measured against: `snapshots/`
    holds a frozen engine built on this, and the A/B match against it is what decides
    whether a net is an improvement. It is deliberately not a runtime fallback. If the
    weights fail to load, nnue.py raises rather than quietly playing hundreds of Elo weak.
    """
    white = state[WOCC]
    black = state[BOCC]
    score = 0
    for piece in range(5):
        score += PIECE_VALUE[piece] * (
            popcount(state[piece] & white) - popcount(state[piece] & black)
        )
    return np.int32(-score if state[STM] else score)


# Warm the jitted function with the argument types the real calls use, so compilation
# lands in the import budget rather than on the clock. `evaluate` is warmed in nnue.py.
_state = np.zeros(16, dtype=np.uint64)
_mailbox = np.full(64, -1, dtype=np.int8)
material_eval(_state, _mailbox)
