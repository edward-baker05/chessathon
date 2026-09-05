"""Evaluation. Centipawns, positive means the side to move is better.

The search calls this at leaf nodes from jitted code, so it has to stay njit-compatible.
That is the whole contract:

    evaluate(state, mailbox) -> int32

`state` is one row of the position state vector and `mailbox` its 64-entry piece array, as
laid out in position.py. Field indices come from bitboard.py.

Two things to know before changing this.

torch and onnxruntime must not appear here. Their per-call overhead is 20 to 50 us, which
is larger than the entire budget for a node, so calling one per leaf would make the engine
slower than it was before it had a real search at all. When this becomes an NNUE, train in
torch offline, export quantised weights to a .npz, and do inference here in numba over an
incrementally updated int16 accumulator. Fields 13 to 15 of the state vector are spare and
reserved for exactly that.

Checkmate and stalemate are not detected here. The search knows when a node has no legal
moves and scores it, so an evaluation that looked for mate would be both slower and wrong
at a node the search has already handled.
"""

from typing import Any

import numpy as np
from numba import int8, int32, njit, uint64

from bitboard import BOCC, STM, WOCC, popcount

Bits = Any

# Pawn, knight, bishop, rook, queen. The king is not counted: both sides always have one.
PIECE_VALUE = np.array([100, 320, 330, 500, 900], dtype=np.int32)


@njit(int32(uint64[:], int8[:]), cache=False)
def evaluate(state: Bits, mailbox: Bits) -> Bits:
    """Material only. This is the piece a stronger evaluation replaces."""
    white = state[WOCC]
    black = state[BOCC]
    score = 0
    for piece in range(5):
        score += PIECE_VALUE[piece] * (
            popcount(state[piece] & white) - popcount(state[piece] & black)
        )
    return np.int32(-score if state[STM] else score)


# Warm the jitted function with the argument types the real calls use, so compilation
# lands in the import budget rather than on the clock.
_state = np.zeros(16, dtype=np.uint64)
_mailbox = np.full(64, -1, dtype=np.int8)
evaluate(_state, _mailbox)
