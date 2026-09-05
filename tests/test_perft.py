"""Perft is the gate. One illegal move loses a game outright."""

import chess
import numpy as np
import pytest

import movegen
import position

CASES = [
    ("startpos", chess.STARTING_FEN, [20, 400, 8902, 197281, 4865609]),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [48, 2039, 97862, 4085603],
    ),
    ("ep-pin", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238, 674624]),
    (
        "promotion",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        [6, 264, 9467, 422333],
    ),
    (
        "position5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        [44, 1486, 62379, 2103487],
    ),
    (
        "position6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        [46, 2079, 89890, 3894594],
    ),
]


@pytest.mark.parametrize("name,fen,expected", CASES, ids=[c[0] for c in CASES])
def test_perft(name: str, fen: str, expected: list[int]) -> None:
    state, mailbox = position.new_stacks()
    moves = np.zeros(64 * movegen.MAX_MOVES, dtype=np.int32)
    for depth, want in enumerate(expected, start=1):
        position.encode(chess.Board(fen), state[0], mailbox[0])
        got = movegen.perft(state, mailbox, moves, 0, depth)
        assert got == want, f"{name} perft({depth}) = {got}, expected {want}"
