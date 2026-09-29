"""The C++ engine's move generator and game tracking, spoken to over UCI.

Perft is the gate: one illegal move loses a game outright. Skipped until the engine is built
with `make -C cpp`.
"""

import chess
import pytest

from tests.engine import Engine, info_field
from tests.engine import pytestmark as pytestmark
from tests.openings import OPENINGS

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

KIWIPETE = CASES[1][1]


@pytest.mark.parametrize("name,fen,expected", CASES, ids=[c[0] for c in CASES])
def test_perft(engine: Engine, name: str, fen: str, expected: list[int]) -> None:
    engine.send(f"position fen {fen}")
    for depth, want in enumerate(expected, start=1):
        engine.send(f"perft {depth}")
        got = int(info_field(engine.until("nodes")[-1], "nodes"))
        assert got == want, f"{name} perft({depth}) = {got}, expected {want}"


def test_a_double_push_nobody_can_take_continues_the_game(engine: Engine) -> None:
    """The harness writes an en passant square only when the capture is legal.

    Comparing raw en passant squares, as agent.py does, reads the FEN after any other
    double push as a different position, and so as a new game.
    """
    board = chess.Board()
    engine.send("debug on")
    engine.send(f"position fen {board.fen()}")
    engine.send("go nodes 5000")
    reply = engine.until("bestmove")[-1].split()[1]
    board.push_uci(reply)
    # After one white move no white pawn stands on the fifth rank to take it.
    board.push_uci("a7a5")
    assert board.ep_square is not None and "a6" not in board.fen()
    engine.send(f"position fen {board.fen()}")
    engine.send("isready")
    assert "info string new game" not in engine.until("readyok")

    engine.send(f"position fen {OPENINGS[0]}")
    engine.send("isready")
    assert "info string new game" in engine.until("readyok")


def test_timed_move_is_legal_and_prompt(engine: Engine) -> None:
    board = chess.Board(KIWIPETE)
    engine.send(f"position fen {board.fen()}")
    engine.send("go wtime 2000 btime 2000 winc 0 binc 0")
    lines = engine.until("bestmove")
    move = chess.Move.from_uci(lines[-1].split()[1])
    assert move in board.legal_moves
    # A 2 s clock allows a fraction of a second; well under the hard limit either way.
    assert int(info_field([line for line in lines if " time " in line][-1], "time")) < 1000
