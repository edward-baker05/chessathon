"""The C++ engine in cpp/ against the Python engine it was ported from.

The port makes the same decisions in the same order and hashes with the same Zobrist keys,
so a fixed-node search visits the same tree: the same move, depth, score and node count.
That is a far stronger check than playing games against it, and it is what these tests hold
it to. A change to either engine's search that is not made in both fails here.

Skipped until the engine is built with `make -C cpp`.
"""

import subprocess
from collections.abc import Iterator
from pathlib import Path

import chess
import pytest

import search
import tt
from bitboard import KEY
from tests.conftest import random_positions
from tests.openings import OPENINGS
from tests.test_nnue import AWKWARD, encoded, evaluate_board
from tests.test_perft import CASES

ENGINE = Path(__file__).resolve().parent.parent / "cpp" / "build" / "engine"

pytestmark = pytest.mark.skipif(
    not ENGINE.exists(), reason="the C++ engine is not built; run `make -C cpp`"
)

SEARCH_NODES = 30_000
# Long enough that the node limit, not the clock, ends every Python search.
UNTIMED_MS = 24 * 60 * 60 * 1000


class Engine:
    """One engine process, spoken to over UCI a command at a time."""

    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [str(ENGINE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def send(self, line: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def until(self, prefix: str) -> list[str]:
        """Every line up to and including the first that starts with `prefix`."""
        assert self.process.stdout is not None
        lines = []
        for line in self.process.stdout:
            lines.append(line.strip())
            if line.startswith(prefix):
                return lines
        raise AssertionError(f"the engine exited before printing {prefix!r}: {lines}")

    def close(self) -> None:
        self.send("quit")
        self.process.wait(timeout=10)


@pytest.fixture
def engine() -> Iterator[Engine]:
    process = Engine()
    yield process
    process.close()


def info_field(line: str, name: str) -> str:
    tokens = line.split()
    return tokens[tokens.index(name) + 1]


@pytest.mark.parametrize("name,fen,expected", CASES, ids=[c[0] for c in CASES])
def test_perft(engine: Engine, name: str, fen: str, expected: list[int]) -> None:
    engine.send(f"position fen {fen}")
    for depth, want in enumerate(expected, start=1):
        engine.send(f"perft {depth}")
        got = int(info_field(engine.until("nodes")[-1], "nodes"))
        assert got == want, f"{name} perft({depth}) = {got}, expected {want}"


def test_keys_and_evaluation_match_python(engine: Engine) -> None:
    boards = [chess.Board(fen) for fen in AWKWARD] + list(random_positions(200, seed=7))
    for board in boards:
        engine.send(f"position fen {board.fen()}")
        engine.send("eval")
        _, value, _, key = engine.until("eval")[-1].split()
        state, _ = encoded(board)
        assert int(key, 16) == int(state[0][KEY]), board.fen()
        assert int(value) == evaluate_board(board), board.fen()


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, *AWKWARD[:4], *OPENINGS[:5]])
def test_fixed_node_search_matches_python(engine: Engine, fen: str) -> None:
    board = chess.Board(fen)
    state, _ = encoded(board)
    tt.tt_clear(tt.TT)
    search.clear_tables()
    search.set_game_history([int(state[0][KEY])])
    expected_move = search.think(board, UNTIMED_MS, increment_ms=0, node_limit=SEARCH_NODES)

    engine.send("ucinewgame")
    engine.send(f"position fen {fen}")
    engine.send(f"go nodes {SEARCH_NODES}")
    lines = engine.until("bestmove")
    final = [line for line in lines if line.startswith("info depth")][-1]

    assert lines[-1].split()[1] == expected_move
    assert int(info_field(final, "nodes")) == search.nodes()
    assert int(info_field(final, "depth")) == int(search.WORK.ints[search.I_DEPTH])
    assert final.split(" score ")[1].split(" nodes")[0] == search.uci_score(
        int(search.WORK.ints[search.I_SCORE])
    )


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
    board = chess.Board(AWKWARD[0])
    engine.send(f"position fen {board.fen()}")
    engine.send("go wtime 2000 btime 2000 winc 0 binc 0")
    lines = engine.until("bestmove")
    move = chess.Move.from_uci(lines[-1].split()[1])
    assert move in board.legal_moves
    # A 2 s clock allows a fraction of a second; well under the hard limit either way.
    assert int(info_field([line for line in lines if " time " in line][-1], "time")) < 1000
