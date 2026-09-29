"""The C++ engine's UCI layer: every command and parameter the specification defines.

The specification asks an engine to accept anything a GUI may send, to skip what it does
not understand and parse the rest of the line, and to answer `isready` even mid-search.
These tests send the awkward forms as well as the usual ones.
"""

import time

import chess
import pytest

from tests.engine import Engine, info_field
from tests.engine import pytestmark as pytestmark

MATED = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"


def ready(engine: Engine) -> list[str]:
    """Everything the engine said before answering an `isready`."""
    engine.send("isready")
    return engine.until("readyok")[:-1]


def fen_of(engine: Engine) -> str:
    engine.send("d")
    while " key " not in (line := engine.until("")[-1]):
        pass
    return line.split(" key ")[0]


def bestmove_of(lines: list[str]) -> tuple[str, str | None]:
    tokens = lines[-1].split()
    return tokens[1], tokens[3] if len(tokens) >= 4 and tokens[2] == "ponder" else None


def bestmove(engine: Engine) -> tuple[str, str | None]:
    return bestmove_of(engine.until("bestmove"))


def test_uci_handshake_declares_options(engine: Engine) -> None:
    engine.send("uci")
    lines = engine.until("uciok")
    assert lines[0].startswith("id name ")
    assert any(line.startswith("id author ") for line in lines)
    options = [line for line in lines if line.startswith("option name ")]
    for name in ("Hash", "Clear Hash", "Ponder", "Threads"):
        assert any(line.startswith(f"option name {name} type ") for line in options), name


def test_unknown_tokens_before_a_command_are_skipped(engine: Engine) -> None:
    engine.send("joho debug on")
    engine.send(f"position fen {MATE_IN_ONE}")
    assert "info string new game" in ready(engine)


def test_unknown_commands_and_blank_lines_are_ignored(engine: Engine) -> None:
    for line in ("", "   ", "xyzzy", "xyzzy plugh", "\t\r"):
        engine.send(line)
    assert ready(engine) == []


def test_crlf_and_tabs_are_whitespace(engine: Engine) -> None:
    engine.send("position\tfen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1\r")
    assert fen_of(engine) == "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"


def test_setoption_in_every_form(engine: Engine) -> None:
    for line in (
        "setoption name Hash value 16",
        "setoption name hash value 32",
        "setoption name Hash value 0",
        "setoption name Hash value lots",
        "setoption name Clear Hash",
        "setoption name Ponder value true",
        "setoption name Threads value 1",
        "setoption name Threads value 0",
        "setoption name Threads value many",
        "setoption name UCI_Opponent value none 2800 computer Some Engine 1.0",
        "setoption name No Such Option value with several words",
        "setoption name",
        "setoption",
    ):
        engine.send(line)
    engine.send("position startpos")
    engine.send("go depth 3")
    assert chess.Move.from_uci(bestmove(engine)[0]) in chess.Board().legal_moves


def test_register_and_debug_are_accepted(engine: Engine) -> None:
    for line in (
        "register later",
        "register name Someone code 1234",
        "debug on",
        "debug off",
        "debug",
    ):
        engine.send(line)
    assert ready(engine) == []


@pytest.mark.parametrize(
    "command,expected",
    [
        ("position startpos", chess.STARTING_FEN),
        (
            "position startpos moves e2e4 e7e5",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        ),
        ("position startpos moves", chess.STARTING_FEN),
        ("position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),
        ("position fen 4k3/8/8/8/8/8/4P3/4K3 w - -", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),
        ("position fen 4k3/8/8/8/8/8/4P3/4K3 w moves e2e4", "4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1"),
        (
            "position fen k7/4P3/8/8/8/8/8/4K3 w - - 0 1 moves e7e8Q",
            "k3Q3/8/8/8/8/8/8/4K3 b - - 0 1",
        ),
        # An illegal move ends the list: the moves before it stand, the rest are ignored.
        (
            "position startpos moves e2e4 e7e9 d2d4",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ),
    ],
)
def test_position_forms(engine: Engine, command: str, expected: str) -> None:
    engine.send(command)
    assert fen_of(engine) == expected


def test_an_invalid_position_keeps_the_last_one(engine: Engine) -> None:
    engine.send("position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    for line in ("position fen not/a/fen w - - 0 1", "position", "position sideways"):
        engine.send(line)
    assert fen_of(engine) == "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"


@pytest.mark.parametrize(
    "go",
    [
        "go depth 4",
        "go nodes 5000",
        "go movetime 50",
        "go wtime 3000 btime 3000",
        "go wtime 3000 btime 3000 winc 100 binc 100 movestogo 20",
        "go btime 3000 wtime 3000 binc 0",
        "go wtime -40 btime 3000",
        "go mate 3 nodes 5000",
        "go depth notanumber nodes 5000",
        "go frobnicate 7 depth 3",
        "go depth 0",
        "go depth 1000 nodes 5000",
    ],
)
def test_go_parameters(engine: Engine, go: str) -> None:
    engine.send("position startpos moves e2e4")
    started = time.monotonic()
    engine.send(go)
    move, _ = bestmove(engine)
    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(move) in board.legal_moves
    assert time.monotonic() - started < 5


def test_searchmoves_restricts_the_root(engine: Engine) -> None:
    engine.send("position startpos")
    engine.send("go searchmoves a2a3 h2h3 depth 6")
    assert bestmove(engine)[0] in {"a2a3", "h2h3"}
    engine.send("go depth 4 searchmoves g1f3")
    assert bestmove(engine)[0] == "g1f3"


def test_searchmoves_with_nothing_legal_searches_everything(engine: Engine) -> None:
    engine.send("position startpos")
    engine.send("go searchmoves 0000 e2e5 depth 3")
    assert chess.Move.from_uci(bestmove(engine)[0]) in chess.Board().legal_moves


def test_movetime_is_honoured(engine: Engine) -> None:
    engine.send("position startpos")
    started = time.monotonic()
    engine.send("go movetime 300")
    bestmove(engine)
    assert 0.2 < time.monotonic() - started < 0.6


def test_infinite_waits_for_stop(engine: Engine) -> None:
    engine.send(f"position fen {MATE_IN_ONE}")
    engine.send("go infinite")
    time.sleep(0.3)
    # A mate in one ends the search at once, and the answer must still wait for `stop`.
    assert not any(line.startswith("bestmove") for line in ready(engine))
    engine.send("stop")
    assert bestmove(engine)[0] == "a1a8"


def test_a_bare_go_can_be_stopped(engine: Engine) -> None:
    engine.send("position startpos")
    engine.send("go")
    time.sleep(0.2)
    engine.send("stop")
    assert chess.Move.from_uci(bestmove(engine)[0]) in chess.Board().legal_moves


def test_a_new_position_interrupts_an_infinite_search(engine: Engine) -> None:
    engine.send("position startpos")
    engine.send("go infinite")
    time.sleep(0.1)
    engine.send(f"position fen {MATE_IN_ONE}")
    engine.until("bestmove")
    engine.send("go depth 3")
    assert bestmove(engine)[0] == "a1a8"


@pytest.mark.parametrize("fen", [MATED, STALEMATE])
def test_no_legal_move_is_none(engine: Engine, fen: str) -> None:
    engine.send(f"position fen {fen}")
    engine.send("go depth 3")
    assert bestmove(engine) == ("(none)", None)


def test_bestmove_names_a_ponder_move(engine: Engine) -> None:
    engine.send("position startpos")
    engine.send("go depth 8")
    move, ponder = bestmove(engine)
    assert ponder is not None
    board = chess.Board()
    board.push_uci(move)
    assert chess.Move.from_uci(ponder) in board.legal_moves


def test_ponderhit_starts_the_clock(engine: Engine) -> None:
    engine.send("position startpos moves e2e4 e7e5")
    engine.send("go ponder wtime 4000 btime 4000")
    time.sleep(0.5)
    # Half a second is more than this clock allows a move, so only pondering explains
    # the search still running.
    assert not any(line.startswith("bestmove") for line in ready(engine))
    hit = time.monotonic()
    engine.send("ponderhit")
    move, _ = bestmove(engine)
    assert time.monotonic() - hit < 1.0
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert chess.Move.from_uci(move) in board.legal_moves


def test_a_stopped_ponder_is_not_our_move(engine: Engine) -> None:
    """A GUI stops pondering when the opponent plays something else, then sends the truth.

    The ponder search's answer was never played, so the real position still follows our
    last real reply and is the same game.
    """
    engine.send("debug on")
    engine.send("position startpos")
    engine.send("go depth 6")
    ours, expected = bestmove(engine)
    assert expected is not None
    board = chess.Board()
    board.push_uci(ours)
    predicted = board.copy()
    predicted.push_uci(expected)
    engine.send(f"position fen {predicted.fen()}")
    engine.send("go ponder wtime 60000 btime 60000")
    time.sleep(0.1)
    engine.send("stop")
    bestmove(engine)

    actual = next(move for move in board.legal_moves if move.uci() != expected)
    board.push(actual)
    engine.send(f"position fen {board.fen()}")
    assert "info string new game" not in ready(engine)


def test_isready_is_answered_mid_search(engine: Engine) -> None:
    engine.send("position startpos")
    engine.send("go infinite")
    started = time.monotonic()
    ready(engine)
    assert time.monotonic() - started < 0.5
    engine.send("stop")
    engine.until("bestmove")


def test_quit_mid_search_exits(engine: Engine) -> None:
    engine.send("position startpos")
    engine.send("go infinite")
    time.sleep(0.1)
    engine.send("quit")
    assert engine.process.wait(timeout=2) == 0
    engine.process = Engine().process  # the fixture's close() needs a live process


def test_ucinewgame_between_games(engine: Engine) -> None:
    engine.send("ucinewgame")
    engine.send("isready")
    engine.until("readyok")
    engine.send("position startpos moves d2d4")
    engine.send("go depth 4")
    bestmove(engine)
    engine.send("ucinewgame")
    engine.send("position startpos")
    engine.send("go depth 4")
    assert chess.Move.from_uci(bestmove(engine)[0]) in chess.Board().legal_moves


def test_info_lines_are_well_formed(engine: Engine) -> None:
    engine.send("position startpos")
    engine.send("go depth 6")
    lines = engine.until("bestmove")
    infos = [line for line in lines if line.startswith("info depth")]
    assert infos
    for line in infos:
        for field in ("depth", "seldepth", "nodes", "nps", "time"):
            assert int(info_field(line, field)) >= 0
        assert info_field(line, "score") in {"cp", "mate"}
        board = chess.Board()
        for move in line.split(" pv ")[1].split():
            board.push_uci(move)


@pytest.mark.parametrize(
    "go",
    ["go depth 8", "go nodes 200000", "go movetime 200", "go wtime 3000 btime 3000 winc 50"],
)
def test_threads_search(engine: Engine, go: str) -> None:
    engine.send("setoption name Threads value 4")
    board = chess.Board()
    for _ in range(4):
        engine.send(f"position fen {board.fen()}")
        engine.send(go)
        lines = engine.until("bestmove")
        move, ponder = bestmove_of(lines)
        board.push_uci(move)
        if ponder is not None:
            assert chess.Move.from_uci(ponder) in board.legal_moves
        final = [line for line in lines if line.startswith("info depth")][-1]
        if go.startswith("go nodes"):
            # Every thread's nodes count, and the limit is checked every 2048 nodes.
            assert 200_000 <= int(info_field(final, "nodes")) < 220_000
        if go == "go depth 8":
            assert int(info_field(final, "depth")) >= 8


def test_threads_stop_ponder_and_mate(engine: Engine) -> None:
    engine.send("setoption name Threads value 3")
    engine.send(f"position fen {MATE_IN_ONE}")
    engine.send("go infinite")
    time.sleep(0.2)
    assert not any(line.startswith("bestmove") for line in ready(engine))
    engine.send("stop")
    assert bestmove(engine)[0] == "a1a8"

    engine.send("position startpos moves e2e4 e7e5")
    engine.send("go ponder wtime 4000 btime 4000")
    time.sleep(0.5)
    assert not any(line.startswith("bestmove") for line in ready(engine))
    hit = time.monotonic()
    engine.send("ponderhit")
    bestmove(engine)
    assert time.monotonic() - hit < 1.0

    engine.send(f"position fen {MATED}")
    engine.send("go depth 5")
    assert bestmove(engine) == ("(none)", None)


def test_back_to_one_thread_is_deterministic_again(engine: Engine) -> None:
    def fixed_search(process: Engine) -> list[str]:
        """The last report and the move, less the time and node rate, which vary."""
        process.send("ucinewgame")
        process.send("position startpos moves d2d4 g8f6")
        process.send("go nodes 30000")
        lines = process.until("bestmove")
        final = [line for line in lines if line.startswith("info depth")][-1].split()
        timing = {i + 1 for i, token in enumerate(final) if token in {"nps", "time"}}
        return [t for i, t in enumerate(final) if i not in timing] + [lines[-1]]

    engine.send("setoption name Threads value 4")
    engine.send("position startpos")
    engine.send("go depth 8")
    bestmove(engine)
    engine.send("setoption name Threads value 1")
    fresh = Engine()
    try:
        expected = fixed_search(fresh)
    finally:
        fresh.close()
    assert fixed_search(engine) == expected
