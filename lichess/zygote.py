"""Serve the agent to lichess-bot over UCI, one forked process per game.

Importing the agent compiles every jitted function, which takes the best part of a minute.
lichess-bot starts a fresh engine for every game, and a game on lichess aborts if the first
move is that slow. So this process imports the agent once, listens on a Unix socket, and
forks a child for each connection. The child inherits the warmed agent and has never played
a move, which is what the harness gives us: the import done before the clock starts, and
module state that lives for one game and no longer.

lichess-bot launches `uci_client.py`, which only relays its stdin and stdout to this socket.
"""

import argparse
import os
import signal
import socket
import sys
from pathlib import Path
from typing import TextIO

import chess

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOCKET = Path(__file__).resolve().parent / "zygote.sock"


def position_from(tokens: list[str]) -> chess.Board:
    """The board a UCI `position` command describes, tokens after the word `position`."""
    if tokens and tokens[0] == "startpos":
        board = chess.Board()
        rest = tokens[1:]
    elif tokens and tokens[0] == "fen":
        end = tokens.index("moves") if "moves" in tokens else len(tokens)
        board = chess.Board(" ".join(tokens[1:end]))
        rest = tokens[end:]
    else:
        raise ValueError(f"unrecognised position command: {' '.join(tokens)!r}")
    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            board.push_uci(uci)
    return board


def clock_from(board: chess.Board, tokens: list[str]) -> tuple[int, int]:
    """Our clock and increment in ms from a UCI `go` command, tokens after the word `go`.

    lichess-bot sends `movetime` for the first move of a game and the clocks after that.
    A fixed move time is handed over as the whole clock with no increment: the agent never
    spends all of what it has, so it stays inside the limit.
    """
    values = dict(zip(tokens[::2], tokens[1::2], strict=False))
    if "movetime" in values:
        return int(values["movetime"]), 0
    ours = "w" if board.turn == chess.WHITE else "b"
    return int(values[f"{ours}time"]), int(values.get(f"{ours}inc", "0"))


def serve_game(stream: TextIO) -> None:
    """Speak UCI on `stream` for a single game, then return."""
    import agent
    import search

    def send(line: str) -> None:
        stream.write(line + "\n")
        stream.flush()

    board = chess.Board()
    for line in stream:
        command, *tokens = line.split() or [""]
        if command == "uci":
            send("id name chess-engine")
            send("id author edward-baker05")
            send("uciok")
        elif command == "isready":
            send("readyok")
        elif command == "position":
            board = position_from(tokens)
        elif command == "go":
            time_left_ms, increment_ms = clock_from(board, tokens)
            # The agent reads this per move; the harness default is 500.
            agent.INCREMENT_MS = increment_ms
            move = agent.get_move(board.fen(), time_left_ms)
            # lichess-bot keeps the last info before bestmove: it logs it after every move,
            # writes it into the saved PGN, and answers `!eval` in the game chat with it.
            if (info := search.uci_info(board)) is not None:
                send(info)
            send(f"bestmove {move}")
        elif command == "quit":
            return
        # ucinewgame, setoption and stop need nothing: each game is a new process, there
        # are no options, and the search is synchronous so a stop always arrives too late.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    arguments = parser.parse_args()

    # One core, and no thread pools that a fork could leave locked.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    # The harness points fd 1 at stderr before the import, so agent output never reaches
    # the protocol. Here the protocol is the socket, so the same move keeps prints in the log.
    os.dup2(2, 1)
    sys.path.insert(0, str(ROOT))
    import agent  # noqa: F401  (the import is the warm-up)

    arguments.socket.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(arguments.socket))
    listener.listen()
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # the kernel reaps finished games
    print(f"zygote ready on {arguments.socket}", file=sys.stderr, flush=True)

    while True:
        connection, _ = listener.accept()
        if os.fork() == 0:
            listener.close()
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            try:
                with connection, connection.makefile("rw", encoding="ascii") as stream:
                    serve_game(stream)
            except (BrokenPipeError, ConnectionResetError):
                pass  # lichess-bot closed the engine mid-move; the game is over
            finally:
                os._exit(0)
        connection.close()


if __name__ == "__main__":
    main()
