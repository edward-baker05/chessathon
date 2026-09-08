"""The submission entrypoint. The platform imports this file and calls get_move.

It also writes a log. The runner points file descriptor 1 at stderr before importing this
file, so `print` cannot reach the protocol stream, and the platform keeps what we write and
shows it on the dashboard after every rated game, next to the PGN. That is the only channel
a rated game has for saying what the engine was thinking, and without it a bad move on the
ladder is a move with no explanation attached.

The budget is 8 KB, kept as the first 4 KB and the last 4 KB, so the middle of a long game
is dropped. That is what `_LINE` is shaped around: one short line per move, about 60 bytes,
so roughly the first and last 60 moves of any game survive. Those are the two ends a lost
game is usually decided at.
"""

import contextlib
import hashlib
import io
import os
import sys

import chess

import nnue
import position
import search
import tt
from bitboard import KEY

# Import time runs once per game, inside a 90 second budget, before the clock starts.
# Importing search pulls in every jitted function and warms it, which is the point. How long
# that took is not measured here: the platform reports it in the validation log and in the
# per-game CSV, and timing it from inside would mean placing a statement above the imports.

# Line buffering is not enough on its own. The platform stops a container between moves and
# ends it abruptly, so anything sitting in a buffer at that moment is lost, and the moves
# worth reading are the last ones. Every log call flushes.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)

# On by default: the log is the point. `CHESSATHON_DEBUG=0` silences it for A/B runs, where
# thousands of games would write it and nobody would read it.
DEBUG = os.environ.get("CHESSATHON_DEBUG", "1") != "0"


def log(message: str) -> None:
    """One line to the dashboard log, flushed immediately. Never raises."""
    if not DEBUG:
        return
    # A closed or broken stdout must never cost a game, and there is nowhere left to report
    # it if it happens.
    with contextlib.suppress(OSError, ValueError):
        print(message, flush=True)

# The rated time control is 120 s + 0.5 s per move. Local fast games use a smaller
# increment, so the harness can say so rather than have the agent over-budget and flag.
INCREMENT_MS = int(os.environ.get("CHESSATHON_INCREMENT_MS", "500"))
NODE_LIMIT = int(os.environ.get("CHESSATHON_NODE_LIMIT", "0"))

_scratch_state, _scratch_mail = position.new_stacks()

# Positions already seen in this game, as Zobrist keys, for repetition detection.
_history: list[int] = []
# The position after our own last reply. The platform only ever shows us positions where
# it is our turn, so this is what lets us tell "the game continued" from "a new game".
_last_reply: chess.Board | None = None


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; our colour is the side to move
    time_left_ms  our clock before this move, in milliseconds
    """
    board = chess.Board(fen)
    _track(board)
    uci = search.think(board, time_left_ms, increment_ms=INCREMENT_MS, node_limit=NODE_LIMIT)
    move = _validated(board, uci)
    _log_move(board, move, time_left_ms)
    _remember(board, move)
    return move.uci()


def _log_move(board: chess.Board, move: chess.Move, time_left_ms: int) -> None:
    """One line per move: what was searched, what it cost, and what was left.

    Deliberately terse. Every field here is one that a lost game has been blamed on at some
    point and that the PGN alone cannot settle: how deep it actually got, whether the score
    was falling, whether the move overran its soft budget, and how much clock remained.
    """
    depth, seldepth, score, nodes, spent = search.last_search()
    soft, _hard = search.budget_of()
    log(
        f"{board.fullmove_number}{'w' if board.turn else 'b'} {move.uci()} "
        f"d{depth}/{seldepth} {score:+d}cp {nodes / 1000:.0f}kn "
        f"{spent:.2f}/{soft:.2f}s clk{time_left_ms / 1000:.1f}s"
    )


def _key_of(board: chess.Board) -> int:
    position.encode(board, _scratch_state[0], _scratch_mail[0])
    return int(_scratch_state[0][KEY])


def _signature(board: chess.Board) -> tuple[str, bool, int, int | None]:
    return (board.board_fen(), board.turn, board.castling_rights, board.ep_square)


def _continues_our_game(board: chess.Board) -> bool:
    """Is `board` one legal move on from the position we last moved into?

    The platform hands over a bare FEN with no game identity, so this is how the agent
    tells a continuing game from the first move of a new one.
    """
    if _last_reply is None:
        return False
    target = _signature(board)
    for move in _last_reply.legal_moves:
        _last_reply.push(move)
        matched = _signature(_last_reply) == target
        _last_reply.pop()
        if matched:
            return True
    return False


def _track(board: chess.Board) -> None:
    global _history
    if not _continues_our_game(board):
        # A new game. Nothing we learned in the last one applies to this one.
        _history = []
        tt.tt_clear(tt.TT)
        search.clear_tables()
        log(f"new game, we are {'white' if board.turn else 'black'}, from {board.fen()}")
    _history.append(_key_of(board))
    search.set_game_history(_history)


def _remember(board: chess.Board, move: chess.Move) -> None:
    global _last_reply
    board.push(move)
    _history.append(_key_of(board))
    _last_reply = board.copy(stack=False)
    board.pop()


def _validated(board: chess.Board, uci: str) -> chess.Move:
    """Never lose a game to a malformed move. This should never fire."""
    try:
        move = chess.Move.from_uci(uci)
    except (ValueError, chess.InvalidMoveError):
        move = chess.Move.null()
    if move in board.legal_moves:
        return move
    log(f"UNPLAYABLE MOVE {uci!r} in {board.fen()}; falling back to a legal one")
    return next(iter(board.legal_moves))


# The last thing import does. It names the network that is about to play, which is the only
# reliable answer to "which build is actually live": the hash is over the weights themselves,
# so two uploads that differ only in the network are still told apart, and a rated game's log
# says which one played it.
_NET_ID = hashlib.sha256(nnue.WEIGHTS_PATH.read_bytes()).hexdigest()[:12]
log(
    f"ready | net {_NET_ID} L1 {nnue.L1} x {nnue.BUCKETS} buckets, "
    f"{nnue.NUM_FEATURES} features, QA {nnue.QA} | increment {INCREMENT_MS}ms"
)
