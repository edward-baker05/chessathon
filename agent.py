"""The submission entrypoint. The platform imports this file and calls get_move."""

import io
import os
import sys

import chess

import position
import search
import tt
from bitboard import KEY

# Import time runs once per game, inside a 90 second budget, before the clock starts.
# Importing search pulls in every jitted function and warms it, which is the point.

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)

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
    _remember(board, move)
    return move.uci()


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
    print(f"engine produced an unplayable move {uci!r}; falling back to a legal one")
    return next(iter(board.legal_moves))
