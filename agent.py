"""The submission entrypoint. The platform imports this file and calls get_move."""

import io
import sys
from time import perf_counter_ns

import chess

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

MATE = 10**6


def evaluate(board: chess.Board) -> int:
    who_to_move = 1 if board.turn else -1

    if board.is_checkmate():
        return -MATE

    return (
        sum(
            value * (len(board.pieces(piece, True)) - len(board.pieces(piece, False)))
            for piece, value in PIECE_VALUE.items()
        )
        * who_to_move
    )


def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    if depth == 0:
        return evaluate(board)

    best_value = -MATE

    for move in board.legal_moves:
        board.push(move)
        current_value = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if current_value > best_value:
            best_value = current_value
            if current_value > alpha:
                alpha = current_value

        if current_value >= beta:
            return best_value

    return best_value


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.

    print() is safe. Your stdout is redirected away from the protocol stream, discarded
    during rated games and shown back to you in the validation log.
    """
    board = chess.Board(fen)

    legal_moves = board.legal_moves
    move_results = {move: 0 for move in legal_moves}

    start = end = perf_counter_ns()

    i = 0
    while (elapsed := (end - start) / 1_000_000_000) < 1:
        i += 1
        print(f"Calculating best move at depth {i}, current elapsed {elapsed:.2f}")
        for current_move in legal_moves:
            board.push(current_move)
            score = -negamax(board, i, -MATE, MATE)
            move_results[board.pop()] = score
        end = perf_counter_ns()

    move = max(move_results, key=lambda k: move_results[k])
    print(f"{move} at depth {i} in time {elapsed:.2f}s")
    return move.uci()
