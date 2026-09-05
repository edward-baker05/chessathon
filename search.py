from time import perf_counter_ns

import chess

from evaluate import evaluate

MATE = 999999


def search(board: chess.Board, time_left_ms: int) -> str:
    legal_moves = board.legal_moves
    move_results = {move: 0 for move in legal_moves}

    start = end = perf_counter_ns()

    i = 0
    while (elapsed := (end - start) / 1_000_000) < (time_left_ms / 50):
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
