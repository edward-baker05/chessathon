"""Shared fixtures. Kept dependency-free so every test module can import it."""

import random
from collections.abc import Iterator

import chess


def random_positions(count: int, seed: int, max_plies: int = 40) -> Iterator[chess.Board]:
    """Yield varied legal positions by walking random legal moves from the start.

    Deterministic for a given seed so a failing test can always be reproduced.
    Games that end early restart from the initial position.
    """
    rng = random.Random(seed)
    produced = 0
    while produced < count:
        board = chess.Board()
        for _ in range(rng.randint(1, max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        yield board.copy()
        produced += 1
