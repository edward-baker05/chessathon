import time
from dataclasses import dataclass
from typing import Literal

import chess
import chess.pgn

from harness.rules import INIT_BUDGET_S
from harness.sandbox import Agent, AgentFailure

RESULT_HEADERS = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2", "void": "*"}
FAILED_TERMINATIONS = frozenset({"crash", "illegal", "flag", "init", "both_failed"})

Result = Literal["white", "black", "draw", "void"]
Decision = Literal["white", "black", "draw"]


@dataclass(frozen=True)
class Outcome:
    result: Result
    termination: str
    pgn: str


def play_match(
    white: Agent,
    black: Agent,
    base_ms: int,
    increment_ms: int,
    ply_cap: int | None = None,
    start_fen: str = chess.STARTING_FEN,
) -> Outcome:
    try:
        return _play(white, black, base_ms, increment_ms, ply_cap, start_fen)
    finally:
        white.stop()
        black.stop()


def _play(
    white: Agent, black: Agent, base_ms: int, increment_ms: int, ply_cap: int | None, start_fen: str
) -> Outcome:
    board = chess.Board(start_fen)
    agents = {chess.WHITE: white, chess.BLACK: black}

    white_failure = _start(white)
    black_failure = _start(black)
    if white_failure is not None and black_failure is not None:
        return _outcome(board, "void", "both_failed")
    if white_failure is not None:
        return _outcome(board, "black", white_failure)
    if black_failure is not None:
        return _outcome(board, "white", black_failure)

    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}

    while True:
        finish = board.outcome(claim_draw=True)
        if finish is not None:
            return _outcome(board, _decide(finish), finish.termination.name.lower())
        if ply_cap is not None and board.ply() >= ply_cap:
            return _outcome(board, "draw", "ply_cap")

        mover = board.turn
        started_at = time.monotonic()
        try:
            uci = agents[mover].move(board.fen(), int(clock[mover]))
        except AgentFailure as failure:
            return _outcome(board, _side(not mover), failure.reason)
        clock[mover] -= (time.monotonic() - started_at) * 1000.0
        if clock[mover] < 0:
            return _outcome(board, _flagged(board, mover), "flag")

        move = _legal_move(board, uci)
        if move is None:
            return _outcome(board, _side(not mover), "illegal")
        board.push(move)
        clock[mover] += increment_ms


def _start(agent: Agent) -> str | None:
    try:
        agent.start(INIT_BUDGET_S)
    except AgentFailure as failure:
        return failure.reason
    return None


def _legal_move(board: chess.Board, uci: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return None
    return move if move in board.legal_moves else None


def _side(colour: chess.Color) -> Decision:
    return "white" if colour == chess.WHITE else "black"


def _decide(finish: chess.Outcome) -> Decision:
    return "draw" if finish.winner is None else _side(finish.winner)


def _flagged(board: chess.Board, mover: chess.Color) -> Decision:
    return "draw" if board.has_insufficient_material(not mover) else _side(not mover)


def _outcome(board: chess.Board, result: Result, termination: str) -> Outcome:
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = RESULT_HEADERS[result]
    game.headers["Termination"] = termination
    return Outcome(result=result, termination=termination, pgn=str(game))
