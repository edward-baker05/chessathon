import chess
import pytest

from lichess.zygote import clock_from, position_from


def test_startpos_with_moves() -> None:
    board = position_from(["startpos", "moves", "e2e4", "e7e5"])
    assert board.fen() == chess.Board(
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    ).fen()


def test_fen_with_and_without_moves() -> None:
    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    assert position_from(["fen", *fen.split()]).fen() == fen
    after = position_from(["fen", *fen.split(), "moves", "e2e4"])
    assert after.fen() == "4k3/8/8/8/4P3/8/8/4K3 b - - 0 1"


def test_unknown_position_is_an_error() -> None:
    with pytest.raises(ValueError):
        position_from(["nonsense"])


def test_clock_is_the_side_to_move() -> None:
    tokens = ["wtime", "60000", "btime", "45000", "winc", "1000", "binc", "2000"]
    assert clock_from(chess.Board(), tokens) == (60000, 1000)
    black = position_from(["startpos", "moves", "e2e4"])
    assert clock_from(black, tokens) == (45000, 2000)


def test_missing_increment_is_zero() -> None:
    assert clock_from(chess.Board(), ["wtime", "60000", "btime", "60000"]) == (60000, 0)


def test_movetime_is_the_whole_clock() -> None:
    assert clock_from(chess.Board(), ["movetime", "10000"]) == (10000, 0)
