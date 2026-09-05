import chess
from numba import int64, njit, uint64

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

MATE = 999999


@njit(uint64(uint64), cache=True)
def popcount(x: int) -> int:
    count = 0
    while x:
        x &= x - uint64(1)  # type: ignore[assignment]  # numba needs a same-width literal here
        count += 1
    return count


@njit(int64(uint64, uint64, uint64, uint64, uint64, uint64, uint64), cache=True)
def material_score(
    pawns: int,
    knights: int,
    bishops: int,
    rooks: int,
    queens: int,
    white: int,
    black: int,
) -> int:
    score = 0
    score += 100 * (popcount(pawns & white) - popcount(pawns & black))
    score += 320 * (popcount(knights & white) - popcount(knights & black))
    score += 330 * (popcount(bishops & white) - popcount(bishops & black))
    score += 500 * (popcount(rooks & white) - popcount(rooks & black))
    score += 900 * (popcount(queens & white) - popcount(queens & black))
    return score


def evaluate(board: chess.Board) -> int:
    who_to_move = 1 if board.turn else -1

    if board.is_checkmate():
        return -MATE

    return (
        material_score(
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.occupied_co[chess.WHITE],
            board.occupied_co[chess.BLACK],
        )
        * who_to_move
    )


def evaluate_pure_python(board: chess.Board) -> int:
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
