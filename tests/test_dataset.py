"""The training pipeline agrees with the engine.

Two things are checked, and they are the two places this pipeline can go wrong silently.

The extractor reads FENs with its own parser rather than python-chess, for speed. So that
parser is checked against python-chess.

The trainer derives features from packed records, and the engine derives them from the
board. If those two ever disagree, the net trains under one convention and plays under
another. It still reaches a plausible loss. It just plays badly, and nothing in the
training curve says why. So the features are compared directly against `nnue.refresh`.
"""

import chess
import numpy as np

import nnue
import position
from tests.conftest import random_positions
from tools import dataset
from tools.extract import parse_board

FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
]


def packed(board: chess.Board) -> np.ndarray:
    parsed = parse_board(board.board_fen().encode())
    assert parsed is not None, board.fen()
    occupancy, codes, _ = parsed
    record = dataset.pack(occupancy, codes, board.turn == chess.BLACK, 123)
    return np.frombuffer(record, dtype=np.uint8).reshape(1, dataset.RECORD)


def test_the_extractor_parses_boards_the_way_python_chess_does() -> None:
    for board in random_positions(count=200, seed=91):
        parsed = parse_board(board.board_fen().encode())
        assert parsed is not None, board.fen()
        occupancy, codes, _ = parsed

        squares = sorted(board.piece_map())
        assert occupancy == sum(1 << square for square in squares), board.fen()
        assert len(codes) == len(squares), board.fen()
        for code, square in zip(codes, squares, strict=True):
            piece = board.piece_at(square)
            assert piece is not None
            colour = 0 if piece.color == chess.WHITE else 1
            assert code == colour * 6 + (piece.piece_type - 1), board.fen()


def test_unpacked_features_build_the_accumulator_the_engine_builds() -> None:
    """The agreement the whole pipeline rests on.

    An accumulator assembled from the trainer's sparse features must equal, element for
    element, the one `nnue.refresh` builds from the board.
    """
    for fen in FENS:
        board = chess.Board(fen)
        index, white, black, stm, score = dataset.unpack(packed(board))
        assert score.tolist() == [123]
        assert stm.tolist() == [1 if board.turn == chess.BLACK else 0]

        built = np.stack([nnue.FT_BIAS.astype(np.int64)] * 2)
        for feature in white[index == 0]:
            built[0] += nnue.FT_WEIGHT[feature].astype(np.int64)
        for feature in black[index == 0]:
            built[1] += nnue.FT_WEIGHT[feature].astype(np.int64)

        state, mail = position.new_stacks()
        position.encode(board, state[0], mail[0])
        acc = nnue.new_accumulator(2)
        nnue.refresh(acc, 0, state[0], mail[0])

        assert np.array_equal(built, acc[0].astype(np.int64)), fen


def test_unpacking_a_batch_keeps_positions_separate() -> None:
    """The sparse layout is a flat coordinate list, so an off-by-one in the row index
    would quietly mix pieces between positions rather than fail."""
    boards = [chess.Board(fen) for fen in FENS]
    batch = np.concatenate([packed(board) for board in boards])
    index, white, _black, stm, _score = dataset.unpack(batch)

    for row, board in enumerate(boards):
        expected = sorted(
            (0 if piece.color == chess.WHITE else 1) * 6 * 64
            + (piece.piece_type - 1) * 64
            + square
            for square, piece in board.piece_map().items()
        )
        assert sorted(white[index == row].tolist()) == expected, board.fen()
    assert stm.tolist() == [1 if board.turn == chess.BLACK else 0 for board in boards]


def test_scores_survive_the_round_trip_including_negatives() -> None:
    for score in (-12800, -900, -1, 0, 1, 900, 12800):
        record = dataset.pack(0x1000_0000_0000_0010, [5, 11], False, score)
        batch = np.frombuffer(record, dtype=np.uint8).reshape(1, dataset.RECORD)
        _index, _white, _black, _stm, unpacked = dataset.unpack(batch)
        assert unpacked.tolist() == [score]
