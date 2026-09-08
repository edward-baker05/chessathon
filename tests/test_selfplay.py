"""The self-play generator writes the same records the extractor writes.

`tools/selfplay.py` builds its occupancy and piece codes from a `chess.Board` rather than
from a FEN field, so it is a second implementation of the thing `tools/extract.py` already
does. Two implementations of a packing convention is exactly the setup that produces a file
which trains to a plausible loss and plays like noise, and nothing in a training curve says
which of the two was wrong. So they are compared directly, byte for byte.
"""

import chess
import numpy as np
import pytest

from tools import dataset
from tools.extract import parse_board
from tools.selfplay import _encode, _label

FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "8/8/8/4k3/8/8/4P3/4K3 b - - 0 1",
]


@pytest.mark.parametrize("fen", FENS)
def test_encoding_a_board_matches_the_extractor(fen: str) -> None:
    """The two encoders agree on occupancy and on every piece code, in order."""
    board = chess.Board(fen)
    parsed = parse_board(board.board_fen().encode())
    assert parsed is not None
    expected_occupancy, expected_codes, _ = parsed

    encoded = _encode(board)
    assert encoded is not None
    occupancy, codes = encoded

    assert occupancy == expected_occupancy
    assert codes == expected_codes


@pytest.mark.parametrize("fen", FENS)
def test_a_packed_selfplay_record_unpacks_to_the_position_it_came_from(fen: str) -> None:
    """Round trip through the record the trainer will actually read."""
    board = chess.Board(fen)
    encoded = _encode(board)
    assert encoded is not None
    occupancy, codes = encoded
    record = dataset.pack(occupancy, codes, board.turn == chess.BLACK, -456)
    batch = np.frombuffer(record, dtype=np.uint8).reshape(1, dataset.RECORD)

    _index, _white, _black, stm, score, pieces = dataset.unpack(batch)
    assert score.tolist() == [-456]
    assert stm.tolist() == [1 if board.turn == chess.BLACK else 0]
    assert pieces.tolist() == [len(board.piece_map())]


def test_positions_in_check_are_dropped() -> None:
    """The evaluation is never asked about a position in check, so it is not trained on."""
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_check()
    assert _label(board.fen(), chess.Move.from_uci("g1f3"), depth=1) is None


def test_positions_whose_move_is_a_capture_or_promotion_are_dropped() -> None:
    """A tactic the quiescence search resolves is not the evaluation's to learn."""
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    assert _label(board.fen(), chess.Move.from_uci("e4d5"), depth=1) is None

    promotion = chess.Board("8/4P3/8/8/8/8/4k3/4K3 w - - 0 1")
    assert _label(promotion.fen(), chess.Move.from_uci("e7e8q"), depth=1) is None
