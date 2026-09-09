"""Independent current-checkout audit; xfails document unfixed engine defects.

Run with .venv/bin/python -m pytest tests/test_fresh_audit.py -q -rx.
No historical games, experiments, checkpoints, or engine source are used as oracles.
"""

import random
from typing import Any

import chess
import numpy as np
import pytest

import agent
import bitboard as bb
import movegen
import nnue
import position
import search
import tt


def prepare(fen: str) -> tuple[chess.Board, Any]:
    board = chess.Board(fen)
    assert board.is_valid(), fen
    work = search.WORK
    tt.tt_clear(work.table)
    search.clear_tables(work)
    search.set_game_history([], work)
    search.set_pruning(True, work)
    search._prepare(board, 3_600_000, 0, 0, work)
    return board, work


def encoded_move(board: chess.Board, uci: str) -> np.int32:
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    flag = 3 if move.promotion else 2 if board.is_castling(move) else (
        1 if board.is_en_passant(move) else 0
    )
    return np.int32(movegen.pack(move.from_square, move.to_square,
                                move.promotion - 1 if move.promotion else 0, flag))


@pytest.mark.parametrize(('fen', 'depth', 'expected'), [
    (chess.STARTING_FEN, 5, 4865609),
    ('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
     4, 4085603),
    ('8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1', 4, 43238),
])
def test_perft(fen: str, depth: int, expected: int) -> None:
    _, work = prepare(fen)
    assert movegen.perft(work.state, work.mail, work.moves, 0, depth) == expected


def test_move_state_hash_accumulator_and_features() -> None:
    from tools import dataset

    rng = random.Random(20260909)
    starts = [chess.STARTING_FEN,
              'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
              '4k3/P6p/8/3pP3/8/8/p6P/4K3 w - d6 0 1',
              '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1']
    positions = edges = 0
    flags: set[int] = set()
    for start in starts:
        board = chess.Board(start)
        for _ in range(750):
            if board.is_game_over() or board.ply() > 350:
                board = chess.Board(start)
            position.encode(board, search.WORK.state[0], search.WORK.mail[0])
            work = search.WORK
            nnue.refresh(work.acc, 0, work.state[0], work.mail[0])
            codes = [(0 if p.color else 6) + p.piece_type - 1
                     for _, p in sorted(board.piece_map().items())]
            record = dataset.pack(board.occupied, codes, not board.turn, 123)
            _, white, black, stm, score = dataset.unpack(
                np.frombuffer(record, dtype=np.uint8).reshape(1, 32))
            assert int(stm[0]) == int(not board.turn) and int(score[0]) == 123
            for perspective, features in enumerate((white, black)):
                reference = nnue.FT_BIAS.astype(np.int64) + nnue.FT_WEIGHT[
                    features].astype(np.int64).sum(axis=0)
                np.testing.assert_array_equal(work.acc[0, perspective], reference)
            count = movegen.generate(work.state[0], work.moves, 0)
            legal: set[str] = set()
            for packed in work.moves[:count].copy():
                position.make(work.state[0], work.mail[0], work.state[1], work.mail[1], packed)
                if not position.legal_after(work.state[1], int(not board.turn)):
                    continue
                uci = movegen.move_to_uci(int(packed))
                legal.add(uci)
                flags.add(int(packed) >> 15)
                assert tt.unpack_move(tt.pack_move(packed)) == packed
                nnue.apply(work.acc, 0, work.state[0], work.mail[0], packed)
                board.push_uci(uci)
                position.encode(board, work.state[2], work.mail[2])
                np.testing.assert_array_equal(work.state[1], work.state[2])
                np.testing.assert_array_equal(work.mail[1], work.mail[2])
                nnue.refresh(work.acc, 2, work.state[2], work.mail[2])
                np.testing.assert_array_equal(work.acc[1], work.acc[2])
                board.pop()
                edges += 1
            assert legal == {move.uci() for move in board.legal_moves}, board.fen()
            board.push_uci(rng.choice(sorted(legal)))
            positions += 1
    assert flags == {0, 1, 2, 3}
    print(f' audited {positions} positions, {edges} legal edges, all four move flags')


def test_shipped_quantisation_bounds() -> None:
    bound = np.abs(nnue.FT_BIAS.astype(np.int64)) + 32 * np.abs(
        nnue.FT_WEIGHT.astype(np.int64)).max(axis=0)
    assert bound.max() <= 32767
    output = nnue.QA**2 * np.abs(nnue.OUT_WEIGHT.astype(np.int64)).sum(axis=1)
    assert output.max() <= 2**31 - 1


def test_quiescence_stalemate() -> None:
    board, work = prepare('7k/5K2/6Q1/8/8/8/8/8 b - - 0 1')
    assert board.is_stalemate()
    assert search.qsearch(work, 0, np.int32(-32000), np.int32(32000)) == 0


@pytest.mark.parametrize('fen', [
    '8/8/8/8/8/5k2/8/R6K w - - 100 1',
    '8/8/8/8/8/5k2/8/6BK w - - 0 1',
])
def test_quiescence_draws(fen: str) -> None:
    board, work = prepare(fen)
    assert board.outcome(claim_draw=True) is not None
    assert search.qsearch(work, 0, np.int32(-32000), np.int32(32000)) == 0


def test_see_promotion() -> None:
    board, work = prepare('7k/P7/8/8/8/8/8/7K w - - 0 1')
    assert position.see(work.state[0], work.mail[0], encoded_move(board, 'a7a8q')) == 800


def test_node_counted_once_on_the_qsearch_handover() -> None:
    """One position entered is one node counted.

    White is in check with a single legal reply, so a depth-one search visits exactly one
    position below the root: negamax at ply one, which has no depth left and hands over to
    quiescence. Counting that handover as a second node is what inflated every reported
    node total and made node-budgeted comparisons incomparable.
    """
    board, _work = prepare('7k/8/8/8/8/8/6PP/6rK w - - 0 1')
    assert [move.uci() for move in board.legal_moves] == ['h1g1']
    assert search.think(board, 3_600_000, 0, max_depth=1) == 'h1g1'
    assert search.nodes() == 1


def test_pinned_pieces_matches_python_chess() -> None:
    """Every absolutely pinned piece, over random play from four different shapes."""
    rng = random.Random(20260910)
    starts = [chess.STARTING_FEN,
              'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
              '4k3/4r3/8/b7/8/3N4/8/4RK2 b - - 0 1',
              '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1']
    positions = pinned_seen = 0
    work = search.WORK
    for start in starts:
        board = chess.Board(start)
        for _ in range(600):
            if board.is_game_over() or board.ply() > 300:
                board = chess.Board(start)
            position.encode(board, work.state[0], work.mail[0])
            for colour in (chess.WHITE, chess.BLACK):
                mask = int(position.pinned_pieces(work.state[0], int(not colour)))
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece is None or piece.color != colour:
                        assert not mask >> square & 1, (board.fen(), square)
                        continue
                    expected = board.is_pinned(colour, square)
                    assert bool(mask >> square & 1) is expected, (board.fen(), square)
                    pinned_seen += expected
            board.push(rng.choice(list(board.legal_moves)))
            positions += 1
    assert pinned_seen > 50, pinned_seen
    print(f' checked pins on {positions} positions, {pinned_seen} pinned pieces')


def test_see_pinned_recapture() -> None:
    board, work = prepare('4k3/4n3/8/3p4/2B5/8/8/K3R3 w - - 0 1')
    assert position.see(work.state[0], work.mail[0], encoded_move(board, 'c4d5')) == 100


def test_history_does_not_penalise_captures() -> None:
    board, work = prepare('7k/8/8/3p4/4P3/8/8/7K w - - 0 1')
    capture = encoded_move(board, 'e4d5')
    quiet = encoded_move(board, 'h1g1')
    # The selected move list still holds the capture the node tried first; only the
    # searched-quiet list may drive the malus.
    work.moves[0], work.moves[1] = capture, quiet
    work.quiets[0] = quiet
    search.update_history(work, 0, quiet, 4, 0, 1)
    assert work.history[0, chess.E4, chess.D5] == 0
    assert work.history[0, chess.H1, chess.G1] > 0


@pytest.mark.parametrize('fen', [
    '7k/8/8/3b4/8/8/8/KB6 w - - 0 1',    # bishops on both sides, both on light squares
    '7k/8/8/8/8/8/8/KB6 w - - 0 1',      # king and one bishop
    '7k/8/8/8/8/8/8/K7 w - - 0 1',       # bare kings
    '7k/8/8/8/8/2b5/8/KB6 w - - 0 1',    # bishops on opposite colours
    '7k/8/8/8/8/2n5/8/KB6 w - - 0 1',    # a knight can change the colour of the net
    '7k/8/8/8/8/8/P7/K7 w - - 0 1',      # a pawn can promote
])
def test_insufficient_material_matches_python_chess(fen: str) -> None:
    board, work = prepare(fen)
    assert bool(position.insufficient_material(work.state[0])) is (
        board.is_insufficient_material()
    )


@pytest.mark.parametrize(('uci', 'fen', 'quiet'), [
    ('e4d5', '7k/8/8/3p4/4P3/8/8/7K w - - 0 1', False),
    ('h1g1', '7k/8/8/3p4/4P3/8/8/7K w - - 0 1', True),
    ('e5d6', 'k7/8/8/3pP3/8/8/8/7K w - d6 0 1', False),
    ('a7a8q', '7k/P7/8/8/8/8/8/7K w - - 0 1', False),
    ('e1g1', '4k3/8/8/8/8/8/8/4K2R w K - 0 1', True),
])
def test_quiet_classification(uci: str, fen: str, quiet: bool) -> None:
    board, work = prepare(fen)
    assert bool(search.is_quiet(work.mail[0], encoded_move(board, uci))) is quiet


def test_root_continuation_piece() -> None:
    board, work = prepare('7k/8/8/8/8/8/6r1/7K w - - 0 1')
    search.search_root(work, 1)
    piece = board.piece_type_at(int(work.played[0]) & 63)
    assert piece is not None
    assert work.moved_piece[0] == piece - 1


@pytest.mark.parametrize(('fen', 'legal'), [
    # The capturing pawn is the only thing between the king and the rook on the e-file.
    ('k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', False),
    # The classic horizontal case: taking removes both pawns from the fifth rank at
    # once, and neither square alone was blocking the rook.
    ('8/8/8/K2pP2r/8/8/8/7k w - d6 0 1', False),
    # A rook on the file the pawn leaves, with the king off that file.
    ('4r3/8/8/3pP3/8/8/8/k5K1 w - d6 0 1', True),
    ('8/8/8/3pP3/8/8/8/k3K3 w - d6 0 1', True),
])
def test_ep_hash_follows_legality(fen: str, legal: bool) -> None:
    board, work = prepare(fen)
    assert board.has_legal_en_passant() is legal
    assert bool(position.ep_is_capturable(work.state[0], chess.D6)) is legal
    # The key may only carry the en passant term when the capture is really available.
    key = int(work.state[0, bb.KEY])
    board.ep_square = None
    position.encode(board, work.state[0], work.mail[0])
    assert (int(work.state[0, bb.KEY]) == key) is not legal


def test_mate_precedes_fifty_move_draw() -> None:
    board, work = prepare('7k/6Q1/5K2/8/8/8/8/8 b - - 100 1')
    assert board.is_checkmate()
    position.encode(board, work.state[1], work.mail[1])
    nnue.refresh(work.acc, 1, work.state[1], work.mail[1])
    assert search.negamax(work, 1, 1, np.int32(-32000), np.int32(32000), True) == -29999


def test_game_tracking_after_opponent_double_pawn_push() -> None:
    board = chess.Board()
    agent._remember(board, chess.Move.from_uci('e2e4'))
    board.push_uci('e2e4')
    board.push_uci('e7e5')
    incoming = chess.Board(board.fen())  # Same serialization used by the unchanged harness.
    assert incoming.ep_square is None and board.ep_square == chess.E6
    assert agent._continues_our_game(incoming)
