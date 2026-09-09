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


# Black is a rook down and has four men: a king with no square and three pawns each
# blocked head on. A gate that only looked for stalemate with three men or fewer scored
# this as the static evaluation of a lost position instead of as the draw it is.
STALEMATE_FOUR_MEN = '7k/7p/7p/7p/7P/8/8/K5R1 b - - 0 1'
# White is two pawns up on material and stalemated: a king with no square, a rook boxed in
# behind its own pawn, and four pawns each blocked head on. Six men, and a rook that makes
# the null move available, which is what lets all three speculative cutoffs fire on it.
STALEMATE_AHEAD = '5r2/8/8/p2k4/P7/P6p/P6P/6KR w - - 0 1'
# One move earlier, Black to move. Black is winning here, so nothing forces ...Rf8; it is
# a legal move into the stalemate, and it is how these tests reach the terminal node the
# way a game reaches it rather than from the root.
STALEMATE_AHEAD_PARENT = '4r3/8/8/p2k4/P7/P6p/P6P/6KR b - - 0 1'


def descend(board: chess.Board, uci: str, work: Any) -> None:
    """Make a real move into ply one, accumulator and all, the way the search does.

    A terminal position tested only at the root is tested on the one code path the search
    never uses. This reaches it the way a game does.
    """
    move = encoded_move(board, uci)
    position.make(work.state[0], work.mail[0], work.state[1], work.mail[1], move)
    nnue.apply(work.acc, 0, work.state[0], work.mail[0], move)


def static_of(work: Any, ply: int = 0) -> int:
    return int(nnue.forward(work.acc, ply, work.state[ply]))


def men_to_move(board: chess.Board) -> int:
    return sum(1 for piece in board.piece_map().values() if piece.color == board.turn)


def test_quiescence_stalemate_with_more_than_three_men() -> None:
    board, work = prepare(STALEMATE_FOUR_MEN)
    assert board.is_stalemate() and men_to_move(board) == 4
    assert static_of(work) < 0, 'the fixture only bites while the static score is wrong'
    assert search.qsearch(work, 0, np.int32(-32000), np.int32(32000)) == 0


def test_quiescence_mate_still_outranks_the_stalemate_test() -> None:
    """The draw test must not swallow a checkmate with the same number of men."""
    board, work = prepare('R5k1/5ppp/8/8/8/8/8/K7 b - - 0 1')
    assert board.is_checkmate() and men_to_move(board) == 4
    assert search.qsearch(work, 0, np.int32(-32000), np.int32(32000)) == -search.MATE


@pytest.mark.parametrize('fen', [STALEMATE_FOUR_MEN, STALEMATE_AHEAD])
@pytest.mark.parametrize('depth', [1, 2, 5])
def test_stalemate_scores_zero_in_every_window(fen: str, depth: int) -> None:
    """Zero is the value of a terminal node, so no window may return anything else.

    A narrow window that straddles the static score is what an interior node of a real
    search hands down, and it is the case the early returns are reached through.
    """
    board, work = prepare(fen)
    assert board.is_stalemate()
    static = static_of(work)
    for alpha in (-31000, static - 200, -1, 0, 1, static + 1000):
        for is_pv in (False, True):
            board, work = prepare(fen)
            got = search.negamax(work, 0, depth, np.int32(alpha), np.int32(alpha + 1), is_pv)
            assert got == 0, (fen, depth, alpha, is_pv, int(got))


def test_stalemate_reached_through_a_legal_move() -> None:
    board, work = prepare(STALEMATE_AHEAD_PARENT)
    assert not board.is_stalemate()
    descend(board, 'e8f8', work)
    # Windows are kept inside the non-mate range. Outside it the mate-distance clamp at
    # ply one collapses the window and returns a bound of its own, which is correct and
    # says nothing about the terminal node below.
    for alpha in (-2000, -1000, -1, 0, 1, 1000, 2000):
        work.ints[search.I_ABORT] = 0
        got = search.negamax(work, 1, 4, np.int32(alpha), np.int32(alpha + 1), False)
        assert got == 0, (alpha, int(got))


def test_reverse_futility_does_not_score_a_stalemate() -> None:
    """The cutoff returns the static score instead of searching, so it has to look first.

    The window is built from the static score so that the cutoff certainly fires. On this
    fixture the value it used to return, the static score of a lost position, was a loose
    lower bound rather than a wrong one; on a stalemate whose static score is above both
    beta and zero the same cutoff is a wrong bound outright. Establishing the terminal
    before the cutoff removes the whole class, and the value it returns here is now the
    value of the position rather than of the pieces standing on it.
    """
    _board, work = prepare(STALEMATE_AHEAD)
    static = static_of(work)
    beta = np.int32(static - search.RFP_MARGIN)
    assert static - search.RFP_MARGIN * 1 >= beta, 'the cutoff has to fire for this to test it'
    assert search.negamax(work, 0, 1, np.int32(beta - 1), beta, False) == 0


def test_null_move_does_not_score_a_stalemate() -> None:
    """Passing the move is only meaningful where there was a move to make.

    This one pins the guard rather than reproducing a failure: before it was added the
    null search fell through to the move loop and reached zero anyway. It is here because
    the null move is the third of the three cutoffs that return a score without searching,
    and it should not be the one left without a terminal test.
    """
    _board, work = prepare(STALEMATE_AHEAD)
    assert position.has_non_pawn_material(work.state[0], 0), 'null move needs a piece'
    static = static_of(work)
    # beta = static clears the null-move condition and leaves reverse futility, which
    # needs a further margin per ply, and razoring, which needs alpha above the static
    # score, both unfired at depth three.
    beta = np.int32(static)
    assert static - search.RFP_MARGIN * 3 < beta <= static
    assert search.negamax(work, 0, 3, np.int32(beta - 1), beta, False) == 0


def test_has_free_move_never_claims_a_move_that_is_not_there() -> None:
    """The fast legality path is one-sided: True has to mean a legal move exists.

    How often it decides matters too, because every undecided node pays for full move
    generation instead. The floor is far below what it measures and is there to catch a
    change that quietly turns the fast path off.
    """
    rng = random.Random(20260912)
    starts = [chess.STARTING_FEN,
              'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
              '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
              '8/5p2/4k3/8/3K4/8/5P2/8 w - - 0 40',
              '8/8/4k3/8/2p5/8/B2P2KP/8 w - - 0 1']
    work = search.WORK
    asked = decided = 0
    for start in starts:
        board = chess.Board(start)
        for _ in range(1200):
            if board.is_game_over() or board.ply() > 250:
                board = chess.Board(start)
            if not board.is_check():
                position.encode(board, work.state[0], work.mail[0])
                certain = bool(movegen.has_free_move(work.state[0]))
                asked += 1
                decided += certain
                if certain:
                    assert any(board.legal_moves), board.fen()
            board.push(rng.choice(list(board.legal_moves)))
    for fen in (STALEMATE_FOUR_MEN, STALEMATE_AHEAD, '7k/5K2/6Q1/8/8/8/8/8 b - - 0 1'):
        position.encode(chess.Board(fen), work.state[0], work.mail[0])
        assert not movegen.has_free_move(work.state[0])
    assert decided / asked > 0.95, decided / asked
    print(f' {asked} positions, fast path decided {decided / asked:.2%}')


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


def test_see_pin_released_by_the_capture() -> None:
    """Rxd1 unpins the knight that was pinned by the rook itself.

    Re1 pins Ne3 to Ke8 down the e-file. The moment the rook steps off that file to take
    on d1 the pin is gone and ...Nxd1 is legal, so the capture loses the exchange. Reading
    the pins once from the starting position keeps the knight excluded and reports a clean
    win of a bishop instead.
    """
    board, work = prepare('4k3/8/8/8/8/4n3/8/K2bR3 w - - 0 1')
    assert position.see(work.state[0], work.mail[0], encoded_move(board, 'e1d1')) == -170


def test_see_recapture_that_promotes() -> None:
    """Rxa1 wins a bishop and loses a rook to a pawn that queens on the recapture."""
    board, work = prepare('8/7k/8/8/8/R7/1p6/b6K w - - 0 1')
    assert position.see(work.state[0], work.mail[0], encoded_move(board, 'a3a1')) == -970


def test_see_pin_created_by_the_capture() -> None:
    """Nxc5 pins the bishop that would have recaptured.

    Be7 and Ne4 both stand between Ke8 and Re1, so neither is pinned to begin with. The
    knight is also the piece that captures on c5, and the moment it leaves the e-file the
    bishop is pinned and ...Bxc5 is illegal. Reading the pins once from the starting
    position lets the bishop recapture and turns a won pawn into a lost knight.
    """
    board, work = prepare('4k3/4b3/8/2p5/4N3/8/8/4R2K w - - 0 1')
    assert position.see(work.state[0], work.mail[0], encoded_move(board, 'e4c5')) == 100


@pytest.mark.parametrize(('fen', 'expected'), [
    # d5 is covered by Nc3, so ...Kxd5 is illegal and the pawn is simply won.
    ('8/8/2k5/3p4/4P3/2N5/8/4K3 w - - 0 1', 100),
    # Nothing covers d5, so ...Kxd5 is legal and the exchange is even.
    ('8/8/2k5/3p4/4P3/8/8/4K3 w - - 0 1', 0),
])
def test_see_king_recapture(fen: str, expected: int) -> None:
    """A king may only recapture onto a square the other side no longer attacks."""
    board, work = prepare(fen)
    assert position.see(work.state[0], work.mail[0], encoded_move(board, 'e4d5')) == expected


SEE_VALUES = dict(zip(range(1, 7), (int(v) for v in position.SEE_VALUE), strict=True))


def see_oracle(board: chess.Board, move: chess.Move) -> tuple[int, set[str]]:
    """The same swap algorithm, but with every recapture checked by python-chess.

    This is the independent reference for `see`. It resolves each recapture by asking for
    legal moves rather than by reasoning about pins, so it is right by construction on the
    geometry and wrong only where the swap algorithm itself is an approximation. The tags
    name the two places that happens, so the test can hold the rest to exact agreement.
    """
    target = move.to_square
    replay = board.copy(stack=False)
    victim = replay.piece_at(target)
    first = SEE_VALUES[victim.piece_type] if victim else 0
    if replay.is_en_passant(move):
        first = SEE_VALUES[chess.PAWN]
    if move.promotion:
        first += SEE_VALUES[move.promotion] - SEE_VALUES[chess.PAWN]
    gains = [first]
    replay.push(move)
    tags: set[str] = set()
    while True:
        # SEE never asks whether the side to move is in check, so a recapture that is only
        # illegal because a check is outstanding is outside what it can express.
        if replay.is_check():
            tags.add('in-check')
        moves = [m for m in replay.legal_moves
                 if m.to_square == target and m.promotion in (None, chess.QUEEN)]
        if not moves:
            break
        moves.sort(key=lambda m: (SEE_VALUES[_piece_type(replay, m)], 0 if m.promotion else 1))
        cheapest = SEE_VALUES[_piece_type(replay, moves[0])]
        # Two attackers of the same value are not interchangeable: which one recaptures
        # decides which lines stay blocked. The swap algorithm commits to one of them.
        if sum(SEE_VALUES[_piece_type(replay, m)] == cheapest for m in moves) > 1:
            tags.add('equal-attackers')
        standing = replay.piece_at(target)
        assert standing is not None
        gain = SEE_VALUES[standing.piece_type] - gains[-1]
        if moves[0].promotion:
            gain += SEE_VALUES[chess.QUEEN] - SEE_VALUES[chess.PAWN]
        gains.append(gain)
        replay.push(moves[0])
    for index in range(len(gains) - 1, 0, -1):
        gains[index - 1] = -max(-gains[index - 1], gains[index])
    return gains[0], tags


def _piece_type(board: chess.Board, move: chess.Move) -> int:
    piece = board.piece_at(move.from_square)
    assert piece is not None
    return piece.piece_type


def test_see_matches_a_legality_checked_oracle() -> None:
    """Every capture and queen promotion over random play, against the oracle above.

    Disagreements are allowed only where the swap algorithm cannot represent the position:
    an outstanding check, or a choice between two attackers of equal value. Anything else
    is a defect in the pin handling or in the material bookkeeping.
    """
    rng = random.Random(20260911)
    starts = [chess.STARTING_FEN,
              'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
              '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
              '4k3/4r3/8/b7/8/3N4/8/4RK2 b - - 0 1',
              '8/7k/8/8/8/R7/1p6/b6K w - - 0 1']
    work = search.WORK
    checked = unexplained = excused = 0
    for index in range(35):
        board = chess.Board(starts[index % len(starts)])
        for _ in range(120):
            if board.is_game_over():
                break
            position.encode(board, work.state[0], work.mail[0])
            for move in board.legal_moves:
                if board.is_castling(move) or move.promotion not in (None, chess.QUEEN):
                    continue
                if not (board.is_capture(move) or move.promotion):
                    continue
                checked += 1
                got = int(position.see(work.state[0], work.mail[0],
                                       encoded_move(board, move.uci())))
                want, tags = see_oracle(board, move)
                if got == want:
                    continue
                if tags:
                    excused += 1
                    continue
                unexplained += 1
                assert got == want, (board.fen(), move.uci(), got, want)
            board.push(rng.choice(list(board.legal_moves)))
    assert checked > 6000, checked
    assert unexplained == 0
    print(f' checked {checked} captures, {excused} excused by an approximation')


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
