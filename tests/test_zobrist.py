"""The incrementally updated key must equal the from-scratch key, always."""

import chess
import numpy as np

import bitboard as bb
import movegen
import position
from tests.conftest import random_positions


def test_incremental_key_matches_full_recompute() -> None:
    state, mailbox = position.new_stacks()
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    for board in random_positions(count=200, seed=51):
        position.encode(board, state[0], mailbox[0])
        count = movegen.generate(state[0], moves, 0)
        for i in range(count):
            position.make(state[0], mailbox[0], state[1], mailbox[1], moves[i])
            assert state[1][bb.KEY] == position.full_key(state[1]), (
                f"key diverged after {movegen.move_to_uci(int(moves[i]))} in {board.fen()}"
            )


def test_key_survives_a_long_random_walk() -> None:
    """Incremental errors compound, so a deep walk catches what one ply hides."""
    state, mailbox = position.new_stacks()
    moves = np.zeros(64 * movegen.MAX_MOVES, dtype=np.int32)
    board = chess.Board()
    position.encode(board, state[0], mailbox[0])
    rng = np.random.default_rng(52)
    ply = 0
    while ply < 60:
        count = movegen.generate(state[ply], moves, ply * movegen.MAX_MOVES)
        mover_black = int(state[ply][bb.STM])
        legal = []
        for i in range(ply * movegen.MAX_MOVES, count):
            position.make(state[ply], mailbox[ply], state[ply + 1], mailbox[ply + 1], moves[i])
            if position.legal_after(state[ply + 1], mover_black):
                legal.append(int(moves[i]))
        if not legal:
            break
        chosen = np.int32(legal[rng.integers(len(legal))])
        position.make(state[ply], mailbox[ply], state[ply + 1], mailbox[ply + 1], chosen)
        ply += 1
        assert state[ply][bb.KEY] == position.full_key(state[ply]), f"diverged at ply {ply}"


def test_transpositions_reach_the_same_key() -> None:
    left = chess.Board()
    for uci in ("g1f3", "g8f6", "d2d4", "d7d5"):
        left.push_uci(uci)
    right = chess.Board()
    for uci in ("d2d4", "d7d5", "g1f3", "g8f6"):
        right.push_uci(uci)
    state, mailbox = position.new_stacks()
    position.encode(left, state[0], mailbox[0])
    position.encode(right, state[1], mailbox[1])
    assert state[0][bb.KEY] == state[1][bb.KEY]


def test_en_passant_only_hashes_when_a_capture_is_available() -> None:
    # A double push with no enemy pawn able to take must hash the same as the quiet
    # position, or otherwise identical positions get different keys.
    with_push = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    with_push.push_uci("e2e4")
    same_without_ep = chess.Board("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1")
    state, mailbox = position.new_stacks()
    position.encode(with_push, state[0], mailbox[0])
    position.encode(same_without_ep, state[1], mailbox[1])
    assert state[0][bb.KEY] == state[1][bb.KEY]


def test_en_passant_does_hash_when_a_capture_is_available() -> None:
    with_push = chess.Board("4k3/8/8/8/5p2/8/4P3/4K3 w - - 0 1")
    with_push.push_uci("e2e4")
    without_ep = chess.Board("4k3/8/8/8/4Pp2/8/8/4K3 b - - 0 1")
    state, mailbox = position.new_stacks()
    position.encode(with_push, state[0], mailbox[0])
    position.encode(without_ep, state[1], mailbox[1])
    assert state[0][bb.KEY] != state[1][bb.KEY], "f4 can take e3, so the ep square matters"
