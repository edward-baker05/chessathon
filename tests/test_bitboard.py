"""Bit primitives and attack tables, checked against python-chess."""

import chess
import numpy as np
import pytest

import bitboard as bb
from tests.conftest import random_positions


@pytest.mark.parametrize("square", range(64))
def test_lsb_finds_lowest_set_bit(square: int) -> None:
    assert bb.lsb(np.uint64(1) << np.uint64(square)) == square


def test_lsb_ignores_higher_bits() -> None:
    value = (np.uint64(1) << np.uint64(9)) | (np.uint64(1) << np.uint64(40))
    assert bb.lsb(value) == 9


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 1), (0xFF, 8), (0xFFFFFFFFFFFFFFFF, 64)])
def test_popcount(value: int, expected: int) -> None:
    assert bb.popcount(np.uint64(value)) == expected


@pytest.mark.parametrize("square", range(64))
def test_knight_and_king_tables_match_python_chess(square: int) -> None:
    assert int(bb.KNIGHT_ATT[square]) == chess.BB_KNIGHT_ATTACKS[square]
    assert int(bb.KING_ATT[square]) == chess.BB_KING_ATTACKS[square]


@pytest.mark.parametrize("square", range(64))
def test_pawn_attack_table_orientation(square: int) -> None:
    # PAWN_ATT[0] is what a WHITE pawn on `square` attacks.
    assert int(bb.PAWN_ATT[0, square]) == int(chess.BB_PAWN_ATTACKS[chess.WHITE][square])
    assert int(bb.PAWN_ATT[1, square]) == int(chess.BB_PAWN_ATTACKS[chess.BLACK][square])


def test_slider_attacks_match_python_chess_on_real_positions() -> None:
    for board in random_positions(count=60, seed=11):
        occ = np.uint64(board.occupied)
        for square in range(64):
            expected_rook = int(
                chess.BB_RANK_ATTACKS[square][board.occupied & chess.BB_RANK_MASKS[square]]
                | chess.BB_FILE_ATTACKS[square][board.occupied & chess.BB_FILE_MASKS[square]]
            )
            expected_bishop = int(
                chess.BB_DIAG_ATTACKS[square][board.occupied & chess.BB_DIAG_MASKS[square]]
            )
            assert int(bb.rook_attacks(square, occ)) == expected_rook, f"rook on {square}"
            assert int(bb.bishop_attacks(square, occ)) == expected_bishop, f"bishop on {square}"
            assert int(bb.queen_attacks(square, occ)) == expected_rook | expected_bishop


def test_zobrist_keys_are_distinct_and_nonzero() -> None:
    keys = set()
    for colour in range(2):
        for piece in range(6):
            for square in range(64):
                key = int(bb.Z_PIECE[colour, piece, square])
                assert key != 0
                keys.add(key)
    assert len(keys) == 2 * 6 * 64, "Zobrist keys must not collide"


def test_castle_mask_clears_the_right_rights() -> None:
    # bits: 1 white kingside, 2 white queenside, 4 black kingside, 8 black queenside
    assert bb.CASTLE_MASK[chess.E1] == 15 ^ 3
    assert bb.CASTLE_MASK[chess.A1] == 15 ^ 2
    assert bb.CASTLE_MASK[chess.H1] == 15 ^ 1
    assert bb.CASTLE_MASK[chess.E8] == 15 ^ 12
    assert bb.CASTLE_MASK[chess.A8] == 15 ^ 8
    assert bb.CASTLE_MASK[chess.H8] == 15 ^ 4
    assert bb.CASTLE_MASK[chess.D4] == 15
