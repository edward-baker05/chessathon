"""Transposition table storage, replacement and mate-score rebasing."""

import numpy as np

import tt


def setup_function() -> None:
    tt.tt_clear(tt.TT)


def test_probe_misses_on_an_empty_table() -> None:
    hit, _, _, _, _, _ = tt.tt_probe(tt.TT, np.uint64(0x1234), 0)
    assert not hit


def test_store_then_probe_round_trips() -> None:
    key = np.uint64(0xDEADBEEFCAFEF00D)
    tt.tt_store(tt.TT, key, 0, np.int32(-137), np.int32(1234), 7, tt.BOUND_EXACT, np.int32(42), 1)
    hit, score, move, depth, bound, static = tt.tt_probe(tt.TT, key, 0)
    assert hit
    assert score == -137
    assert move == 1234
    assert depth == 7
    assert bound == tt.BOUND_EXACT
    assert static == 42


def test_negative_static_eval_survives_packing() -> None:
    key = np.uint64(0x1111)
    tt.tt_store(
        tt.TT, key, 0, np.int32(-30000), np.int32(7), 3, tt.BOUND_UPPER, np.int32(-4321), 1
    )
    hit, score, _, _, bound, static = tt.tt_probe(tt.TT, key, 0)
    assert hit and score == -30000 and static == -4321 and bound == tt.BOUND_UPPER


def test_different_keys_do_not_collide_within_a_bucket() -> None:
    for i in range(4):
        tt.tt_store(
            tt.TT, np.uint64(0x1000 + i), 0, np.int32(i), np.int32(i), 5,
            tt.BOUND_EXACT, np.int32(0), 1,
        )
    for i in range(4):
        hit, score, _, _, _, _ = tt.tt_probe(tt.TT, np.uint64(0x1000 + i), 0)
        assert hit and score == i


def test_mate_scores_are_stored_relative_to_the_node() -> None:
    key = np.uint64(0xABCD)
    mate_at_root = np.int32(tt.MATE - 10)
    # Stored while searching at ply 4, so the entry holds a mate 4 plies nearer from there.
    tt.tt_store(tt.TT, key, 4, mate_at_root, np.int32(0), 9, tt.BOUND_EXACT, np.int32(0), 1)
    hit, score, _, _, _, _ = tt.tt_probe(tt.TT, key, 4)
    assert hit and score == mate_at_root

    hit, score, _, _, _, _ = tt.tt_probe(tt.TT, key, 0)
    assert hit and score == mate_at_root + 4, "a mate found deeper is nearer when probed shallower"


def test_being_mated_scores_rebase_the_other_way() -> None:
    key = np.uint64(0xBEEF)
    mated = np.int32(-tt.MATE + 10)
    tt.tt_store(tt.TT, key, 4, mated, np.int32(0), 9, tt.BOUND_EXACT, np.int32(0), 1)
    hit, score, _, _, _, _ = tt.tt_probe(tt.TT, key, 0)
    assert hit and score == mated - 4


def test_non_mate_scores_are_not_rebased() -> None:
    key = np.uint64(0xFEED)
    tt.tt_store(tt.TT, key, 6, np.int32(250), np.int32(0), 4, tt.BOUND_EXACT, np.int32(0), 1)
    hit, score, _, _, _, _ = tt.tt_probe(tt.TT, key, 0)
    assert hit and score == 250


def test_deeper_entry_is_not_replaced_by_a_shallower_one_of_the_same_age() -> None:
    key = np.uint64(0x5555)
    tt.tt_store(tt.TT, key, 0, np.int32(100), np.int32(11), 12, tt.BOUND_EXACT, np.int32(0), 1)
    tt.tt_store(tt.TT, key, 0, np.int32(200), np.int32(22), 3, tt.BOUND_EXACT, np.int32(0), 1)
    hit, score, move, depth, _, _ = tt.tt_probe(tt.TT, key, 0)
    assert hit and depth == 12 and score == 100 and move == 11


def test_deeper_entry_does_replace_a_shallower_one() -> None:
    key = np.uint64(0x6666)
    tt.tt_store(tt.TT, key, 0, np.int32(100), np.int32(11), 3, tt.BOUND_EXACT, np.int32(0), 1)
    tt.tt_store(tt.TT, key, 0, np.int32(200), np.int32(22), 12, tt.BOUND_EXACT, np.int32(0), 1)
    hit, score, move, depth, _, _ = tt.tt_probe(tt.TT, key, 0)
    assert hit and depth == 12 and score == 200 and move == 22


def test_clear_empties_the_table() -> None:
    tt.tt_store(
        tt.TT, np.uint64(0x99), 0, np.int32(1), np.int32(1), 1, tt.BOUND_EXACT, np.int32(0), 1
    )
    tt.tt_clear(tt.TT)
    hit, _, _, _, _, _ = tt.tt_probe(tt.TT, np.uint64(0x99), 0)
    assert not hit


def test_table_is_a_power_of_two_number_of_buckets() -> None:
    assert tt.BUCKETS & (tt.BUCKETS - 1) == 0, "index masking requires a power of two"


def test_every_move_flag_survives_a_round_trip() -> None:
    """A move needs seventeen bits. Masking it to sixteen drops the high flag bit, which
    turns a stored castle into a quiet move and a stored promotion into an en passant
    capture. Neither corrupted move ever matches a generated one, so the node silently
    loses its TT move and, with it, a ply to the internal iterative reduction."""
    moves = {
        "castle kingside": 4 | (6 << 6) | (2 << 15),
        "castle queenside": 4 | (2 << 6) | (2 << 15),
        "promotion to queen": 52 | (60 << 6) | (4 << 12) | (3 << 15),
        "promotion to knight": 52 | (60 << 6) | (1 << 12) | (3 << 15),
        "en passant": 36 | (43 << 6) | (1 << 15),
        "quiet": 12 | (28 << 6),
    }
    for index, (name, move) in enumerate(moves.items()):
        key = np.uint64(0x5000 + index)
        tt.tt_store(tt.TT, key, 0, np.int32(1), np.int32(move), 4, tt.BOUND_EXACT, np.int32(0), 1)
        hit, _, stored, _, _, _ = tt.tt_probe(tt.TT, key, 0)
        assert hit and stored == move, f"{name} did not survive packing"


def test_the_packed_word_uses_every_bit_it_claims_and_no_more() -> None:
    widths = (
        (tt.SCORE_SHIFT, tt.SCORE_BITS),
        (tt.MOVE_SHIFT, tt.MOVE_BITS),
        (tt.DEPTH_SHIFT, tt.DEPTH_BITS),
        (tt.BOUND_SHIFT, tt.BOUND_BITS),
        (tt.AGE_SHIFT, tt.AGE_BITS),
        (tt.STATIC_SHIFT, tt.STATIC_BITS),
    )
    occupied = 0
    for shift, bits in widths:
        field = ((1 << bits) - 1) << shift
        assert occupied & field == 0, "two fields overlap in the packed word"
        occupied |= field
    assert occupied == (1 << 64) - 1, "the packed word has unused or overflowing bits"


def test_extreme_field_values_do_not_bleed_into_their_neighbours() -> None:
    key = np.uint64(0x6001)
    # The widest real move there is: h8 to h8, promoting to a queen.
    move = 63 | (63 << 6) | (4 << 12) | (3 << 15)
    tt.tt_store(
        tt.TT, key, 0, np.int32(-32768), np.int32(move), tt.DEPTH_MASK,
        tt.BOUND_LOWER, np.int32(32767), tt.AGE_MASK,
    )
    hit, score, stored, depth, bound, static = tt.tt_probe(tt.TT, key, 0)
    assert hit
    assert score == -32768 and stored == move and depth == tt.DEPTH_MASK
    assert bound == tt.BOUND_LOWER and static == 32767


def test_every_move_the_generator_can_produce_round_trips() -> None:
    """The stored form is sixteen bits and a generator move is seventeen, so the encoding
    has to be exhaustively reversible rather than merely reversible for the common case."""
    assert tt.unpack_move(tt.pack_move(np.int32(0))) == 0, "the empty move must survive"
    for frm in (0, 7, 31, 56, 63):
        for to in (0, 12, 40, 63):
            for flag, promos in ((0, (0,)), (1, (0,)), (2, (0,)), (3, (1, 2, 3, 4))):
                for promo in promos:
                    move = frm | (to << 6) | (promo << 12) | (flag << 15)
                    assert tt.unpack_move(tt.pack_move(np.int32(move))) == move, (
                        f"from {frm} to {to} promo {promo} flag {flag} did not round trip"
                    )
