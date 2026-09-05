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
