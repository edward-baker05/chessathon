"""Search: iterative deepening, PVS, quiescence, move ordering and time control.

Everything the search mutates travels inside one `Work` namedtuple, because numba exposes
module globals to jitted code as *readonly* arrays. Read-only tables (attack tables,
Zobrist keys) stay as globals in bitboard.py, where being readonly is exactly right.

Time is read through numba's objmode, since time.time() is not available in nopython mode.
An objmode call costs about 1.8 us, so it happens every 2048 nodes: roughly 0.9 ns a node.
"""

import math
import os
import time
from collections.abc import Iterable
from typing import Any, NamedTuple

import chess
import numpy as np
from numba import njit, objmode

from bitboard import FLAG_EP, FLAG_PROMO, HALF, KEY, PAWN, STM
from evaluate import evaluate
from movegen import (
    MAX_MOVES,
    generate,
    generate_captures,
    has_free_move,
    has_legal_move,
    move_to_uci,
)
from nnue import apply as nnue_apply
from nnue import copy as nnue_copy
from nnue import new_accumulator
from nnue import refresh as nnue_refresh
from position import (
    SEE_VALUE,
    STACK_PLIES,
    encode,
    has_non_pawn_material,
    in_check,
    insufficient_material,
    legal_after,
    make,
    make_null,
    new_stacks,
    see,
)
from tt import (
    AGE_MASK,
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER,
    TT,
    tt_clear,
    tt_probe,
    tt_store,
)
from tt import MATE as MATE
from tt import MATE_IN_MAX as MATE_IN_MAX

Bits = Any
Square = Any
Flag = Any

INF = 32000
MAX_DEPTH = 127

# Indices into Work.ints.
I_NODES, I_ABORT, I_NODE_LIMIT, I_HIST_LEN, I_AGE, I_SELDEPTH, I_NO_PRUNING = 0, 1, 2, 3, 4, 5, 6
# Consecutive completed iterations whose best move did not change.
I_STABLE = 7
# The last iteration that actually completed, and its score. Written only so that agent.py
# can log what the search did; nothing in the search reads them back.
I_DEPTH, I_SCORE = 8, 9
# One bit per selective mechanism; see MECHANISMS.
I_MECHANISMS = 10
# One counter per mechanism, from here. They live in `ints` rather than in an array of
# their own because every jitted function takes the whole `Work` tuple, and a tuple with
# one more field in it measured slower than the counters themselves cost.
I_COUNTS = 11
INT_SLOTS = I_COUNTS + 8

# --------------------------------------------------------------------------------------
# Selectivity switches and counters.
#
# `set_pruning(False)` turns everything off at once, and on the way also turns off the
# check extension and the transposition cutoffs. That makes it an oracle for "what does
# plain alpha-beta say", and it makes it useless for "which mechanism lost this move":
# three things changed, so the answer names none of them.
#
# These are the individual switches. Each mechanism reads one bit, and each writes one
# counter every time it actually removes work, so a diagnostic can turn exactly one off,
# leave the rest running, and see both what changed and how often the mechanism fired.
# The order of the names is the order of the bits and of the counter slots.
MECHANISMS = ("rfp", "razor", "null", "futility", "lmp", "see", "lmr", "iir")
M_RFP, M_RAZOR, M_NULL, M_FUTILITY, M_LMP, M_SEE, M_LMR, M_IIR = (
    1 << index for index in range(len(MECHANISMS))
)
C_RFP, C_RAZOR, C_NULL, C_FUTILITY, C_LMP, C_SEE, C_LMR, C_IIR = (
    I_COUNTS + index for index in range(len(MECHANISMS))
)
ALL_MECHANISMS = (1 << len(MECHANISMS)) - 1

# Indices into Work.floats. Both arrays are one-dimensional, so adding slots does not
# change any jitted signature and costs no compile time.
F_HARD, F_SOFT, F_SOFT_SPAN, F_START = 0, 1, 2, 3
F_STRETCH, F_LAST_ITER, F_PREV_ITER = 4, 5, 6
FLOAT_SLOTS = 7

# --------------------------------------------------------------------------------------
# Pruning margins, in centipawns. These were tuned against a material evaluation and mean
# something different under a network: the same scale, but a different distribution of
# scores across it, so a margin that cut the right 5% of nodes before may cut 2% or 15% now.
#
# They are read from the environment so a sweep does not need a code edit per setting.
# numba freezes module globals at compile time and compilation happens at import, so a value
# read here is baked into the jitted code exactly as a hard coded literal would be, and costs
# nothing at run time. That is what lets `tests/match.py` A/B two settings as two snapshots.
# --------------------------------------------------------------------------------------


def _margin(name: str, default: int) -> int:
    """One pruning margin, overridable as `CHESS_<name>` for a sweep."""
    override = os.environ.get(f"CHESS_{name}")
    return default if override is None else int(override)


# Reverse futility: so far ahead that giving back this much per remaining ply still beats
# beta, so the opponent would have avoided the line.
RFP_MARGIN = _margin("RFP_MARGIN", 75)
# Razoring: so far below alpha that only a capture sequence could rescue the position.
RAZOR_MARGIN = _margin("RAZOR_MARGIN", 200)
# Futility: a quiet move this far below alpha is not going to close the gap.
FUTILITY_BASE = _margin("FUTILITY_BASE", 100)
FUTILITY_MARGIN = _margin("FUTILITY_MARGIN", 90)
# SEE pruning, per ply of remaining depth, for quiet moves and for captures.
SEE_QUIET_MARGIN = _margin("SEE_QUIET_MARGIN", 50)
SEE_CAPTURE_MARGIN = _margin("SEE_CAPTURE_MARGIN", 100)
# Delta pruning in quiescence: this far below alpha even after the capture wins its victim.
DELTA_MARGIN = _margin("DELTA_MARGIN", 200)

# --------------------------------------------------------------------------------------
# The fifty-move rule is the one thing that decides a game and is not in the position key.
# Two consequences, and this constant is the scope of the answer to both.
#
# A score cached in the transposition table was searched under whatever halfmove clock the
# position had at the time. The same position hashes identically at clock 3 and at clock
# 99, and it is worth +2100 at one and zero at the other, so the entry carries one bit
# saying which side of this band it was searched on, and a score may only be reused when
# the entry and the node asking both sit outside it.
#
# A cutoff that returns a score instead of searching cannot stand in for a search that was
# about to find the draw, so inside the band the speculative prunings are off and the
# search runs.
#
# Correctness scope. Reverse futility runs to depth 8 and razoring to depth 3, so a line
# either of them replaces cannot reach the draw from outside a band of sixteen plies. The
# null move is not bounded that way, because make_null zeroes the clock underneath it, so
# the pruning test widens to the remaining depth where that is larger. What is left
# outside the scope is a node more than sixteen plies from the threshold whose own search
# shuffles all the way to it; that node's score, and any entry made from it, is still
# computed without the draw in view. Widening the band is the fix for that if it ever
# matters, and it costs pruning in exactly the endgames where the clock is already high.
RULE50_BAND = 16
# The counter the rule fires on, in plies. FIDE counts fifty moves by each side.
RULE50_LIMIT = 100

# Abandon a root iteration the moment the aspiration window fails high. The bound is
# already established and the depth has to be searched again regardless, so the remaining
# root moves look like pure waste: an audit measured 5.15% of all nodes spent after the
# condition already held.
#
# Off by default, because measuring it says otherwise. At fixed depth 11 over eight
# positions the break cost 1,216,532 nodes against 1,152,282, 5.6% MORE work to reach the
# same depth, and changed the move chosen in three of the eight. The moves searched after
# a fail-high are not wasted: they fill the transposition table and the history tables
# that the widened re-search of that same depth then relies on. Set CHESS_ROOT_FAIL_HIGH
# to 1 to A/B it under a real time control before believing either number.
ROOT_FAIL_HIGH_BREAK = _margin("ROOT_FAIL_HIGH", 0)


# A depth whose best move changed, or whose score fell by this much, is worth more time.
INSTABILITY_DROP = 30
INSTABILITY_FACTOR = 1.5

# The other side of the same coin: a best move that has survived several iterations is
# settled, so hand the unspent time to a later move that needs it. Without this the engine
# spends its whole allowance on every position, including the easy ones.
STABLE_STEP = 0.075
STABLE_FLOOR = 0.70
# Contracting on two or three trivial early iterations says nothing, so the counter only
# starts here.
STABLE_MIN_DEPTH = 5

# An aspiration re-search re-runs a whole depth, so a failed window is bounded by this
# multiple of the soft limit rather than by the hard limit. It sits above the instability
# stretch and below the hard limit, so an unsettled position keeps its extension.
STRETCH_MULTIPLE = 2.0

# How much longer the next iteration is expected to take than the last one. Measured from
# the two most recent iterations where there are two, and clamped, so that one anomalous
# depth cannot poison the estimate.
GROWTH_DEFAULT = 2.2
GROWTH_MIN = 1.6
GROWTH_MAX = 4.0

# How often to read the clock. Must be a power of two minus one when used as a mask.
CLOCK_INTERVAL_MASK = 2047

RESERVE_MS = 300.0

# Stockfish's optScale. The share of the remaining clock spent on one move, rising with the
# ply. A flat share of a draining clock falls away monotonically, which spends the most on
# the opening and the least on the middlegame; letting the share rise roughly cancels the
# fall and keeps the allocation level. Refitting these steeper was tried and was worse.
SOFT_BASE = 0.0084
SOFT_PLY_SCALE = 0.0042
# Caps. A single move may never take more than a twelfth of the clock, and the hard limit
# is the backstop one level above the stretch.
SOFT_CAP_DIVISOR = 12.0
HARD_DIVISOR = 6.0
HARD_MULTIPLE = 3.0

# Most valuable victim, least valuable attacker. Indexed [victim][attacker].
MVV_LVA = np.zeros((6, 6), dtype=np.int32)
for _victim in range(6):
    for _attacker in range(6):
        MVV_LVA[_victim, _attacker] = 100 * (_victim + 1) - _attacker

# Ordering scores. Captures and killers sit above every quiet move, and history fills the
# space below, so the bands can never cross.
SCORE_TT = 1 << 24
SCORE_GOOD_CAPTURE = 1 << 22
SCORE_KILLER_1 = (1 << 21) + 2
SCORE_KILLER_2 = (1 << 21) + 1
SCORE_COUNTER = 1 << 21
SCORE_BAD_CAPTURE = -(1 << 22)
HISTORY_MAX = 1 << 14

# Late move reductions. Later moves in a well-ordered list are progressively less likely
# to be best, so they are searched shallower first and only re-searched if they beat alpha.
LMR_TABLE = np.zeros((64, 64), dtype=np.int64)
for _depth in range(1, 64):
    for _index in range(1, 64):
        _reduction = int(0.75 + math.log(_depth) * math.log(_index) / 2.25)
        # Never reduce below one ply of real search, or the value is meaningless.
        LMR_TABLE[_depth, _index] = max(0, min(_reduction, _depth - 1))


class Work(NamedTuple):
    """Every mutable array the search touches. Passed, never global."""

    state: np.ndarray
    mail: np.ndarray
    moves: np.ndarray
    scores: np.ndarray
    table: np.ndarray
    killers: np.ndarray
    history: np.ndarray
    counter: np.ndarray
    ints: np.ndarray
    floats: np.ndarray
    hist_keys: np.ndarray
    static_evals: np.ndarray
    played: np.ndarray
    quiets: np.ndarray
    probe: np.ndarray
    moved_piece: np.ndarray
    cont_hist: np.ndarray
    acc: np.ndarray


def new_work(table: np.ndarray = TT) -> Work:
    state, mail = new_stacks()
    work = Work(
        state=state,
        mail=mail,
        moves=np.zeros(STACK_PLIES * MAX_MOVES, dtype=np.int32),
        scores=np.zeros(STACK_PLIES * MAX_MOVES, dtype=np.int32),
        table=table,
        killers=np.zeros((STACK_PLIES, 2), dtype=np.int32),
        history=np.zeros((2, 64, 64), dtype=np.int32),
        counter=np.zeros((2, 64, 64), dtype=np.int32),
        ints=np.zeros(INT_SLOTS, dtype=np.int64),
        floats=np.zeros(FLOAT_SLOTS, dtype=np.float64),
        hist_keys=np.zeros(2048, dtype=np.uint64),
        static_evals=np.zeros(STACK_PLIES, dtype=np.int32),
        played=np.zeros(STACK_PLIES, dtype=np.int32),
        # The quiet moves each ply actually searched, in the order they were tried.
        quiets=np.zeros(STACK_PLIES * MAX_MOVES, dtype=np.int32),
        # Scratch for the legal-move existence probe. One ply deep: the probe always
        # finishes before the node that asked it generates anything of its own.
        probe=np.zeros(MAX_MOVES, dtype=np.int32),
        moved_piece=np.zeros(STACK_PLIES, dtype=np.int8),
        # [distance][piece][to][piece][to]: how a reply fared after a given earlier move.
        # Two distances, one and two plies back, at about 1.2 MB in total.
        cont_hist=np.zeros((2, 6, 64, 6, 64), dtype=np.int32),
        # The network's per-ply accumulator. C-contiguous so acc[ply, p] vectorises.
        acc=new_accumulator(STACK_PLIES),
    )
    # Every mechanism on. A Work whose switches defaulted to zero would search plain
    # alpha-beta and look like a bug somewhere else entirely.
    work.ints[I_MECHANISMS] = ALL_MECHANISMS
    return work


WORK = new_work()


@njit(cache=False)
def check_time(work: Bits) -> None:
    """Abort the search when the hard limit or the node limit is reached.

    numba cannot call time.time() in nopython mode, so the clock is read through objmode.
    Doing that every node would cost more than the search; every 2048 nodes it is free.
    """
    if work.ints[I_NODES] & CLOCK_INTERVAL_MASK != 0:
        return
    limit = work.ints[I_NODE_LIMIT]
    if limit != 0 and work.ints[I_NODES] >= limit:
        work.ints[I_ABORT] = 1
        return
    with objmode(now="f8"):
        now = time.time()
    if now >= work.floats[F_HARD]:
        work.ints[I_ABORT] = 1


@njit(cache=False)
def read_clock() -> float:
    """Wall clock seconds, readable from jitted code.

    This replaces a `past_soft_limit` predicate. Every deadline in `search_root` is now
    compared against the same reading, so one objmode call serves the soft limit, the
    stretch limit and the iteration timing, and no extra jitted function is compiled. The
    import budget is the binding constraint on this engine, so that matters.
    """
    with objmode(now="f8"):
        now = time.time()
    return now


@njit(cache=False)
def is_repetition(work: Bits, ply: Square) -> Flag:
    """A position seen before, either in this line or earlier in the real game.

    One repetition scores as a draw. Waiting for a third occurrence inside the search
    loses far more than the strict reading gains, because the opponent can always decline.
    """
    key = work.state[ply][KEY]
    half = np.int64(work.state[ply][HALF])
    back = 0
    node = ply - 2
    while node >= 0 and back < half:
        if work.state[node][KEY] == key:
            return True
        node -= 2
        back += 2
    # agent.py builds the game history as [us, them, us, ..., us], so the root sits at
    # the last index and entries alternate side to move from there. A position can only
    # repeat one with the same side to move, so which end of that alternation to start
    # from depends on the parity of `ply`: an even ply has the root's side to move and
    # matches index length-3, an odd ply matches length-2. Walking from length-2 for
    # every ply, as this did, scans one parity only, and so could never find a repetition
    # of a position from earlier in the game where it is our own turn, which is exactly
    # the shape a threefold takes. The root itself is already covered by the loop above,
    # through work.state[0], and is skipped here rather than tested twice.
    length = work.ints[I_HIST_LEN]
    index = length - 3 + (ply & 1)
    # Distance in plies from this node back to a history entry, bounded by the halfmove
    # clock: an irreversible move makes everything before it unreachable.
    while index >= 0 and (length - 1 - index) + ply <= half:
        if work.hist_keys[index] == key:
            return True
        index -= 2
    return False


@njit(cache=False)
def continuation_score(work: Bits, ply: Square, piece: Square, to: Square) -> Square:
    """How this reply has fared after the moves that led here."""
    total = 0
    for distance in range(2):
        previous = ply - 1 - distance
        if previous < 0 or work.played[previous] == 0:
            continue
        prior_piece = np.int64(work.moved_piece[previous])
        if prior_piece < 0:
            continue
        prior_to = np.int64((work.played[previous] >> 6) & 63)
        total += work.cont_hist[distance, prior_piece, prior_to, piece, to]
    return total


@njit(cache=False, inline="always")
def is_quiet(mail: Bits, move: Bits) -> Flag:
    """One definition of a quiet move, shared by ordering, history and the malus list.

    En passant and promotions win material without landing on an occupied square, so a
    bare `mail[to] < 0` calls them quiet and feeds them to the quiet history tables.
    Castling has no victim and genuinely is quiet.
    """
    flag = np.int64((move >> 15) & 3)
    if flag == FLAG_EP:
        return False
    if flag == FLAG_PROMO:
        return False
    return mail[np.int64((move >> 6) & 63)] < 0


@njit(cache=False)
def score_moves(work: Bits, ply: Square, base: Square, count: Square, tt_move: Bits) -> None:
    """Assign an ordering score to each generated move. Never sorts the whole list."""
    state = work.state[ply]
    mail = work.mail[ply]
    black = np.int64(state[STM])
    previous = work.played[ply - 1] if ply > 0 else np.int32(0)
    counter_move = np.int32(0)
    if previous != 0:
        counter_move = work.counter[black, previous & 63, (previous >> 6) & 63]

    for i in range(base, count):
        move = work.moves[i]
        if move == tt_move and tt_move != 0:
            work.scores[i] = SCORE_TT
            continue
        frm = np.int64(move & 63)
        to = np.int64((move >> 6) & 63)
        victim = np.int64(mail[to])
        # Promotions are ordered by their SEE like captures. Left out, a quiet promotion
        # falls through to quiet history, which knows nothing about the queen it wins.
        if not is_quiet(mail, move):
            attacker = np.int64(mail[frm])
            gain = see(state, mail, move)
            base_score = MVV_LVA[victim if victim >= 0 else 0, attacker]
            if gain >= 0:
                work.scores[i] = SCORE_GOOD_CAPTURE + base_score
            else:
                work.scores[i] = SCORE_BAD_CAPTURE + base_score
        elif move == work.killers[ply, 0]:
            work.scores[i] = SCORE_KILLER_1
        elif move == work.killers[ply, 1]:
            work.scores[i] = SCORE_KILLER_2
        elif move == counter_move and counter_move != 0:
            work.scores[i] = SCORE_COUNTER
        else:
            piece = np.int64(mail[frm])
            work.scores[i] = work.history[black, frm, to] + continuation_score(
                work, ply, piece, to
            )


@njit(cache=False)
def pick_move(work: Bits, index: Square, count: Square) -> Square:
    """Selection sort one step: swap the best remaining move into `index`.

    Cheaper than sorting the list, because a beta cutoff usually happens in the first few
    moves and the rest are never looked at.
    """
    best = index
    for i in range(index + 1, count):
        if work.scores[i] > work.scores[best]:
            best = i
    if best != index:
        work.moves[index], work.moves[best] = work.moves[best], work.moves[index]
        work.scores[index], work.scores[best] = work.scores[best], work.scores[index]
    return work.moves[index]


@njit(cache=False)
def update_continuation(
    work: Bits, ply: Square, piece: Square, to: Square, bonus: Square
) -> None:
    if piece < 0:
        return
    for distance in range(2):
        previous = ply - 1 - distance
        if previous < 0 or work.played[previous] == 0:
            continue
        prior_piece = np.int64(work.moved_piece[previous])
        if prior_piece < 0:
            continue
        prior_to = np.int64((work.played[previous] >> 6) & 63)
        entry = work.cont_hist[distance, prior_piece, prior_to, piece, to] + bonus
        if entry > HISTORY_MAX:
            entry = HISTORY_MAX
        elif entry < -HISTORY_MAX:
            entry = -HISTORY_MAX
        work.cont_hist[distance, prior_piece, prior_to, piece, to] = entry


@njit(cache=False)
def update_history(work: Bits, ply: Square, best_move: Bits, depth: Square, base: Square,
                   quiet_count: Square) -> None:
    """Reward the move that caused a cutoff and punish the quiets that did not.

    The malus walks `work.quiets`, which holds only the quiet moves this node actually
    searched. Walking the selected move list instead punishes captures, moves rejected
    for king safety, and moves skipped by pruning, none of which were ever tried.
    """
    state = work.state[ply]
    black = np.int64(state[STM])
    bonus = np.int32(min(depth * depth, 400))

    if work.killers[ply, 0] != best_move:
        work.killers[ply, 1] = work.killers[ply, 0]
        work.killers[ply, 0] = best_move

    frm = np.int64(best_move & 63)
    to = np.int64((best_move >> 6) & 63)
    piece = np.int64(work.mail[ply][frm])
    work.history[black, frm, to] += bonus
    update_continuation(work, ply, piece, to, bonus)
    if work.history[black, frm, to] > HISTORY_MAX:
        for a in range(64):
            for b in range(64):
                work.history[black, a, b] //= 2

    previous = work.played[ply - 1] if ply > 0 else np.int32(0)
    if previous != 0:
        work.counter[black, previous & 63, (previous >> 6) & 63] = best_move

    for i in range(base, base + quiet_count):
        move = work.quiets[i]
        if move == best_move:
            continue
        move_from = np.int64(move & 63)
        move_to = np.int64((move >> 6) & 63)
        work.history[black, move_from, move_to] -= bonus
        update_continuation(work, ply, np.int64(work.mail[ply][move_from]), move_to, -bonus)


@njit(cache=False, inline="always")
def rule50_context(state: Bits) -> Square:
    """One bit: is this node inside the band where the halfmove clock changes scores?

    A function of the clock alone, so that an entry written by one search and read by
    another are comparing the same thing. The depth-aware test below is separate, and is
    for pruning, where the depth of the search being replaced is known.
    """
    return 1 if np.int64(state[HALF]) + RULE50_BAND >= RULE50_LIMIT else 0


@njit(cache=False, inline="always")
def near_fifty_move(state: Bits, depth: Square) -> Flag:
    """Can a search of this depth from this node run into the fifty-move draw?

    The band is the floor. A null move search reaches further than the band on its own,
    because `make_null` zeroes the clock under it and it cannot see the draw at all, so
    the remaining depth widens the test when it is the larger of the two.
    """
    horizon = depth if depth > RULE50_BAND else RULE50_BAND
    return np.int64(state[HALF]) + horizon >= RULE50_LIMIT


@njit(cache=False, inline="always")
def any_legal_move(work: Bits, ply: Square, state: Bits, mail: Bits) -> Flag:
    """Does the side to move have a legal move? Exact, and cheap when the answer is yes.

    Probing for stalemate by generating and filtering every move at each quiet leaf
    measured 837 knps against 748, an 11% loss, which is what a piece-count gate used to
    buy back. The gate was an assumption about where stalemate happens rather than a
    proof: a valid stalemate with four pieces on the side to move walked straight past it
    and was scored as the static evaluation of a lost position.

    `has_free_move` replaces the assumption with a sound one-sided test that answers
    almost every node without making a move at all, and the exact generator runs only
    where that test is undecided. Nothing is traded away: the answer is the same answer
    `has_legal_move` alone would give.

    The caller must have established that the side to move is not in check.
    """
    return has_free_move(state) or has_legal_move(
        state, mail, work.state[ply + 1], work.mail[ply + 1], work.probe
    )


@njit(cache=False)
def qsearch(work: Bits, ply: Square, alpha: Bits, beta: Bits) -> Bits:
    """Search captures until the position is quiet, so the evaluation is not measured
    halfway through an exchange.

    A node is counted by whoever enters it, once. negamax has already counted the position
    it hands over here, whether through the depth floor or through razoring, so counting
    it again on the way in would count one position twice: an audit found 18.6% of all
    reported nodes were these duplicates, which silently made every node-budgeted
    comparison a different experiment from the one it claimed to be.
    """
    check_time(work)
    if work.ints[I_ABORT] != 0:
        return np.int32(0)
    if ply >= STACK_PLIES - 2:
        return evaluate(work.acc, ply, work.state[ply])
    if ply > work.ints[I_SELDEPTH]:
        work.ints[I_SELDEPTH] = ply

    state = work.state[ply]
    mail = work.mail[ply]
    base = ply * MAX_MOVES
    checked = in_check(state)

    # Terminal adjudication, before the static evaluation is allowed to speak for the
    # position. negamax checks these on the way in, which protects the node qsearch was
    # entered at and none of the ones it reaches afterwards. Mate outranks every draw
    # rule, so a drawn-by-clock position is only a draw if the side to move can move.
    if state[HALF] >= 100 or insufficient_material(state) or (
        # A position cannot repeat in fewer than four plies, and the halfmove clock is
        # reset by every irreversible move, so this skips the history walk at most nodes.
        state[HALF] >= 4 and is_repetition(work, ply)
    ):
        if not checked or has_legal_move(
            state, mail, work.state[ply + 1], work.mail[ply + 1], work.probe
        ):
            return np.int32(0)
    elif not checked and not any_legal_move(work, ply, state, mail):
        # Stalemate. Reached here before the static evaluation is allowed to speak, so a
        # quiescence leaf cannot report the material on the board as the score of a
        # position that is over.
        return np.int32(0)

    if checked:
        # Standing pat while in check would claim a score the side to move cannot
        # actually hold, so every evasion has to be searched.
        stand_pat = np.int32(-INF)
        best = np.int32(-INF)
        count = generate(state, work.moves, base)
    else:
        stand_pat = evaluate(work.acc, ply, state)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        best = stand_pat
        count = generate_captures(state, work.moves, base)

    score_moves(work, ply, base, count, np.int32(0))

    black = np.int64(state[STM])
    legal = 0
    for index in range(base, count):
        move = pick_move(work, index, count)
        if not checked and work.ints[I_NO_PRUNING] == 0:
            # A capture that loses material cannot rescue a position this far behind.
            # The bound has to include everything the move wins, or it discards moves on
            # the strength of a gain it never counted: a queening pawn arrives on an empty
            # square and would otherwise be bounded by the 100 cp default.
            victim = np.int64(mail[(move >> 6) & 63])
            flag = np.int64((move >> 15) & 3)
            gain = SEE_VALUE[victim] if victim >= 0 else np.int32(0)
            if flag == FLAG_EP:
                gain = SEE_VALUE[PAWN]
            elif flag == FLAG_PROMO:
                gain += SEE_VALUE[(move >> 12) & 7] - SEE_VALUE[PAWN]
            if stand_pat + gain + DELTA_MARGIN < alpha:
                continue
            if see(state, mail, move) < 0:
                continue
        make(state, mail, work.state[ply + 1], work.mail[ply + 1], move)
        if not legal_after(work.state[ply + 1], black):
            continue
        nnue_apply(work.acc, ply, state, mail, move)
        legal += 1
        work.played[ply] = move
        work.moved_piece[ply] = mail[np.int64(move & 63)]
        work.ints[I_NODES] += 1
        value = -qsearch(work, ply + 1, -beta, -alpha)
        if work.ints[I_ABORT] != 0:
            return np.int32(0)
        if value > best:
            best = value
            if value > alpha:
                alpha = value
                if alpha >= beta:
                    break

    if checked and legal == 0:
        return np.int32(-MATE + ply)
    return np.int32(best)


@njit(cache=False)
def negamax(
    work: Bits, ply: Square, depth: Square, alpha: Bits, beta: Bits, is_pv: Flag,
    can_null: Flag = True,
) -> Bits:
    """Principal variation search."""
    work.ints[I_NODES] += 1
    check_time(work)
    if work.ints[I_ABORT] != 0:
        return np.int32(0)

    if ply >= STACK_PLIES - 4:
        return evaluate(work.acc, ply, work.state[ply])

    state = work.state[ply]
    mail = work.mail[ply]
    checked = in_check(state)

    if ply > 0:
        # Checkmate ends the game before any of these do, so a mating position at
        # halfmove 100 is a mate and not a draw.
        if (
            state[HALF] >= 100 or insufficient_material(state) or is_repetition(work, ply)
        ) and (
            not checked
            or has_legal_move(
                state, mail, work.state[ply + 1], work.mail[ply + 1], work.probe
            )
        ):
            return np.int32(0)
        # Mate distance pruning: a mate found elsewhere is already nearer than anything
        # this subtree can produce, so there is nothing left to look for.
        alpha = max(alpha, np.int32(-MATE + ply))
        beta = min(beta, np.int32(MATE - ply - 1))
        if alpha >= beta:
            return np.int32(alpha)

    # Check extension. Disabled with pruning off so the oracle compares like with like.
    if checked and work.ints[I_NO_PRUNING] == 0:
        depth += 1

    if depth <= 0:
        return qsearch(work, ply, alpha, beta)

    key = state[KEY]
    hit, tt_score, tt_move, tt_depth, tt_bound, tt_static, tt_rule50 = tt_probe(
        work.table, key, ply
    )
    context = rule50_context(state)
    # The move and the static evaluation are properties of the position and are used
    # whatever the clock said when they were written. The score is not: it was searched
    # under one rule-clock context and may only be reused under the same one, and inside
    # the band "the same one" is not good enough, because two clocks in there are two
    # different distances from the draw. So the score is reused only when the entry and
    # this node both sit outside the band. See RULE50_BAND.
    if (
        hit
        and not is_pv
        and tt_depth >= depth
        and work.ints[I_NO_PRUNING] == 0
        and tt_rule50 == 0
        and context == 0
    ):
        if tt_bound == BOUND_EXACT:
            return tt_score
        if tt_bound == BOUND_LOWER and tt_score >= beta:
            return tt_score
        if tt_bound == BOUND_UPPER and tt_score <= alpha:
            return tt_score

    static = tt_static if hit and tt_static != 0 else evaluate(work.acc, ply, state)
    work.static_evals[ply] = static

    black = np.int64(state[STM])
    # Each mechanism's own switch is the last clause of its condition, not the first, so
    # that the test costs nothing at a node the mechanism was not going to fire on.
    mechanisms = work.ints[I_MECHANISMS]
    # Every pruning below returns or skips a score without searching for the draw the
    # halfmove clock is about to force, so near the threshold none of them may run.
    near_fifty = near_fifty_move(state, depth)
    prunable = (
        work.ints[I_NO_PRUNING] == 0
        and not is_pv
        and not checked
        and beta < MATE_IN_MAX
        and beta > -MATE_IN_MAX
        and not near_fifty
    )

    if prunable:
        # Reverse futility. So far ahead that giving back a margin per remaining ply still
        # beats beta, so the opponent would have avoided this line.
        if depth <= 8 and static - RFP_MARGIN * depth >= beta and mechanisms & M_RFP:
            # A stalemate scores zero however good the pieces on it look, so a cutoff that
            # stands in for the search has to know that the search had something to do.
            # `prunable` has already established that the side to move is not in check.
            if not any_legal_move(work, ply, state, mail):
                return np.int32(0)
            work.ints[C_RFP] += 1
            return np.int32(static)

        # Razoring. So far behind that only a capture sequence could rescue it.
        if depth <= 3 and static + RAZOR_MARGIN * depth < alpha and mechanisms & M_RAZOR:
            razor = qsearch(work, ply, alpha, beta)
            if razor <= alpha:
                work.ints[C_RAZOR] += 1
                return razor

        # Null move. Skipping a turn and still failing high means the real move will too.
        # Not tried without a piece on the board: a side with only pawns can be in
        # zugzwang, where passing is better than every legal move.
        if (
            can_null
            and depth >= 3
            and static >= beta
            and has_non_pawn_material(state, black)
            and mechanisms & M_NULL
        ):
            # Passing the move is only meaningful where there was a move to make.
            if not any_legal_move(work, ply, state, mail):
                return np.int32(0)
            reduction = 3 + depth // 4 + min((static - beta) // 200, 3)
            make_null(state, mail, work.state[ply + 1], work.mail[ply + 1])
            nnue_copy(work.acc, ply)
            work.played[ply] = np.int32(0)
            work.moved_piece[ply] = np.int8(-1)
            null_value = -negamax(
                work, ply + 1, depth - reduction - 1, -beta, np.int32(-beta + 1), False, False
            )
            if work.ints[I_ABORT] != 0:
                return np.int32(0)
            if null_value >= beta:
                # A mate score proved by passing is not a real mate.
                if null_value >= MATE_IN_MAX:
                    null_value = beta
                if depth < 10:
                    work.ints[C_NULL] += 1
                    return np.int32(null_value)
                # Deep enough that zugzwang is worth ruling out explicitly.
                verify = negamax(
                    work, ply, depth - reduction - 1, np.int32(beta - 1), beta, False, False
                )
                if work.ints[I_ABORT] != 0:
                    return np.int32(0)
                if verify >= beta:
                    work.ints[C_NULL] += 1
                    return np.int32(null_value)

    base = ply * MAX_MOVES
    count = generate(state, work.moves, base)
    score_moves(work, ply, base, count, tt_move if hit else np.int32(0))

    best = np.int32(-INF)
    best_move = np.int32(0)
    original_alpha = alpha
    legal = 0

    # Internal iterative reduction: with no TT move the ordering is poor, so a full-depth
    # search here is mostly wasted. Search shallower and let the TT move guide the retry.
    if (
        work.ints[I_NO_PRUNING] == 0
        and depth >= 4
        and not (hit and tt_move != 0)
        and work.ints[I_MECHANISMS] & M_IIR
    ):
        work.ints[C_IIR] += 1
        depth -= 1

    quiets_tried = 0
    quiets_searched = 0
    improving = ply < 2 or static > work.static_evals[ply - 2]

    for index in range(base, count):
        move = pick_move(work, index, count)
        to = np.int64((move >> 6) & 63)
        is_capture = mail[to] >= 0 or np.int64((move >> 15) & 3) != 0
        move_score = work.scores[index]

        if (
            work.ints[I_NO_PRUNING] == 0
            and not is_pv
            and not checked
            and legal > 0
            and best > -MATE_IN_MAX
            and not near_fifty
        ):
            if not is_capture:
                # Late move pruning: this far down a well-ordered list, at low depth,
                # a quiet move is not going to be the best one.
                cap = 3 + depth * depth
                if not improving:
                    cap //= 2
                if depth <= 8 and quiets_tried >= cap and mechanisms & M_LMP:
                    work.ints[C_LMP] += 1
                    continue
                # Futility: too far below alpha for a quiet move to close the gap.
                if (
                    depth <= 6
                    and static + FUTILITY_BASE + FUTILITY_MARGIN * depth <= alpha
                    and mechanisms & M_FUTILITY
                ):
                    work.ints[C_FUTILITY] += 1
                    continue
                if (
                    depth <= 8
                    and see(state, mail, move) < -SEE_QUIET_MARGIN * depth
                    and mechanisms & M_SEE
                ):
                    work.ints[C_SEE] += 1
                    continue
            elif (
                depth <= 8
                and see(state, mail, move) < -SEE_CAPTURE_MARGIN * depth
                and mechanisms & M_SEE
            ):
                work.ints[C_SEE] += 1
                continue

        make(state, mail, work.state[ply + 1], work.mail[ply + 1], move)
        if not legal_after(work.state[ply + 1], black):
            continue
        nnue_apply(work.acc, ply, state, mail, move)
        legal += 1
        work.played[ply] = move
        work.moved_piece[ply] = mail[np.int64(move & 63)]
        if not is_capture:
            quiets_tried += 1
        if is_quiet(mail, move):
            work.quiets[base + quiets_searched] = move
            quiets_searched += 1

        if legal == 1:
            value = -negamax(work, ply + 1, depth - 1, -beta, -alpha, is_pv)
        else:
            reduction = 0
            if (
                work.ints[I_NO_PRUNING] == 0
                and depth >= 3
                and legal >= 3
                and not is_capture
                and mechanisms & M_LMR
            ):
                reduction = LMR_TABLE[min(depth, 63), min(legal, 63)]
                if is_pv:
                    reduction -= 1
                if move_score >= SCORE_COUNTER:
                    reduction -= 1
                if not improving:
                    reduction += 1
                if reduction < 0:
                    reduction = 0
                if reduction > depth - 2:
                    reduction = depth - 2 if depth >= 2 else 0

            if reduction > 0:
                work.ints[C_LMR] += 1
            value = -negamax(
                work, ply + 1, depth - 1 - reduction, -alpha - 1, -alpha, False
            )
            # A reduced search that beat alpha proves nothing until it is repeated at
            # full depth. Skipping this re-search is how an engine looks fine in tests
            # and quietly plays bad moves.
            if reduction > 0 and value > alpha:
                value = -negamax(work, ply + 1, depth - 1, -alpha - 1, -alpha, False)
            if alpha < value < beta:
                value = -negamax(work, ply + 1, depth - 1, -beta, -alpha, is_pv)

        if work.ints[I_ABORT] != 0:
            return np.int32(0)

        if value > best:
            best = value
            best_move = move
            if value > alpha:
                alpha = value
                if alpha >= beta:
                    if is_quiet(mail, move):
                        update_history(work, ply, move, depth, base, quiets_searched)
                    break

    if legal == 0:
        return np.int32(-MATE + ply) if checked else np.int32(0)

    bound = BOUND_EXACT
    if best <= original_alpha:
        bound = BOUND_UPPER
    elif best >= beta:
        bound = BOUND_LOWER
    tt_store(
        work.table, key, ply, np.int32(best), best_move, depth, bound, np.int32(static),
        work.ints[I_AGE], context,
    )
    return np.int32(best)


@njit(cache=False)
def search_root(work: Bits, max_depth: Square) -> Bits:
    """Iterative deepening with aspiration windows. Returns the best move found."""
    best_move = np.int32(0)
    best_score = np.int32(0)
    previous_move = np.int32(0)
    previous_score = np.int32(0)
    base = 0
    state = work.state[0]
    mail = work.mail[0]
    black = np.int64(state[STM])

    # `improving` at ply 2 asks whether the side to move has bettered its evaluation of
    # two plies back, which at ply 2 is the root. negamax never runs at ply 0 from here,
    # so nothing else ever writes this slot: left alone it holds whatever the last search
    # put there, and in real play it is never written at all, which quietly turns the test
    # into "is the static evaluation positive" and changes the reductions that follow.
    work.static_evals[0] = evaluate(work.acc, 0, state)

    count = generate(state, work.moves, base)
    # Establish a legal move before anything is allowed to abort.
    for index in range(base, count):
        move = work.moves[index]
        make(state, mail, work.state[1], work.mail[1], move)
        if legal_after(work.state[1], black):
            best_move = move
            break
    if best_move == 0:
        return np.int32(0)

    growth = GROWTH_DEFAULT
    for depth in range(1, max_depth + 1):
        iteration_start = read_clock()
        window = np.int32(18)
        # A failed window re-runs the whole depth. The first failure widens by four, the
        # second goes straight to full width, so a depth costs at most three passes and the
        # last of them cannot fail. Escalating by four indefinitely took three or four
        # passes to reach full width from a window of 18, each one a complete search.
        widenings = 0
        abandoned = False
        if depth >= 5:
            low = best_score - window
            high = best_score + window
            alpha = np.int32(low if low > -INF else -INF)
            beta = np.int32(high if high < INF else INF)
        else:
            alpha = np.int32(-INF)
            beta = np.int32(INF)

        while True:
            score = np.int32(-INF)
            iteration_move = np.int32(0)
            count = generate(state, work.moves, base)
            score_moves(work, 0, base, count, best_move)
            legal = 0
            local_alpha = alpha
            for index in range(base, count):
                move = pick_move(work, index, count)
                make(state, mail, work.state[1], work.mail[1], move)
                if not legal_after(work.state[1], black):
                    continue
                nnue_apply(work.acc, 0, state, mail, move)
                legal += 1
                work.played[0] = move
                # Continuation history below the root reads this slot. Left at zero it
                # attributes every root move to a pawn.
                work.moved_piece[0] = mail[np.int64(move & 63)]
                if legal == 1:
                    value = -negamax(work, 1, depth - 1, -beta, -local_alpha, True)
                else:
                    value = -negamax(work, 1, depth - 1, -local_alpha - 1, -local_alpha, False)
                    if local_alpha < value < beta:
                        value = -negamax(work, 1, depth - 1, -beta, -local_alpha, True)
                if work.ints[I_ABORT] != 0:
                    break
                # An iteration whose cost was underestimated is otherwise stopped only by
                # the hard limit, which is how a move still reached 3x its soft limit after
                # the checks between depths were added. The first root move is the previous
                # best, so once one has finished there is always a move to fall back on.
                if legal >= 1 and read_clock() >= work.floats[F_STRETCH]:
                    work.ints[I_ABORT] = 1
                    break
                if value > score:
                    score = value
                    iteration_move = move
                    if value > local_alpha:
                        local_alpha = value
                        if ROOT_FAIL_HIGH_BREAK and local_alpha >= beta:
                            # Failed high. See ROOT_FAIL_HIGH_BREAK: this looks like free
                            # work to skip and measured as the opposite.
                            break

            if work.ints[I_ABORT] != 0:
                break
            if score <= alpha:
                # Failed low: widen downwards and try this depth again. On a fail-low every
                # root score is an upper bound, so their ordering is unreliable and the
                # partial result must not be committed; abandoning keeps the last depth's
                # move instead.
                if read_clock() >= work.floats[F_STRETCH]:
                    abandoned = True
                    break
                widenings += 1
                if widenings >= 2:
                    alpha = np.int32(-INF)
                else:
                    window *= 4
                    low = score - window
                    alpha = np.int32(low if low > -INF else -INF)
                continue
            if score >= beta:
                if read_clock() >= work.floats[F_STRETCH]:
                    abandoned = True
                    break
                widenings += 1
                if widenings >= 2:
                    beta = np.int32(INF)
                else:
                    window *= 4
                    high = score + window
                    beta = np.int32(high if high < INF else INF)
                continue
            best_score = score
            if iteration_move != 0:
                best_move = iteration_move
            break

        if work.ints[I_ABORT] != 0 or abandoned:
            break
        # This depth finished and its result was committed, so it is the one to report.
        work.ints[I_DEPTH] = depth
        work.ints[I_SCORE] = best_score
        # A forced mate is found; searching deeper cannot improve on it.
        if best_score >= MATE_IN_MAX or best_score <= -MATE_IN_MAX:
            break

        now = read_clock()
        work.floats[F_PREV_ITER] = work.floats[F_LAST_ITER]
        work.floats[F_LAST_ITER] = now - iteration_start

        # Stability, in both directions. A changed best move or a falling score means this
        # position is not settled and is worth more time; a move that has survived several
        # iterations is settled and the unspent time is worth more to a later move. The
        # soft limit is derived from the base span each iteration rather than accumulated,
        # because a limit that is only ever written upwards cannot contract.
        unsettled = depth >= 4 and (
            best_move != previous_move or best_score < previous_score - INSTABILITY_DROP
        )
        if not unsettled and depth >= STABLE_MIN_DEPTH:
            work.ints[I_STABLE] += 1
        else:
            work.ints[I_STABLE] = 0
        if unsettled:
            scale = INSTABILITY_FACTOR
        else:
            scale = 1.0 - STABLE_STEP * work.ints[I_STABLE]
            if scale < STABLE_FLOOR:
                scale = STABLE_FLOOR
        span = work.floats[F_SOFT_SPAN]
        soft = work.floats[F_START] + span * scale
        if soft > work.floats[F_HARD]:
            soft = work.floats[F_HARD]
        work.floats[F_SOFT] = soft
        # The prediction below is allowed to aim at the stretch rather than the soft limit.
        # Requiring the next iteration to finish inside the soft limit sounds right and is
        # not: iterations grow by a factor of two to four, so that rule lands between a
        # quarter and all of the budget and throws most of it away. Aiming at twice the
        # limit lands either side of it, which is what a soft limit is supposed to mean.
        stretch = work.floats[F_START] + span * scale * STRETCH_MULTIPLE
        if stretch > work.floats[F_HARD]:
            stretch = work.floats[F_HARD]
        work.floats[F_STRETCH] = stretch

        previous_move = best_move
        previous_score = best_score

        if now >= soft:
            break
        # Do not start an iteration that cannot finish. Checking only whether the limit has
        # already passed lets a depth that begins at 99% of it run to completion however
        # long it takes, which is where the overshoot came from.
        if work.floats[F_PREV_ITER] > 0.0:
            growth = work.floats[F_LAST_ITER] / work.floats[F_PREV_ITER]
            if growth < GROWTH_MIN:
                growth = GROWTH_MIN
            elif growth > GROWTH_MAX:
                growth = GROWTH_MAX
        if now + work.floats[F_LAST_ITER] * growth >= stretch:
            break
    return best_move


def budget_ms(time_left_ms: int, increment_ms: int, ply: int = 0) -> tuple[float, float]:
    """Soft and hard limits in milliseconds.

    The share of the remaining clock rises with the ply, so that a draining clock does not
    drag the allocation down with it. A flat share spent 5.7 s on move one and 1.4 s on
    move forty, which is backwards: move one comes out of a curated opening and move forty
    does not.

    Floored rather than allowed to go negative: late in a long game the base clock is gone
    and play is increment-only, and a negative budget would return no move at all.
    """
    usable = max(float(time_left_ms) - RESERVE_MS, 10.0)
    fraction = SOFT_BASE + math.sqrt(float(ply) + 3.0) * SOFT_PLY_SCALE
    # Only half the increment is credited. Over-crediting it is how engines flag, and a
    # flag costs a whole game, so the asymmetry is deliberate.
    soft = min(usable * fraction + 0.5 * increment_ms, usable / SOFT_CAP_DIVISOR)
    hard = min(usable / HARD_DIVISOR, soft * HARD_MULTIPLE)
    return min(soft, hard), hard


def ply_of(board: chess.Board) -> int:
    """Plies played, from the FEN alone.

    Rated games start from curated positions rather than the standard start, so the
    fullmove number carries real information about how far into the game we are. Clamped
    because a hand-written FEN can say anything.
    """
    ply = (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)
    return min(max(ply, 0), 400)


def set_pruning(enabled: bool, work: Work = WORK) -> None:
    """Turn every heuristic off, so the search can be compared with plain alpha-beta.

    This is the oracle switch, not an experiment: it also removes the check extension and
    the transposition cutoffs. Use `set_mechanisms` to change one thing at a time.
    """
    work.ints[I_NO_PRUNING] = 0 if enabled else 1


def set_mechanisms(names: Iterable[str] | None = None, work: Work = WORK) -> None:
    """Enable exactly these selective mechanisms and no others. None means all of them."""
    if names is None:
        work.ints[I_MECHANISMS] = ALL_MECHANISMS
        return
    mask = 0
    for name in names:
        mask |= 1 << MECHANISMS.index(name)
    work.ints[I_MECHANISMS] = mask


def without_mechanism(name: str, work: Work = WORK) -> None:
    """Everything on except this one. The single-variable form of `set_pruning(False)`."""
    work.ints[I_MECHANISMS] = ALL_MECHANISMS & ~(1 << MECHANISMS.index(name))


def mechanism_counts(work: Work = WORK) -> dict[str, int]:
    """How often each mechanism removed work in the last search."""
    return {name: int(work.ints[I_COUNTS + i]) for i, name in enumerate(MECHANISMS)}


def set_game_history(keys: list[int], work: Work = WORK) -> None:
    """Positions already played in this game, for repetition detection above the root."""
    length = min(len(keys), work.hist_keys.shape[0])
    for i in range(length):
        work.hist_keys[i] = np.uint64(keys[i])
    work.ints[I_HIST_LEN] = length


def clear_tables(work: Work = WORK) -> None:
    work.history[:] = 0
    work.counter[:] = 0
    work.killers[:] = 0
    work.played[:] = 0
    work.moved_piece[:] = 0
    work.cont_hist[:] = 0


def _prepare(
    board: chess.Board, time_left_ms: int, increment_ms: int, node_limit: int, work: Work
) -> None:
    encode(board, work.state[0], work.mail[0])
    # The only full rebuild. Every ply below this is reached incrementally.
    nnue_refresh(work.acc, 0, work.state[0], work.mail[0])
    work.ints[I_COUNTS:I_COUNTS + len(MECHANISMS)] = 0
    work.ints[I_NODES] = 0
    work.ints[I_ABORT] = 0
    work.ints[I_SELDEPTH] = 0
    work.ints[I_NODE_LIMIT] = node_limit
    work.ints[I_AGE] = (int(work.ints[I_AGE]) + 1) & AGE_MASK
    work.ints[I_STABLE] = 0
    # Cleared, not carried. A move that aborts before depth 1 commits would otherwise
    # report the previous move's depth, which is the reading a log is least able to correct.
    work.ints[I_DEPTH] = 0
    work.ints[I_SCORE] = 0
    soft, hard = budget_ms(time_left_ms, increment_ms, ply_of(board))
    now = time.time()
    work.floats[F_START] = now
    work.floats[F_SOFT_SPAN] = soft / 1000.0
    work.floats[F_SOFT] = now + soft / 1000.0
    work.floats[F_HARD] = now + hard / 1000.0
    work.floats[F_STRETCH] = min(now + soft * STRETCH_MULTIPLE / 1000.0, work.floats[F_HARD])
    work.floats[F_LAST_ITER] = 0.0
    work.floats[F_PREV_ITER] = 0.0


def think(
    board: chess.Board,
    time_left_ms: int,
    increment_ms: int = 500,
    node_limit: int = 0,
    max_depth: int = MAX_DEPTH,
    work: Work = WORK,
) -> str:
    """Best move for `board` in UCI, within the clock."""
    _prepare(board, time_left_ms, increment_ms, node_limit, work)
    move = int(search_root(work, min(max_depth, MAX_DEPTH)))
    if move == 0:
        # No legal move exists. The referee ends a game before asking, so this cannot
        # happen in play; raising clearly beats a bare StopIteration out of an iterator.
        raise ValueError(f"no legal move in {board.fen()}")
    return move_to_uci(move)


def search_value(board: chess.Board, depth: int, work: Work = WORK) -> int:
    """Score rather than move, at a fixed depth. Used by the oracle tests."""
    _prepare(board, 3_600_000, 0, 0, work)
    return int(negamax(work, 0, depth, np.int32(-INF), np.int32(INF), True))


def nodes(work: Work = WORK) -> int:
    return int(work.ints[I_NODES])


def last_search(work: Work = WORK) -> tuple[int, int, int, int, float]:
    """What the search just did: depth, seldepth, score in cp, nodes, seconds.

    For `agent.py` to log. A rated game gives back nothing but the moves, so this is the
    only channel through which a live game can say what the engine was thinking.
    """
    return (
        int(work.ints[I_DEPTH]),
        int(work.ints[I_SELDEPTH]),
        int(work.ints[I_SCORE]),
        int(work.ints[I_NODES]),
        time.time() - float(work.floats[F_START]),
    )


def budget_of(work: Work = WORK) -> tuple[float, float]:
    """The soft and hard limits the last search was given, in seconds from its start."""
    start = float(work.floats[F_START])
    return (float(work.floats[F_SOFT]) - start, float(work.floats[F_HARD]) - start)


# Warm every jitted function at import, with the argument types the real calls use.
_board = chess.Board()
_prepare(_board, 1000, 0, 4096, WORK)
search_root(WORK, 2)
qsearch(WORK, 0, np.int32(-INF), np.int32(INF))
any_legal_move(WORK, 0, WORK.state[0], WORK.mail[0])
is_repetition(WORK, 0)
WORK.quiets[0] = WORK.moves[0]
update_history(WORK, 0, np.int32(WORK.moves[0]), 1, 0, 1)
tt_clear(WORK.table)
clear_tables(WORK)
