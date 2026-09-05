# Jitted bitboard search

Design for replacing `search.py` with a numba-jitted bitboard engine.

Date: 2026-09-05

## Goal

Maximise playing strength within the AI Chessathon contract. The current search is a
plain alpha-beta with no ordering, no transposition table and no quiescence, running on
`python-chess` move generation. This design replaces the whole substrate as well as the
algorithm.

## Measured baseline

A throwaway probe built a minimal jitted bitboard movegen (magics for sliders, tables for
leapers, copy-make, pseudo-legal generation with a post-make legality filter) and verified
it against `python-chess`.

| Measurement | Result |
| --- | --- |
| Magic table generation, jitted, self-generated constants | 1.0 s |
| JIT compile of movegen, make and perft | 1.9 s |
| Total import-equivalent cost | 2.9 s of the 90 s budget |
| Jitted `perft(6)` from the start position | 11.7 Mnps |
| `python-chess` `push`/`legal_moves`/`pop` equivalent | 0.275 Mnps |
| Ratio | about 43x |
| `njit` self-recursion with an explicit signature | works |
| `uint64` magic multiply and shift semantics | works |
| `time.time()` under `njit` | unsupported; `objmode` costs 1.8 us per call |

Perft matched on the start position, Kiwipete, the en passant pin position and the
promotion position. The 11.7 Mnps figure is a floor rather than a ceiling: the probe used
a linear-scan `lsb` and a six-way loop for `piece_at`, both of which the real engine
replaces.

Two conclusions follow. The 90 s import budget is not close to binding, so the search can
be large without restructuring. And an `objmode` clock read every 2048 nodes costs about
0.9 ns per node, roughly 0.1% overhead, so time control does not need a cheaper mechanism.

## Platform constraints that shape the design

Taken from `https://aichessathon.com/docs/agent-contract.md` and
`https://aichessathon.com/docs/rules.md`, fetched 2026-09-05.

- One AMD EPYC 9V74 core at 2.60 GHz, 2 GB RAM, no network, no GPU.
- 90 s import budget, then 120 s base plus 0.5 s per move on wall time.
- Python 3.12 with stdlib plus torch, numpy, python-chess, onnxruntime and numba. Nothing
  else installs.
- No native binaries in the zip. Source must be readable by a judge.
- Read-only filesystem apart from 256 MB at `/tmp`, cleared per game.
- The process lives for one game and is suspended while the opponent thinks.
- 50 MB unzipped, and `harness/package.py` ships root `*.py` files only.

## Architecture

### Module layout

All engine modules sit flat at the repo root, because `harness/package.py` globs root
`*.py` and a subpackage directory would silently not ship. None of the names shadow an
importable module, which is the failure mode `CLAUDE.md` warns about.

| File | Contents |
| --- | --- |
| `agent.py` | entrypoint: FEN to state, game history tracking, move to UCI, legality safety net |
| `bitboard.py` | bit primitives, leaper tables, self-generated magics, Zobrist keys |
| `position.py` | state layout, `make`, attack and check detection, SEE |
| `movegen.py` | pseudo-legal generation, staged ordering |
| `tt.py` | transposition table over flat numpy arrays |
| `search.py` | iterative deepening, PVS, quiescence, pruning, time control |
| `evaluate.py` | owned by the user, not built here |

### Position representation

A `uint64` state vector per ply holding six piece bitboards, two colour occupancies, side
to move, castling rights, en passant square, halfmove clock and the Zobrist key, plus an
`int8[64]` mailbox giving O(1) `piece_at`. Both live in preallocated per-ply stacks, so no
allocation happens inside the search.

En passant is stored as square plus one, with zero meaning none, because the state vector
is unsigned and a signed sentinel would compare wrongly. The probe hit exactly this bug.

Copy-make rather than make and unmake. The probe reached 11.7 Mnps while copying state and
scanning six bitboards for `piece_at`, so about 190 bytes per node is demonstrably not the
bottleneck, and copy-make removes an entire class of state-restoration bugs. It also makes
an NNUE accumulator cheap to carry later.

### Evaluation boundary

`evaluate(board: chess.Board)` cannot be called from jitted code, and calling it once per
node would cap the engine near 30 knps, worse than the current agent. The contract becomes:

```python
@njit(int32(uint64[:], int8[:]), cache=False)
def evaluate(state, mailbox) -> int:
    """Centipawns, positive means the side to move is better."""
```

The existing material evaluation is ported to that signature so the engine runs end to
end. Nothing more is built here: evaluation quality is the user's work.

When NNUE arrives, the same seam takes an incrementally updated `int16` accumulator.
Weights train in torch offline, export to a quantised `.npz`, and inference runs in numba.
torch and onnxruntime stay out of the runtime entirely. Their per-call overhead of roughly
20 to 50 us would make them slower than the handcrafted evaluation they replaced.

Known consequence: with a material-only evaluation, extra depth is partly spent on
positions the evaluation cannot distinguish, so arena results will understate the search's
real gain until the evaluation improves. This is a measurement caveat, not a design flaw.

## Move generation

Pseudo-legal generation followed by a post-make legality filter that tests whether the
mover's king is attacked. Simpler and less bug-prone than fully legal generation with pin
masks, and the probe shows the cost is affordable.

Sliders use fancy magic bitboards with magics found at import by a seeded search inside
`njit`, so the constants are self-generated rather than copied. Leapers use precomputed
tables. Move encoding is a single `int32`: `from | to << 6 | promo << 12 | flag << 15`,
where flag is 0 normal, 1 en passant, 2 castle, 3 promotion.

## Search

### Root

Iterative deepening from depth 1 to 127. Aspiration windows from depth 5: plus or minus
18 cp around the previous score, widening asymmetrically on the failing side through 18,
72, 288, then infinity. Root moves are reordered each iteration by the previous
iteration's scores. Depth 1 always completes before aborting is permitted, so a legal move
always exists.

### Interior nodes

Principal Variation Search: full window on the first move, null window on the rest, and a
re-search on fail-high. Node order is mate-distance pruning, TT probe, static evaluation,
whole-node pruning, staged move loop, TT store.

### Transposition table

A single `uint64[N, 2]` array so key and data are adjacent, grouped four to a 64-byte
cache line. One entry packs into 64 bits exactly: `score` i16, `move` u16, `depth` u8,
`bound` and `age` u8, `static eval` i16. Replacement prefers an empty slot, then the same
key, then the lowest `depth - 2 * age_gap`. 128 MB allocated at import, comfortably inside
2 GB.

Mate scores are stored ply-relative and re-based on probe. Skipping this is the classic
source of wrong mate lines.

The table is not cleared between moves. The process lives for one game, so entries from
the previous move are still valid and worth keeping. The age field distinguishes searches.

### Quiescence

Stand-pat with a beta cutoff, captures and queen promotions only, full evasions on the
first ply when in check. Delta pruning, captures with SEE below zero pruned, TT probed at
depth 0.

### Move ordering

Staged and lazy. The full move list is never sorted.

TT move, then good captures by MVV-LVA filtered to SEE at least zero, then killer 1,
killer 2, counter-move, quiets by history, then losing captures.

History is `int32[2][64][64]`, receiving a depth-squared bonus on cutoff, a malus on moves
that failed, and halving on overflow. Continuation history adds one-ply and two-ply tables
indexed by `[piece][to][piece][to]`, about 1.2 MB in total, contributing to both ordering
and reduction decisions. SEE uses the swap algorithm on bitboards and is shared between
ordering and pruning.

### Pruning schedule

| Technique | Condition | Effect |
| --- | --- | --- |
| Reverse futility | `d <= 8`, non-PV, not in check, `eval - 75d >= beta` | return `eval` |
| Null move | `d >= 3`, non-PV, `eval >= beta`, side has non-pawn material | `R = 3 + d/4 + min((eval-beta)/200, 3)`, verification search at `d >= 10` |
| Razoring | `d <= 3`, `eval + 200d < alpha` | drop to quiescence |
| Futility | `d <= 6`, quiet move, `eval + 100 + 90d <= alpha` | skip remaining quiets |
| Late move pruning | `d <= 8`, quiet index above `3 + d*d`, halved when not improving | skip |
| SEE pruning | `d <= 8` | skip quiets below `-50d`, captures below `-100d` |
| Late move reductions | `d >= 3`, move index at least 3 | table `r = 0.75 + ln(d) * ln(m) / 2.25`, adjusted by PV, check, killer, improving and history; full re-search on fail-high |
| Internal iterative reduction | `d >= 4`, no TT move | `d - 1` |
| Check extension | in check | `+1`, capped by total extensions |

### Singular extensions

At a node with a TT move whose TT depth is at least `d - 3`, whose bound is lower or
exact, and whose score is not a mate score, with `d >= 8`: search the remaining moves with
the TT move excluded, at depth `(d - 1) / 2`, in the window
`[ttScore - margin, ttScore - margin + 1]` where `margin = 2 * d`. If that search fails
low, the TT move is singular and gets extended by one ply. If it fails high and
`ttScore >= beta`, return beta as a multi-cut.

Singular extensions and continuation history are the two features here most likely to be
subtly wrong, so each is added and measured separately rather than folded in with the rest
of the pruning schedule.

### Draws and repetition

Fifty-move detection from the halfmove clock, in-tree repetition where a single repeat
inside the search scores as a draw, and insufficient material.

The platform hands over only a FEN per move, but the process survives the game, so
`agent.py` chains positions across calls: it records the Zobrist key of every position it
is asked about and the key after its own reply, which together reconstruct the full
sequence. If an incoming FEN does not chain onto what it remembers, the history resets to
that position alone and the halfmove clock still covers the fifty-move rule.

Draw score is a flat 0. No contempt term in this version.

### Time control

Reserve 300 ms against the referee's 500 ms watchdog grace.

- Soft limit: `(left - reserve) / 22 + 0.75 * increment`.
- Hard limit: `min((left - reserve) / 4, 5 * soft)`.
- A new iteration is not started past the soft limit.
- The hard limit aborts mid-iteration and returns the last completed result.
- Soft extends up to 1.5x when the best move changed at this depth or the score fell.
- The clock is read through `objmode` every 2048 nodes.

Both limits are floored rather than allowed to go negative when `left` is below the
reserve, which happens late in a long game where play degrades to increment-only. The
depth-1 result is always available in that case.

The search also accepts a node limit, used by the A/B harness for deterministic matches.

## Integration

`agent.py` parses the FEN with `python-chess` once per move, roughly 50 us, encodes to the
state vector, updates the game history, calls the search, and decodes the result to UCI.

Before returning, it validates the UCI string against `python-chess` legal moves and falls
back to any legal move if validation fails. This should never trigger. It costs about
50 us and converts a catastrophic loss into a bad move.

## Verification

### Correctness

| Test | What it catches |
| --- | --- |
| `tests/perft.py` | the six standard positions to meaningful depth; the non-negotiable gate, since one illegal move loses a game |
| `tests/fuzz_movegen.py` | our legal move set against `python-chess` at every node of thousands of random walks, finding castling, en passant and promotion interactions that fixed positions miss |
| `tests/make_state.py` | resulting bitboards, rights, en passant, halfmove clock and mailbox against `python-chess` after every generated move |
| `tests/zobrist.py` | the incrementally updated key equals the from-scratch key after every move |
| `tests/see.py` | SEE against hand-computed exchanges plus a reference implementation on random positions |
| `tests/search_invariants.py` | with all pruning disabled the search score equals plain alpha-beta at the same depth; mate-in-N found with the correct ply-adjusted score; never returns an illegal move |

The pruning-disabled equivalence check matters most. Every technique in the pruning
schedule is a heuristic that changes the tree but must not change the value of a
full-width search. Without that oracle, a broken null move implementation just looks like
a slightly worse engine.

### Strength

`harness/arena.py` never passes `start_fen` to `play_match`, so every arena game starts
from the standard position and a deterministic engine replays the same two games. It is
therefore close to useless for measuring a search change.

`tests/match.py` imports `harness.referee.play_match` and `harness.sandbox.local` directly,
without editing `harness/`, and drives games from a varied opening set with alternating
colours. This also mirrors the competition, where rated games start from curated openings.

Two modes:

- Fixed-node A/B for feature work. Deterministic and immune to machine load, so a 15 Elo
  change is measurable in a few hundred games rather than a few thousand.
- Real time control for final validation, because time management only shows up on the
  clock.

Opponents are snapshot directories under `snapshots/<tag>/` holding a frozen copy of the
engine, so each change is measured against the version before it rather than against
`greedy`, which stops being informative almost immediately.

`Makefile` gains `test`, `bench` and `ab` targets. `pyproject.toml` gains pytest as a dev
dependency and extends the mypy `files` list to the new modules, with a relaxed
`disallow_untyped_decorators` override because numba's decorators are untyped and would
otherwise force blanket ignores.

## Staging

Each stage leaves the repo working and verified.

1. `bitboard.py`, `position.py`, `movegen.py`. Perft, fuzz, make-state and zobrist tests
   green. Nothing plays yet.
2. `tt.py`, draw detection, SEE.
3. `search.py` v1: iterative deepening, PVS, quiescence, ordering, TT, time control.
   `agent.py` switched over with the legality safety net. First real games.
4. Pruning schedule, one technique at a time, each with a recorded fixed-node A/B result.
5. Singular extensions and continuation history, measured individually.
6. Constant tuning against self-play.

## Risks

- **Compile time growth.** 2.9 s today. `tests/bench.py` asserts import stays under 30 s,
  so drift surfaces locally rather than as a validation failure on upload.
- **numba caching.** Off by default, matching the platform, where `/tmp` is cleared per
  game and caching never helps. Opt-in through an environment variable for local
  iteration only, so local and platform behaviour never silently diverge.
- **Type checking.** numba decorators are untyped, so the new modules get a targeted mypy
  override rather than blanket ignores, keeping `make gate` meaningful.
- **Memory.** 128 MB transposition table, about 2.5 MB of magic tables, plus the numba
  runtime. Comfortably inside 2 GB.
- **Clock exhaustion.** Late in a long game the base clock is gone and play is
  increment-only. Budget formulas are floored, and depth 1 always completes.

## Out of scope

- **Opening book.** Rated games start from unpublished curated positions, so a book keyed
  on the standard start is out of book immediately.
- **Endgame tablebases.** Five-man Syzygy is about 1 GB against a 50 MB limit. Three and
  four man fit but add little.
- **Evaluation.** Owned by the user. Only the seam and a ported material evaluation are
  built here.

Both excluded lookups are permitted by the rules. Neither earns its place.
