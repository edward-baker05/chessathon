# Board representation audit — 9 September 2026

The evidence does **not** justify replacing the board representation. The tested move
sets and resulting positions agree with python-chess. There are two reproducible SEE
errors and some opportunities for local optimization, but no evidence here of a board
layer failure that explains hundreds of Elo.

## Scope and isolation

Examined the current `bitboard.py`, `position.py`, `movegen.py`, and incremental NNUE
operations. Used a frozen copy in `/tmp/chessathon-representation/current` so concurrent
search changes cannot affect the experiments. `source-hashes.json` identifies the exact
source and weights. No production files were edited. No git history or old experiment
logs were consulted for this audit.

## Correctness evidence

`probe.py --verify` passed on 3,004 deterministically sampled positions from 80 newly
generated random games, starting from four positions selected to include ordinary play,
castling, promotions, and en passant. This is broad regression coverage, not a proof of
correctness or a representative distribution of search nodes.

Checks included:

- All legal move sets against python-chess; 63,339 legal child transitions.
- Every child bitboard, occupancy, mailbox, side, rights, en passant field, move counters,
  and incremental hash against a fresh encoding of the python-chess child.
- Incremental NNUE accumulators against a complete rebuild for every legal child.
- Both colours' attacks on all 64 squares and both colours' pinned pieces.
- Check detection, legal-move existence, null-move hash, and null-move board preservation.
- 69 positions with an en passant target; legal EP availability used for hashing agreed.
- All knight, king, pawn, BETWEEN, and LINE table entries against python-chess.
- 215,296 slider cases: every relevant occupancy subset for every rook/bishop origin,
  with and without occupancy outside the relevant mask, against ray tracing. Independently,
  the position attack tests above compare against python-chess.

The legal transitions included 28 en passant captures, 131 castlings, and 528 promotions.
The capture-only generator also matched its current policy: captures plus queen
promotions, excluding underpromotions. That exclusion is a quiescence-search policy
limitation; the full generator passed all four promotion choices.

The sampled maximum pseudo-legal list length was 62. This does not establish a worst-case
bound for the 256-move buffer.

## Confirmed SEE errors

These affect the material estimates used by search, not the legal moves or actual board
updates. The expected values below use the engine's own piece values. Each example has
a legal initial move and a straightforward legal recapture, verified with python-chess.

### A capture can release a pin

FEN: `4k3/8/8/8/8/4n3/8/K2bR3 w - - 0 1`

Move: `e1d1` (Rxd1). SEE returns **+330**; the exchange is **−170**.

The rook initially pins Ne3 to Ke8. After Rxd1, that pin disappears and ...Nxd1 is legal.
SEE caches `pinned_pieces` from the original state and excludes that defender anyway.
The existing comment describes pins appearing later but misses the opposite failure:
original pins can disappear. Consequently a losing capture can be classified as winning.

Fix direction: evaluate recapture legality against the evolving exchange occupancy and
piece placement. Simply recalculating pins from the unchanged original state will not
fix it. Add regressions for pins both appearing and disappearing, plus king recaptures.

### Recapturing pawns can promote

FEN: `8/7k/8/8/8/R7/1p6/b6K w - - 0 1`

Move: `a3a1` (Rxa1). SEE returns **−170**; after ...bxa1=Q the exchange is **−970**.

The initial capture wins a bishop, but the recapture wins the rook and promotes the pawn.
The code handles promotion on the initial move; inside the exchange loop it assigns
`on_square = piece` without the promotion gain or replacement piece type.

Fix direction: handle promotion within every recapture step, including its material gain
and the piece available to be captured next. Test uncontested and subsequently recaptured
promotions. These examples establish wrong answers; their game frequency and Elo cost
are not measured.

## Representation and generated code

The state is 16 uint64 fields plus a 64-byte mailbox: 192 bytes per ply. Copy-make is a
reasonable design here. The existing compiled mailbox loop already has a vectorized
path. The mailbox also supplies direct moved/captured-piece lookup; deleting it would
trade copying for other work. An unmake rewrite is not supported by these measurements.

The popcount loop already compiles to LLVM `ctpop`. Replacing it by another hand-written
bit trick has no demonstrated benefit.

The least-significant-bit helper is different: its de Bruijn implementation compiles to
bit isolation, multiplication, a shift, and a table lookup. A variant using Numba's
`trailing_zeros` intrinsic compiles to a native bit scan. The variant passed the same
correctness suite. It uses an internal Numba API, so its import should be included in
the submission's environment validation if adopted.

Rook and bishop lookup tables use 819,200 and 41,984 bytes respectively; BETWEEN and LINE
use 32,768 bytes each. These sizes alone are not evidence of a cache bottleneck.

## Measured performance

Initial compiled microbenchmark medians, nanoseconds per operation:

| Operation | Current | Contiguous signatures | Native bit scan |
|---|---:|---:|---:|
| Generate pseudo-legal moves | 142.1 | 141.2 | 129.4 |
| Generate captures/promotions | 68.0 | 65.5 | 60.4 |
| Make move | 90.2 | 106.4 | 94.3 |
| Make + legality check | 114.7 | 129.0 | 115.6 |
| Any legal move? | 360.8 | 386.8 | 344.3 |

The contiguous-signature experiment changes array type signatures in position/movegen
only. Despite appearing like an obvious optimization, it made copying slower. Do not
apply that change on the assumption that contiguous types must produce better code.
Separating state and mailbox copying into two compiled helper functions, on top of the
contiguous signatures, also passed correctness but measured **108.3 ns** for make and
**131.1 ns** for make + legality. That experiment does not support restructuring copying
either; its raw results are in `copyhelpers.json`.

The native bit scan reduced isolated generation time by about 9% and capture generation
by about 11%. **It did not demonstrate a wall-time improvement in full search.** Across
15 phase-spanning positions, three repetitions each, with a 1,048,576-node limit and
cleared TT/history for each repetition:

- Moves, scores, depths, selective depths, and node counts matched in all 45 comparisons.
- Sum of per-position median wall times: current **34.942 s**, bit scan **34.998 s**.
- Sum of per-position median process CPU times: current **34.493 s**, bit scan **33.977 s**.

The roughly 1.5% CPU-time difference is too small, and these sequential local runs too
exposed to machine variability, to claim a reliable benefit. The optional `bitscan.patch`
is provided for further controlled testing, not as an established Elo improvement.
No equal-time playing-strength claim follows from this benchmark.

## Targeted follow-up

Pseudo-legal generation produced 70,274 candidates, of which 63,339 were legal. In check,
only 700 of 3,636 candidates survived; outside check, 62,639 of 66,638 survived. A dedicated
check-evasion generator is therefore a plausible next localized optimization: double
check needs king moves, and single check restricts non-king moves to captures/blocks,
with special handling for en passant. Measure actual search-node frequencies before
estimating the return. These corpus ratios are not search runtime shares.

`has_legal_move` generates a whole list before stopping at the first legal move. An
early-exit generator may help terminal detection. Its value depends on call frequency
and the positions where search invokes it. Neither idea requires changing the board
layout or rewriting the engine.

Recommended order: fix and regression-test the two SEE cases; retain the existing board
layout; then benchmark a specialized check-evasion generator against the unchanged full
search. Only adopt speed changes that survive full-search timing. A board rewrite would
introduce substantial new correctness work without a measured bottleneck to justify it.

## Reproduction

From the repository root, with the frozen snapshot still present:

```sh
.venv/bin/python audit/representation/probe.py --engine /tmp/chessathon-representation/current --name current --verify
.venv/bin/python audit/representation/variants.py
.venv/bin/python audit/representation/probe.py --engine /tmp/chessathon-representation/bitscan --name bitscan --verify
CHESSATHON_DEBUG=0 .venv/bin/python audit/representation/search_bench.py --engine /tmp/chessathon-representation/current --name current
CHESSATHON_DEBUG=0 .venv/bin/python audit/representation/search_bench.py --engine /tmp/chessathon-representation/bitscan --name bitscan
```

Run timed processes sequentially. JSON files retain raw repetitions and checksums;
`.ll` and `.asm` files retain the inspected generated code. Microbenchmarks repeat a fixed
corpus inside compiled loops. They are not additive runtime shares; the NNUE timing loop
exercises update/forward operations using a fixed parent accumulator, while correctness
is checked separately using each position's true accumulator.
