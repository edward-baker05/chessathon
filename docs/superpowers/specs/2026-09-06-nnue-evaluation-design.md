# NNUE evaluation design

Replaces the material-only `evaluate` with a quantised, incrementally updated neural
evaluation. Every number in this document was measured on the development machine, an
Apple M1, before the design was fixed. Where a figure decided a choice, the measurement
that produced it is named.

## What the platform allows, read from the source

Fetched from `https://aichessathon.com/docs/agent-contract.md` and
`https://aichessathon.com/docs/rules.md` on 2026-09-06.

- A network that ships must be one the team trained. Starting from a published chess
  network counts as shipping it, so no published net is used at any stage.
- Training data is unrestricted and explicitly includes positions annotated by an existing
  engine. The ban covers what is inside the zip, not what the net learned from.
- Weights are not native binaries. `.npz` ships fine.
- 50 MB unzipped for the whole submission. The net designed here is 800 KB.
- One core of an AMD EPYC 9V74 at 2.60 GHz, 2 GB, no network, 120 s + 0.5 s per move.
- 90 s import budget before the clock. numba compilation and weight loading land there.

## The measurement that fixed the architecture

The first question was what an evaluation can cost. The engine currently runs at
1.65 Mnps on the development machine, so a node costs roughly 600 ns and an evaluation is
requested at nearly every node: `negamax` computes a static eval and `qsearch` computes a
stand-pat.

Two findings came out of micro-benchmarking numba, both of which changed the design.

**numba only vectorises C-contiguous array types.** The same accumulator update loop
compiled from an `int16[:, :]` argument runs at 261 ns and from `int16[::1]` arguments at
22 ns, a 12x difference, because LLVM will not vectorise a loop over an array it cannot
prove is unit-stride. Every array in the inference path is therefore typed `int16[::1]`
or indexed out of a C-contiguous stack so that it is. A `(plies, 2, L1)` stack indexed
`acc[ply, p]` types as `array(int16, 1d, C)` and hits the vectorised speed; a flat 1-D
array with variable offsets does not, and measured 36x slower.

**`np.dot` is unavailable.** numba's linear algebra requires scipy, which is not in the
platform's five packages. numpy slice expressions inside `njit` allocate and measured 18x
slower than an explicit loop. So: explicit loops over contiguous int16, nothing else.

Measured cost per node, accumulator update for both perspectives plus one forward pass:

| L1 | CReLU | SCReLU | share of a 600 ns node |
| --- | --- | --- | --- |
| 256 | 44 ns | 65 ns | 11% |
| 512 | 86 ns | 126 ns | 21% |
| 1024 | 171 ns | 253 ns | 42% |

And the decisive negative result: a Stockfish-style output stack of `2*L1 -> 16 -> 32 -> 1`
costs **1386 ns per node at L1=512**, more than doubling node cost, because the first
matrix alone is 16x the work of a single output row and there is no way to make it cheaper
in this instruction set. A deep output stack is not affordable here.

What is affordable is the single-layer perceptron shape, `768 -> L1x2 -> 1`, which is what
the modern `bullet`-trained engines use and which reaches 3200+ Elo in C. This is not a
compromise forced by Python. It is the same shape the strongest open engines run.

## Architecture

```
feature(perspective, colour, piece, square)
    = ((colour ^ perspective) * 6 + piece) * 64 + (square ^ (perspective * 56))

acc[white] , acc[black]          int16[512] each, per ply
output      = sum over i of screlu(acc[stm][i]) * W[bucket][i]
            + sum over i of screlu(acc[!stm][i]) * W[bucket][512 + i]
            + B[bucket]
bucket      = (popcount(occupancy) - 2) >> 2          0..7
```

- **768 input features**, piece-square-colour, mirrored per perspective. Chosen over
  HalfKP, HalfKA or king buckets because every move is then exactly one subtract and one
  add with no accumulator refresh path anywhere, which removes the single largest source
  of correctness risk. King buckets are a later, weights-shaped change: the accumulator
  code gains only a bucket-change refresh.
- **L1 = 512, SCReLU.** Squared clipped ReLU is worth roughly 20-30 Elo over CReLU in
  published ablations for one extra multiply, measured at 40 ns per node. L1 is a single
  constant so 256 and 1024 can be retrained and A/B tested rather than argued about.
- **8 output buckets by piece count.** One popcount and a row index into a contiguous
  `(8, 1024)` array, so it costs nothing measurable and the row still types contiguous.

## Quantisation

`QA` scales the input weights and therefore the accumulator; `QB` scales the output
weights; `SCALE` converts the network's output into centipawns.

```
acc_q     = round(w_real * QA)                    int16
out_w_q   = round(w_real * QB)                    int16
eval_cp   = (sum / QA + bias) * SCALE / (QA * QB)
```

SCReLU **must accumulate in int32**. int64 accumulation measured at 516 ns against 81 ns
for int32, because NEON carries two int64 lanes against eight int16. int32 accumulation
can overflow: at `QA = 255` the worst case over 1024 terms is 9.3e9 against an int32
maximum of 2.1e9.

The answer is not to pick a scale that looks safe. At export time the quantiser computes
the exact worst case from the weights it is actually writing,
`max over buckets of sum_i (QA^2 * |w_i|)`, and refuses to write the file unless it is
below `2^31`. The same check bounds the accumulator against int16 overflow using the real
maximum feature weight and the maximum number of pieces on the board. A scale that cannot
be proven safe for the specific net being shipped is not shipped.

`SCALE` needs no calibration step, and an early plan to fit it against the material
evaluation was dropped as actively wrong. The trainer's target is `sigmoid(cp / 400)`
against Stockfish centipawns and the float and quantised networks are the same function by
construction, so the network already emits centipawns on a better scale than a material
count's. Fitting to the material evaluation would have moved it off that scale rather than
onto one. The six pruning margins in `search.py` still want rechecking, because the
*distribution* of a network's evaluations differs from a material count's even on the same
scale, and that is a Phase 4 task in the plan rather than a quantisation concern.

## Runtime integration

`nnue.py` is a new root module holding the weights, the feature indexing, the accumulator
operations and the forward pass. It imports only `numpy`, `numba` and constants from
`bitboard.py`, so there is no import cycle with `position.py`.

```
refresh(acc, ply, state, mail)          rebuild from scratch, root only
apply(acc, ply, state, mail, move)      parent ply -> ply + 1, one sub and one add
copy(acc, ply)                          null move
forward(acc, ply, state) -> int32       the evaluation
```

The update is called from `search.py` at the five `make` sites, immediately **after** the
`legal_after` check rather than inside `position.make`. Three reasons: `position.py` stays
free of evaluation concerns so the perft and SEE suites are untouched; pseudo-legal moves
that turn out illegal cost no update; and the call sites already know the move was legal.
The cost is that the from/to/capture/castle decode exists in two places, which a
differential test against a full refresh over random playouts pins down exactly.

`evaluate.py` keeps its name and its role, delegating to `nnue.forward`. The material
evaluation stays in the tree as `material_eval`, used by the tests and by the
`snapshots/material` A/B opponent, and **not** as a runtime fallback: a missing or
mis-shaped `weights/net.npz` raises at import. A loud failure appears in the validation
log before the build ever plays a rated game, and the previous valid build keeps playing.
A silent fallback would play a whole round hundreds of Elo weak and look like a
mysterious rating drop.

## Training data

`lichess_db_eval.jsonl.zst`, 21.7 GB, refreshed 2026-08-02. FEN plus Stockfish `cp` or
`mate` at a stated depth. Legal as training data by the rule quoted above.

**The sign convention was determined empirically, not assumed.** Over 10,482 decisive,
materially imbalanced positions from a sample, the eval sign agreed with White's point of
view 81.0% of the time and with the side to move's 49.8%, a coin flip. `cp` is
White-relative and the extractor negates when Black is to move. Assuming the other
convention would have produced a net that plays close to randomly, with nothing in the
training curve to indicate why.

Target and filters:

```
target = sigmoid(cp / 400)                  train in probability space
mate   -> cp = +/- 12800                    saturating
loss   = MSE(sigmoid(out / 400), target)

keep only: not in check
           best move is not a capture       leave tactics to the quiescence search
           depth >= 12
           |cp| < 10000
```

Probability space rather than raw centipawns because it stops already-decided positions
from dominating the gradient, which is what every modern trainer does.

Positions are packed to about 28 bytes: an occupancy bitboard, four bits per occupied
square, side to move, and an int16 score.

## Plan for strength beyond v1

The Lichess file is analysis positions, which skew tactical and critical rather than
toward the quiet positions a search actually evaluates. After v1 ships and is measured:

1. v2, the same data at larger L1 or longer training, A/B against v1.
2. v3, adding positions sampled from our own engine's self-play and labelled by our own
   search at fixed depth. In-domain, entirely our own, and the standard way to push past
   the teacher's distribution.
3. A full sweep of the six centipawn pruning margins, which were tuned against the
   material evaluation and mean something different under a network.

Ship whichever version last won its match.

## How a change is accepted

Fixed-node A/B is the cheap mode and it is the **wrong instrument** for this change: it
hands both sides the same node count and so hides the 21% of node rate the network costs.

- **Acceptance** runs at a real clock against the previous snapshot. That is the verdict.
- **Fixed nodes** stays as a diagnostic. It isolates evaluation quality from speed, so a
  net that wins on nodes and loses on time means L1 is too large, not that the net is bad.

## What the platform costs, measured from a rated game log

`logs/AI Chessathon Round 35 Log.log` is a real rated game and it reports the init time the
platform actually charged. That is the only direct measurement of how much slower their
core is than the development machine, and it turns the 90 second init budget from an
abstraction into a number.

```
INIT
  Ready in       60.2 s
  Budget         90.0 s
  Used           67 percent
```

Round 35 was played by the material build, which imports in 24.8 s here. So the platform
is **2.43x slower** at numba compilation than this machine.

| build | local import | platform | share of the 90 s budget |
| --- | --- | --- | --- |
| material | 24.8 s | 60.2 s (measured) | 67% |
| with the network | 28.2 s | 68.8 s and 69.4 s (measured) | 76% |

The prediction from the 2.43x ratio was 68.5 s and the validation log for the network build
reported 68.8 s and 69.4 s across its two smoke games, so the ratio is confirmed and the
build validates. It also confirms that Round 35 was the material build, which had been an
inference from its move quality rather than a fact.

The network fits, with roughly **21 s of platform slack, which is 8.8 s of local import
time**. That is now a design constraint on everything that follows, because every feature
that adds a jitted function adds compile time: correction history, input king buckets and a
larger L1 all spend from those 8.8 s. Missing the init budget loses every game in the round.

`cache=True` cannot buy the headroom back. numba bakes the contents of global arrays into
cached binaries with no warning, and this design keeps the network's weights in globals, so
a cached build would serve stale weights.

There is one free saving available, not yet taken: `perft` in `movegen.py` and
`material_eval` in `evaluate.py` are both warmed at import but used only by tests. Dropping
those two warm-ups costs the test suite a one-off compile and buys back platform budget.

Time management for the material build looked healthy: 131.0 s used of the 144.0 s
available over 48 moves, 13.0 s left at the end, slowest move 6.8 s, which is 1.2x its soft
limit.

Under the network it is **not** healthy. The v4 validation log reports slowest moves of
14.4 s and 11.5 s, and those positions reproduce locally at 10.7 s and 17.8 s against a
5.69 s soft limit, up to 3.1x over. The soft limit is only sampled between completed depths
and the aspiration re-search loop sits inside that check, so a single iteration or a failed
window runs until the hard limit. The plan carries the fix; it is listed ahead of the margin
sweep because margins measured under erratic time use are measured against noise.

## Risks

| Risk | Mitigation |
| --- | --- |
| Incremental accumulator drifts from a true refresh | Differential test over random playouts, asserting equality every ply |
| int32 overflow in the output sum | Exact worst-case bound computed from the shipped weights at export, hard failure |
| Sign or perspective error in feature indexing | Test that the evaluation of a position and of its colour-mirrored twin are negatives |
| Import budget | 28.2 s local, about 68.5 s of the platform's 90 s. Only 8.8 s of local slack left. `tests/bench.py` gates it, and the section above explains why the margin is thinner than it looks |
| Net trained on the wrong target | Held-out loss plus a fixed-node A/B against `snapshots/material` before anything ships |
