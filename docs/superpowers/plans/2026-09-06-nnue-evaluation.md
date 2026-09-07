# NNUE Evaluation Implementation Plan

**Goal:** Replace the material-only evaluation with a quantised `768 -> 512x2 -> 1` SCReLU
network, incrementally updated per ply, costing about 21% of node rate and worth several
hundred Elo.

**Spec:** `docs/superpowers/specs/2026-09-06-nnue-evaluation-design.md`

**Tech Stack:** Python 3.12, numba 0.67, numpy 2.5, torch 2.13 (training only, never
imported by shipped code), python-chess 1.11 (boundary and tools only), pytest.

## Global Constraints

Every task's requirements implicitly include this section.

- **Contiguous types or nothing.** Every array in the inference path is typed `int16[::1]`,
  `int32[::1]` or indexed out of a C-contiguous stack. A non-contiguous numba type in this
  path is a 12x slowdown that no test will catch. Measured, not assumed.
- **No `np.dot` and no numpy slice expressions inside `njit`.** numba's linear algebra
  needs scipy, which the platform lacks, and slice expressions allocate. Explicit loops.
- **`cache=False` on every `njit` decorator.** numba bakes global array contents into
  cached binaries with no warning, and this design keeps weights in globals.
- **Root `*.py` and `weights/` are what ship.** `harness/package.py` globs exactly those.
  `tools/` is development only and must never be imported by shipped code. Training
  checkpoints go in `data/`, never `weights/`, because `weights/` ships whole.
- **torch must not appear in any shipped module.** Its per-call overhead is larger than a
  whole node's budget. Training is offline; inference is numba over int16.
- **Never name a root file after an importable module.** The zip is first on `sys.path`.
- **Do not edit `harness/`.**
- **No new runtime dependencies.** The platform preinstalls torch, numpy, python-chess,
  onnxruntime and numba and installs nothing else.
- **No commit watermark.** No `Co-Authored-By` or `Claude-Session` lines.
- **No em dashes** in any prose, comment or docstring.
- **Style:** Python 3.12, type-annotated, `ruff` and `mypy --strict` clean at
  `line-length = 100`.
- **Warm every jitted function at import** with the exact argument types it will really
  see. Import budget is 90 s and currently stands at 24.8 s.
- **A net that ships is one we trained.** No published network at any stage, including as
  an initialisation.

---

## File Structure

| File | Responsibility | Ships |
| --- | --- | --- |
| `nnue.py` | Weight loading, feature indexing, accumulator ops, forward pass | yes |
| `evaluate.py` | `evaluate` delegating to `nnue`; `material_eval` retained for tests | yes |
| `search.py` | Accumulator stack in `Work`; update at the 5 make sites | yes |
| `weights/net.npz` | The single shipped network, about 800 KB | yes |
| `tools/extract.py` | Lichess eval file to a packed training binary | no |
| `tools/train.py` | torch trainer, device-agnostic, checkpoints per epoch | no |
| `tools/quantise.py` | Checkpoint to `weights/net.npz`, with overflow proof | no |
| `tools/dataset.py` | The packed record, and the one place features are derived | no |
| `tests/test_nnue.py` | Differential, mirror and contiguity tests | no |
| `tests/test_dataset.py` | Proof that trainer features equal engine features | no |

---

## Phase 1: runtime, on this machine, light compute  [DONE, commit 2aa84b3]

Proves the engine can run a network correctly and fast, using a randomly initialised net.
No training strength is expected or measured here. This phase is worthless if rushed and
everything after it depends on being able to trust the accumulator.

### Task 1: `nnue.py`, weights, features and forward pass

**Files:** Create `nnue.py`, `tests/test_nnue.py`

- [x] Load `weights/net.npz` at import. Raise with a readable message if it is absent or
      any array shape disagrees with the compiled constants. No fallback.
- [x] `L1`, `QA`, `QB`, `SCALE`, `BUCKETS` are module constants read from the npz before
      the jitted functions are defined, so numba compiles them as literals and the loop
      bounds are known at compile time.
- [x] `feature(perspective, colour, piece, square) -> int32`, matching the spec exactly.
- [x] `refresh(acc, ply, state, mail)` rebuilding both perspectives from the mailbox.
- [x] `forward(acc, ply, state) -> int32`, SCReLU, int32 accumulation, bucket by popcount.
- [x] Warm every jitted function at import with the real argument types.
- [x] Test: evaluation of a position and of its colour-mirrored twin are exact negatives.
      This is the test that catches a perspective or sign error in the indexing.
- [x] Test: `forward` after `refresh` equals a plain numpy reimplementation of the same
      arithmetic, so the jitted version is checked against something readable.

### Task 2: incremental update and its differential test

**Files:** Modify `nnue.py`, `tests/test_nnue.py`

- [x] `apply(acc, ply, state, mail, move)` writing ply+1 from ply: one subtract and one
      add for a quiet move, plus the captured piece, the promotion piece swap, the castled
      rook and the en passant victim. Decoded from the parent state and mailbox.
- [x] `copy(acc, ply)` for the null move.
- [x] Test: over several hundred random playouts, at every ply, the incrementally updated
      accumulator equals `refresh` on the same position, element for element. This is the
      test the whole design rests on.
- [x] Test: the update path is exercised for every move flag, castling, en passant,
      promotion and promotion-with-capture, rather than only whatever random play produced.

### Task 3: wire into the search

**Files:** Modify `search.py`, `evaluate.py`

- [x] Add `acc` to `Work` as `np.zeros((STACK_PLIES, 2, L1), dtype=np.int16)`.
- [x] `evaluate.py` exposes `evaluate(acc, ply, state) -> int32` delegating to `nnue`, and
      keeps `material_eval(state, mailbox)` for the tests and the snapshot opponent.
- [x] Call `nnue.apply` at the five `make` sites, after `legal_after` passes; `nnue.copy`
      after `make_null`; `nnue.refresh` once per `think` at ply 0.
- [x] Update the three `evaluate` call sites in `negamax` and `qsearch`.
- [x] Test: a fixed-depth search from a set of positions returns the same score whether
      the evaluation came through the accumulator or through a refresh at every node.

### Task 4: measure the real cost

**Files:** Modify `tests/bench.py` if needed

- [x] Snapshot the current engine as `snapshots/material` **before** any of this lands, so
      there is a fixed opponent to measure against.
- [x] Record node rate before and after with a random net. Expect roughly 1.65 Mnps to
      1.30 Mnps. A larger drop means something in the path is not vectorising, and the
      cause is a non-contiguous numba type until proven otherwise.
- [x] Record import time. Must stay well inside the 90 s budget.

---

## Phase 2: data and trainer, smoke tested here  [DONE, commit a9641ae]

### Task 5: `tools/extract.py`

**Files:** Create `tools/extract.py`, add a `data` target to the `Makefile`

- [x] Stream the `.zst` without decompressing it to disk. Parse one JSON line per position.
- [x] Take the deepest eval per position. **Negate `cp` when Black is to move**, because
      the file is White-relative. This was determined empirically, not assumed.
- [x] Apply the spec's filters: not in check, best move not a capture, depth at least 12,
      `|cp| < 10000`. Mates saturate to +/- 12800.
- [x] Write a packed binary, about 28 bytes per position, plus a small JSON sidecar
      recording counts, the filter settings and the source file's date, so a trained net
      can be traced to the data that produced it.
- [x] Run on a small slice here to confirm correctness and throughput. The full pass is a
      desktop job.

### Task 6: `tools/train.py`

**Files:** Create `tools/train.py`, add a `train` target to the `Makefile`

- [x] Device selection: CUDA, then MPS, then CPU. The desktop has an RTX 2060; this
      machine has MPS. The same script must run on both without edits.
- [x] `768 -> L1 -> 1` per perspective with shared input weights, 8 output buckets,
      SCReLU, MSE in probability space against `sigmoid(cp / 400)`.
- [x] Clamp output weights during training so the export-time overflow bound can pass.
- [x] Checkpoint every epoch to `data/checkpoints/`, with a held-out loss logged per epoch.
- [x] Smoke test here on the small slice: loss must fall, and the net must beat
      `snapshots/material` on a fixed-node match by a visible margin. A short run on a
      small slice will produce a weak net, and that is the expected result. What is being
      tested is that the pipeline is wired correctly end to end.

### Task 7: `tools/quantise.py`

**Files:** Create `tools/quantise.py`, add a `quantise` target to the `Makefile`

- [x] Checkpoint to `weights/net.npz` at `QA`, `QB`, `SCALE`.
- [x] **Compute the exact worst case from the weights being written**,
      `max over buckets of sum_i (QA^2 * |w_i|)`, and refuse to write unless it is below
      `2^31`. Do the same for int16 accumulator overflow using the real maximum feature
      weight and the maximum piece count. Report both margins.
- [x] Report quantisation error: mean absolute difference between the float net and the
      quantised net over a sample of positions.

### Task 8: scale calibration, dropped

Planned as `tools/calibrate.py`, fitting `SCALE` so the network's output lined up with the
material evaluation. Dropped once the trainer was written, because it would have calibrated
against the worse of the two references.

The trainer's target is `sigmoid(cp / 400)` against Stockfish centipawns, and the float and
quantised networks are the same function by construction, so the network already emits
centipawns on a better scale than a material count's. Fitting to the material evaluation
would have moved it off that scale, not onto one.

What survives from this task is the observation that prompted it, and it moves to Phase 4:
the pruning margins in `search.py` still want rechecking, because the *distribution* of a
network's evaluations differs from a material count's even on the same scale.

---

## Phase 3: the real net, on the desktop

### Task 9: full extraction and training

Everything below is committed, so the desktop needs only the repo and the data file.

```
git clone <this repo> && cd chessathon && uv sync
curl -L --retry 10 -C - -o data/lichess_db_eval.jsonl.zst \
    https://database.lichess.org/lichess_db_eval.jsonl.zst      # 21.7 GB, resumable

uv run python tools/extract.py --workers 11                     # -> data/train.bin
uv run python tools/train.py --epochs 30                        # -> data/checkpoints/
uv run python tools/quantise.py --checkpoint data/checkpoints/epoch030.pt
uv run pytest tests/test_nnue.py tests/test_dataset.py -q       # must pass
```

Measured on the development machine, for comparison against the desktop:

| | rate |
| --- | --- |
| extraction, 6 worker processes | about 25,000 positions/s kept |
| training, MPS, L1 512, batch 16384 | about 80,000 positions/s |

- [x] `make data` on the desktop against the full 21.7 GB file. It keeps about 68% of what
      it reads: roughly a third is dropped because the best move is a capture, which is
      the filter doing its job.
- [x] Train to convergence. Checkpoint every epoch so a long run can be cut short at any
      point without losing the day.
- [x] Quantise a checkpoint and check the overflow margins it prints. The smoke run used
      26.7% of int32 and 12.9% of int16, both at QA 255.
- [x] `git add -f weights/net.npz`. It is gitignored so a random or smoke-test net can
      never be committed by accident and mistaken for a trained one.

### Task 10: accept or reject, then ship

- [x] **Fixed time** against `snapshots/material` is the verdict. Accept on that.
- [x] **Fixed nodes** as the diagnostic. A net that wins on nodes and loses on time means
      L1 is too large, not that the net is bad. Retrain at 256 rather than blaming the net.
- [x] `make gate`, `make zip`, confirm `agent.py` and `weights/net.npz` are at the zip root
      and the total is inside 50 MB, then upload.

### Task 11: L1 sweep, folded into the feature set decision

Originally "retrain at L1 = 256 and L1 = 1024 and pick by A/B at a real clock". Still the
right way to choose L1, but it is no longer a standalone task, because the L1 that wins
depends on the feature set and the feature set is now the first Phase 4 item. A HalfKP
input layer at L1 512 does not fit the size budget and a 768 input layer at L1 1024 is
choosing width over the thing that is actually limiting the network.

Sweep L1 once, against whichever feature set survives Phase 4 item 1, and by A/B at a real
clock rather than by held-out loss, because held-out loss cannot see the node rate L1 costs.

---

---

## Phase 4: beyond v1, in the order the evidence supports

Reordered after analysing the four lost rated games against Stockfish 19. The evidence is
in "What the four lost games showed" below, and it moved three things: the feature set went
from fourth to first, time allocation went from a Phase 3 loose end to second, and "train
for more epochs" came off the list as a lever worth planning around.

The one line summary: the epoch 30 network fixed four of the five moves that lost those
games, the ones it did not fix are king safety and quiet positional judgement, and the
feature set cannot represent either.

### 1. HalfKP or HalfKA input features

The single largest lever, and the reason the other evaluation items are ordered behind it.

`nnue.feature(perspective, colour, piece, square)` does not take the king square. The
network is a plain piece-square net and has no way to express "this configuration is
dangerous **because** my king is on g1". The blunders it still plays are exactly that:
an unsound `Bxh6` sacrifice from a level position, and a slow positional drift in R43.
More epochs cannot create a representation the features do not carry.

The size budget is almost entirely unspent, which is what makes this affordable:

| | size |
| --- | --- |
| `weights/net.npz` today, 768 x 512 | 0.53 MB |
| HalfKP, 40960 x 256 int16 | about 21.0 MB |
| HalfKA, 45056 x 256 int16 | about 23.1 MB |
| Cap on everything in the zip, unzipped | 50 MB |

- [x] Move `feature` to a king-conditioned index, and change `tools/dataset.py` in the same
      commit. `tests/test_dataset.py` is what proves the trainer and the engine agree, and
      it is the test that stops a net trained under one convention and played under another.
      Done in commit 46d34f4, as **HalfKA rather than HalfKP**, mirrored across the d/e file.
      HalfKP has no feature for the enemy king at all, so it cannot represent one side of
      the very relationship this item exists to capture. Ten percent more features to be
      able to see the thing being evaluated. The mirror folds the king onto files e to h,
      halving the table and doubling the data each weight sees; it conflates a kingside
      position with its queenside reflection, which nothing in a piece-square feature set
      could have distinguished anyway. 32 buckets by 11 slots by 64 squares, 22528 features.
- [x] Drop L1 to 256. At 512 a HalfKP layer is 42 MB and leaves no room for anything else.
      Note the size worry was overstated: `savez_compressed` puts the shipped HalfKA net at
      **5.4 MB**, and the whole zip at 5.5 MB against the 50 MB cap. Size is not a
      constraint on this decision and never was. L1 256 was kept anyway, because it halves
      the accumulator work and so pays for the king-move rebuild.
- [x] King moves now force a full accumulator refresh. Extend the Phase 1 Task 2
      differential test to cover it: over random playouts, the incrementally updated
      accumulator must still equal `refresh` at every ply, king moves included. Done, plus
      `test_a_king_move_matches_a_full_refresh`, which exercises over a hundred king moves
      specifically rather than relying on random play to produce them.
- [x] Measure node rate and import time before and after. Measured against
      `snapshots/net768` in the same session, which matters: an earlier baseline was taken
      on a quieter machine and comparing across the two invents a 9.7 s regression that does
      not exist.

      | | import | node rate |
      | --- | --- | --- |
      | 768 x 512 | 54.4 s | 720 knps |
      | HalfKA x 256 | 58.7 s | 688 knps |

      **+7.9% import, -4.4% node rate.** The npz load is not where it goes, and neither is
      refresh: `nnue.py` moved only 1.67 s to 1.99 s. `search.py` moved 44.8 s to 48.4 s.
      Applied to the worst platform init ever observed, R46's 77.8 s, +7.9% projects to
      about **84 s of the 90 s budget**; against the median round of 66.6 s it projects to
      about 72 s. So the risk is the tail, not the typical case.
- [ ] A/B at a real clock against the current network. Accept on that, not on holdout loss.
- [ ] Read the init time out of the validation log after uploading. The 1.55x desktop to
      platform ratio is an estimate from a single pair of numbers, and the log replaces the
      whole projection with a fact for the price of one of ten daily uploads.

### 2. Time allocation at the real control

Cheap, measurable today, and a whole-game risk rather than an Elo tweak. Promoted above the
margin sweep for the reason already given below: margins measured under erratic time use are
measured against noise.

The clocks in the lost games say the budget is front loaded. R41 reached move 28, the move
that lost the game, with **15.0 s left**, which buys a 0.78 s soft budget, having had 59.3 s
at move 14. R45's slowest move was 13.7 s against a 13.66 s hard limit at a 120 s clock,
so it hit the hard cap on an early move. All four losses finished with 4.6 s to 6.9 s.

This sits awkwardly against the replay measurement below, which found a 0.71x mean overshoot
and 61 s to 72 s unspent, and asked whether `SOFT_BASE` was too low. Those two readings
disagree, and the disagreement is the point: one is a replay of a won game, the others are
real losses.

- [ ] Run `tools/replay.py` over the four lost PGNs, not just the won one, before touching
      `SOFT_BASE`. Raising it on the evidence of a single replayed game would be exactly the
      mistake the existing note warns against.
- [ ] Confirm which build actually played Rounds 32 to 46 before drawing conclusions from
      their clocks. The dashboard check below is the same question.

### 3. Search throughput and depth

Worth about as much as evaluation right now, which was not obvious before it was measured.
On the 19 positions where the old build went wrong, going from the real clock to a 10M node
search cut mean centipawn loss from 72.9 to 39.4. Roughly 12x the nodes halved the error.

- [ ] Sweep the six centipawn pruning margins in `search.py`, which were tuned against the
      material evaluation and mean something different under a network. Lines 374, 459,
      463, 477 and 540. One at a time, each against the current snapshot.

      This needs a change first: the margins are literals inside jitted functions, so a
      sweep means lifting them to module constants read from the environment at import.
      numba freezes globals at compile time and compilation happens at import, so that
      works, and it lets `tests/match.py` A/B two settings as two snapshots.

      Note on scale: no separate calibration step is needed. The trainer's target is
      `sigmoid(cp / 400)` against Stockfish centipawns, so the network's output is already
      on a centipawn scale, and a better one than the material evaluation's. The margins
      still want checking, because the *distribution* of a network's evaluations differs
      from a material count's even on the same scale.
- [ ] **Correction history.** Genuinely absent from `search.py`: a running correction to the
      static evaluation, keyed on pawn structure and material, applied when search results
      systematically disagree with the static eval. It pairs particularly well with a
      network and is self-contained. Perhaps 15 to 30 Elo. Check its compile cost against
      the import headroom below before committing to it.

      Note that the evaluation's weakness is now known to be ranking rather than scale: it
      beats material on correlation and barely beats it on absolute error. That is the
      shape of error a correction history is built to absorb, so this may be worth more here
      than the 15 to 30 Elo estimate assumed.
- [ ] Singular extensions and probcut are the notable absences from an otherwise complete
      search. Null move, LMR, killers, history, continuation history, aspiration windows,
      reverse futility, razoring and SEE are all present. Incremental next to items 1 and 2.

### 4. A regression suite from the lost games

The analysis produced specific positions the engine got wrong. They are worth keeping as a
fast check that a retrain has not regressed, and they cost seconds rather than arena hours.

- [ ] Freeze the 19 probed positions with their Stockfish evaluations as a test fixture.
      Assert centipawn loss against a threshold per position rather than a specific move,
      because matching Stockfish exactly is not the bar and never will be.
- [ ] The two that are still wrong are the interesting ones: R41 move 28 (Stockfish wants
      `Ng5`, still 321 cp wrong at 10M nodes) and R43's quiet middlegame positions.

### 5. Data, once capacity can absorb it

- [ ] Self-play data: positions from our own engine, labelled by our own search at fixed
      depth. In-domain and entirely our own. Train on the combined set, A/B against the
      current net.
- [ ] The data scaling experiment below stays **after** the feature change, for the reason
      it already gives: more data only pays once capacity can absorb it. A 768 input net is
      the capacity limit, so running it now would measure the wrong architecture.

### 6. Buy back import headroom, dropped: it buys 0.4 s

Measured, twice each, by removing both functions outright and timing `import agent` in a
fresh process: 47.98 s before, 47.6 s after. **About 0.4 s.**

The premise was wrong. Both functions carry explicit `njit` signatures, so numba compiles
them eagerly at decoration whether or not anything ever calls them. Removing the warm-up
*call* saves nothing at all; removing the whole function saves 0.4 s.

Where the time actually goes, per module, on this machine:

| | |
| --- | --- |
| `bitboard` 2.6 s, `position` 1.8 s, `movegen` 1.7 s | |
| `nnue` 1.4 s, `evaluate` 0.1 s, `tt` 0.7 s | |
| **`search`** | **39.8 s, 83% of the total** |

So import headroom is not a constraint on the feature set, which was the reason this item
existed. `nnue.py` is 3% of import. If headroom is ever genuinely needed it comes out of
`search.py` or it does not come at all.

### Not a lever: more training epochs

Held-out loss is still falling, so the 60 epoch run should help a little, but the curve is
flattening and it is not where the remaining strength is.

| checkpoint | holdout loss |
| --- | --- |
| epoch 1 | 0.010735 |
| epoch 15 | 0.009784 |
| epoch 30 | 0.008602 |

Take the 60 epoch net if it wins its A/B. Do not plan around it.

Ship whichever version last won its match.

---

## Evidence from the rated losses, 2026-09-07

### What the four lost games showed

Rounds 32, 41, 43 and 45, the four rated losses in `logs/`, analysed with Stockfish 19 at
depth 18 with a 2.0 s cap per position. 508 positions, 253 of them ours. Centipawn loss for
one of our moves is the drop in Stockfish's evaluation between the position before it and
the position after it, both clamped to +/- 1500 so that mate distance cannot masquerade as a
blunder. The epoch 30 network was then asked for its own move in each position where the old
build went wrong, at the clock it actually had there and again at a fixed 10M nodes.

The method above is the whole of what is needed to reproduce it. The Stockfish binary was
fetched to a scratch directory and is deliberately not in the repo: native binaries are
rejected at upload, and `harness/package.py` globs root `*.py` and `weights/`, so anything
committed at the root is a submission risk rather than a tool.

**Four of the five game-losing blunders are fixed.** A game-losing blunder here means a loss
of 350 cp or more from a position that was still playable, evaluation above -600.

| game | move | old loss | at its real clock | at 10M nodes |
| --- | --- | --- | --- | --- |
| R32 | 36.Bf2 | 536 | **16** (Rg1) | 0 (Rg1) |
| R45 | 16.Bxh6 | 491 | **34** (b4) | 39 (b4) |
| R45 | 23.fxe4 | 487 | **0** (Qg8+) | 0 (Qg8+) |
| R45 | 51.Kh4 | 366 | **-17** (Kg3) | 38 (Kg2) |
| R41 | 28.Ne5 | 583 | **614** (Re8+) | 321 (c7) |

Over all 19 probed positions: mean loss 215.5 for what was played, 72.9 for the new network
at the real clock, 39.4 at 10M nodes. Better on 13, unchanged on 4, worse on 2. It matches
Stockfish's first choice 8 times of 19, where the old build matched it 0 times.

R45's `Bxh6` is the most encouraging single result. It is an unsound sacrifice from a level
position played with 51.7 s on the clock, so it was never a time problem, and the network no
longer plays it.

**R41 move 28 is a real remaining defect.** It is still 321 cp wrong at 10M nodes, so deep
search does not rescue it and it is the evaluation, not the search or the budget.

**R43 is unchanged and structurally so.** No single blunder: a slow drift from level to lost
across moves 16 to 30. The new network repeats the old move on 24.h4, 32.Ba7 and 33.Rg7.

### How good the evaluation actually is

Measured against the Stockfish labels on a 60,000 position sample of the training holdout,
with plain material scored on the same positions as a baseline. MAE is after a best linear
rescale, so a scale convention difference is not counted as error.

| band | NNUE r | material r | NNUE MAE | material MAE |
| --- | --- | --- | --- | --- |
| all | +0.798 | +0.672 | 1364 | 1691 |
| \|label\| < 800 | +0.648 | +0.438 | 74.0 | 83.6 |
| \|label\| < 400 | +0.462 | +0.225 | 56.1 | 57.5 |
| \|label\| < 200 | +0.344 | +0.139 | 37.7 | **37.3** |

The network has learned real positional content: it roughly doubles material's ranking power
in the near-level positions that decide games. But read the two right-hand columns as well.
In absolute error it is barely better than counting material, and below 200 cp it is worse.
**It ranks positions better than it scores them**, and +0.46 in the band that matters is
modest. That is the gap item 1 is meant to close.

A first pass measured this on the 253 positions from the lost games instead and found no
edge over material at all. That result did not survive: 85 positions inside +/- 400 cp, drawn
from four games and heavily autocorrelated, is an adversarially selected sample. The holdout
numbers above are the ones to trust. Recorded because the discarded version is the more
alarming one and someone will otherwise measure it again.

### The shipped network matches its checkpoint, and the tests do not check that

Verified directly: `weights/net.npz` against `data/checkpoints/epoch030.pt` over 253 real
positions gives r = 1.0000, mean absolute difference 6.2 cp, worst 35.6 cp. Quantisation and
the numba inference path are both correct, so none of the weakness above is a bug.

Worth noticing how that was established, though. `tests/test_nnue.py` passes 11 tests and
none of them would have caught it. They prove the int path agrees with a numpy
reimplementation **of the same int weights**, that incremental updates match `refresh`, and
that evaluation is mirror-symmetric. Nothing compares against torch, which is what the
training loss was measured on. A bucket rule or perspective convention that diverged between
`tools/train.py` and `nnue.py` would pass all 11.

- [x] Add a test that loads a checkpoint and asserts the quantised net agrees with the float
      model over a sample of positions. It is the one differential test the suite is missing,
      and item 1 changes both sides of exactly that boundary. Done, `tests/test_quantise.py`.

      It compares a float checkpoint against the network `tools/quantise.py` exports from
      it, evaluated by a numpy transcription of `nnue`'s integer arithmetic. It cannot call
      `nnue` itself, because that module loads one fixed weights file at import and compiles
      it into numba literals, so it cannot be pointed at the file under test.

      Its thresholds sit in a gap that was measured rather than guessed, by deliberately
      breaking the exporter two ways:

      | export | mean error / eval spread | correlation |
      | --- | --- | --- |
      | correct | 0.11 to 0.16 | 0.989 to 0.991 |
      | factoriser dropped | 0.54 to 0.73 | 0.64 to 0.77 |
      | perspectives transposed | 0.99 to 1.53 | -0.11 to 0.34 |

---

## Constraints and open questions carried out of the first day

### The import budget, recomputed after the work moved machines

**Every local timing above this line was measured on the Mac and none of them transfer.**
The plan says "24.8 s here" and "28.2 s here" and derives a 2.43x platform ratio from them.
Those were measured on the laptop, which had MPS. Development has moved to the desktop, an
RTX 2060 with 12 cores, which is slower single core and can run overnight, and nothing on
the Mac is being maintained. On the desktop the same build imports in about **48 s** and
benches at **720 knps**, so the machine is roughly 1.8x slower single core than the numbers
this document was written against.

The 2.43x ratio is therefore dead: it was Mac to platform. Desktop to platform, from the
worst init ever observed, is `77.8 / 50.1 =` about **1.55x**, so the 90 s platform ceiling
is about 58 s of local import.

Treat that 1.55x as a rough estimate, not a measurement. It rests on one pair of numbers,
and `logs/aichessathon-games.csv` shows init varying from 56.8 s to 77.8 s across rounds on
differently named machines, so a good part of that spread is the platform's hardware rather
than our build. The honest way to settle it is to upload and read the validation log.

Local timings are also worthless while anything else is running. A 9.7 s import "regression"
was measured and chased before it turned out to be a game open in the background. Compare
builds **in the same session**, or not at all.

### The soft time limit overshoots, and it is the most expensive class of bug

Found in the v4 validation log, which reported `slowest 14.4 s` and `slowest 11.5 s` on the
two smoke games. That log is no longer in the tree; the figures are kept here because they
are what the fix below was measured against.

Reproduced locally from the same two smoke positions with a 120 s clock, against a soft
limit of 5.69 s and a hard limit of 22.76 s:

| position | move time | overshoot vs soft |
| --- | --- | --- |
| smoke 1 | 10.7 s | 1.9x |
| smoke 2 | 17.8 s | 3.1x |

17.8 s is 78% of the way to the hard limit. For comparison, the material build in rated
Round 35 had a slowest move of 6.8 s, only 1.2x its soft limit, so this got worse with the
network: slower nodes mean each iteration takes longer in wall clock, and the soft limit is
only sampled between them.

Two causes, both in `search_root`:

1. `past_soft_limit` is checked only **after** a completed depth, so an iteration that
   starts just under the limit runs to completion regardless of how long it takes.
2. The aspiration window `while True:` re-search loop sits **inside** that check. A failed
   window re-searches the same depth with nothing but the hard limit to stop it.

The hard limit does prevent an outright flag, so this is time trouble rather than an
immediate loss, and the budget formula's `time_left / 22` self-corrects as the clock drains.
But an engine that habitually spends 3x its intended budget arrives in the endgame with
very little left, and Round 35 already finished with only 13.0 s of 144.0 s.

Two games were played at the real 120 s + 0.5 s control against `snapshots/material` to
check. Both were won by checkmate with no flag and no failed termination. That is
reassuring but weak evidence: two games, both decisive, so neither reached the long endgame
where the accumulated overshoot would actually bite. It rules out an immediate loss, not the
time trouble. Treat the fix as still worth making.

- [x] Do not start a new iteration unless it is likely to finish. Done by predicting the
      next iteration from the growth of the last two, not by a fixed fraction. Note that the
      prediction has to aim at the stretch limit rather than the soft limit; aiming at the
      soft limit guarantees no overshoot and costs three quarters of the budget.
- [x] Check the soft limit inside the aspiration re-search loop as well, so a failed window
      cannot run unbounded. Also inside the root move loop, which the original list missed:
      an iteration whose cost was underestimated was still reaching the hard limit.
- [x] Re-measure at the real 120 s + 0.5 s control. `tools/replay.py` does it from a played
      PGN in about two minutes, without needing a whole arena match.

Done, in `docs/superpowers/specs/2026-09-06-time-management-design.md`, together with the
budget reshape the overshoot was hiding. Measured old against new on the same machine, same
network, over the Round 36 game, now `logs/aichessathon-round-36-epoch-mate.pgn`:

| | old | new |
| --- | --- | --- |
| overshoot mean / median / max | 1.78x / 1.85x / 3.35x | 0.71x / 0.79x / 1.36x |
| worst single move | 13.4 s, 84% of hard | 4.5 s, 45% of hard |
| clock after 20 of our moves | 16.5 s | 71.7 s |

Confirmed on the platform. `logs/aichessathon-v5-591ad99320da.log` is the validation log for
the build carrying both the fix and the trained network, and it reports slowest moves of
**10.2 s and 5.7 s** across its two smoke games, against 14.4 s and 11.5 s for v4. The same
log is now the best available reading of init cost:

```
smoke game 1   ready in 70.7 s of the 90 s init budget, played 7 moves, slowest 10.2 s
smoke game 2   ready in 71.6 s of the 90 s init budget, played 8 moves, slowest 5.7 s
                                                             upload 639,862 bytes
```

Two things to carry forward from it. 70.7 s and 71.6 s is **79% of the init budget**, higher
than the 68.5 s the 2.43x ratio predicted, so the headroom in the spec is optimistic and item
1 of Phase 4 spends from a smaller margin than 8.8 s of local time. And 10.2 s is still a
large single move for a smoke game, so the overshoot is much improved rather than closed.

Still open, and the reason this is not finished:

- [ ] The A/B at 30 s + 0.125 s against a snapshot frozen before the change. Nothing above
      is an Elo measurement.
- [ ] Decide whether a mean of 0.71x is underspending. The easy moves come in at 0.4x to
      0.6x by design and the hard ones at 1.0x to 1.4x, which is the intended shape, but the
      replayed game ended with 61 s to 72 s unspent. Raising `SOFT_BASE` is the lever. Do
      not touch it on one game's evidence.

Worth doing **before** the margin sweep: it is cheap, it is a whole-game risk rather than an
Elo tweak, and margins measured under erratic time use are measured against noise.

### Is more data worth acquiring? Measure before deciding

The network is about 402k parameters against roughly 189 million available positions, some
470 positions per parameter. Engines that train on billions run 1.5M parameter nets. So
data volume is plausibly **not** the binding constraint, and the honest way to find out is
cheap:

- [ ] Train the same architecture on 25%, 50% and 100% of `train.bin` and plot holdout loss
      against data size. Still falling steeply at 100% means data-limited and acquisition
      pays. Flattened means capacity-limited and more data buys nothing.

Do this **after** the L1 sweep, not before. More data only pays once capacity can absorb
it, so the architecture decision comes first.

If it does turn out data-limited, in preference order:

| Source | Volume | Notes |
| --- | --- | --- |
| Lichess monthly PGN dumps | about 66M usable per month, 4+ months live | 29 GB per month. Measured: **10.2% of games carry `[%eval]`**, about 65 per annotated game. Labels are shallower than the eval file's, but these are real games and they carry the **result**, which unlocks the eval/WDL lambda blend that was dropped on pipeline cost |
| Self-play, labelled by our own search | unlimited | Free, entirely ours, and in-domain. Already a Phase 4 item above |
| Public Stockfish binpacks | hundreds of GB | Legal as training data. Download size is the practical blocker |

Prefer the PGN dumps over more eval-file data: the value is the game results, not the count.

**Do not rent cloud compute.** The binding constraint is single-core inference speed on a
2.60 GHz core, not training throughput. A faster GPU buys nothing that can be spent.

### Measured figures for planning the full run

| | |
| --- | --- |
| Lines in `lichess_db_eval.jsonl.zst` | about 278 million (5.33x compression, 416 bytes per line) |
| Kept after filters, about 68% | about 189 million positions |
| `data/train.bin` | about 6.0 GB |
| Extraction, 11 workers | about 1 hour |
| One training epoch at 200k/s | about 16 minutes, so 30 epochs is about 8 hours |
| `dataset.unpack` throughput | 2.0M positions/s, 25x the training rate, so the data path is **not** a bottleneck and needs no prefetching |
| A/B harness | about 1 minute per game, almost entirely numba compile in the freshly spawned agent processes. Budget match sizes by that, not by node count |

### What the ladder logs showed, and what to do about it

`logs/` holds a Round 31 PGN and a Round 35 log. Both games were played by the material
evaluation, identifiable from the aimless quiet play that an evaluation with no positional
terms produces.

Round 31 was **not** played by the build in `snapshots/material`. In that game the engine
declined `d8=Q+` eight consecutive times, eventually blocking its own promotion square with
its own bishop, and stalemated with a rook, bishop and pawn against a bare king. The
current material build promotes there at depth 1 on 200 nodes, and still promotes when the
whole game is replayed through `agent.get_move` with the real clocks and history. So this
is an older upload rather than a live bug, and no fix is outstanding.

- [ ] Check the dashboard for which submission is actually live. Round 31 suggests the
      deployed build had drifted behind the local work, which would mean the network upload
      is worth more than the +168 Elo measured against `snapshots/material`.
- [ ] After uploading, read the new validation log's init time against the 68.5 s
      prediction above. That confirms or corrects the 2.43x platform ratio.

### Not worth doing, so they do not eat the remaining days

- **Opening book.** Rated games start from curated, unpublished positions, so a book keyed
  on the start position is out of book on move one.
- **Syzygy tablebases.** 3-4-5 man WDL is about 380 MB against a 50 MB cap. Only 3-4 man
  fits and is worth almost nothing, and probing would mean building a `python-chess` board
  per probe from the bitboard state.
- **ONNX or torch at runtime.** Per-call overhead is 20 to 50 us against a node budget of
  about 600 ns.
- **Threads.** One core, and the contract says threads past the first cost time.

### Minor, noted but not worth doing alone

- `search.py` uses `tt_static != 0` as "a static eval is stored", so a position that
  genuinely evaluates to exactly 0 is recomputed. A wasted evaluation, not a wrong one.
