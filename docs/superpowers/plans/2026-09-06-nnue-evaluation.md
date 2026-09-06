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

- [ ] `make data` on the desktop against the full 21.7 GB file. It keeps about 68% of what
      it reads: roughly a third is dropped because the best move is a capture, which is
      the filter doing its job.
- [ ] Train to convergence. Checkpoint every epoch so a long run can be cut short at any
      point without losing the day.
- [ ] Quantise a checkpoint and check the overflow margins it prints. The smoke run used
      26.7% of int32 and 12.9% of int16, both at QA 255.
- [ ] `git add -f weights/net.npz`. It is gitignored so a random or smoke-test net can
      never be committed by accident and mistaken for a trained one.

### Task 10: accept or reject, then ship

- [ ] **Fixed time** against `snapshots/material` is the verdict. Accept on that.
- [ ] **Fixed nodes** as the diagnostic. A net that wins on nodes and loses on time means
      L1 is too large, not that the net is bad. Retrain at 256 rather than blaming the net.
- [ ] `make gate`, `make zip`, confirm `agent.py` and `weights/net.npz` are at the zip root
      and the total is inside 50 MB, then upload.

### Task 11: L1 sweep

- [ ] Retrain at L1 = 256 and L1 = 1024 on the same data. Pick by A/B at a real clock, not
      by held-out loss, because held-out loss cannot see the node rate that L1 costs.

---

## Phase 4: beyond v1, in the order the evidence supports

- [ ] Sweep the six centipawn pruning margins in `search.py`, which were tuned against the
      material evaluation and mean something different under a network. Lines 374, 459,
      463, 477 and 540. One at a time, each against the current snapshot.

      This needs a change first: the margins are literals inside jitted functions, so a
      sweep means lifting them to module constants read from the environment at import.
      numba freezes globals at compile time and compilation happens at import, so that
      works, and it lets `tests/match.py` A/B two settings as two snapshots.

      Deliberately not done yet. Sweeping margins against a smoke-test net measures the
      net's weakness, not the margins. This waits for the real net.

      Note on scale: no separate calibration step is needed. The trainer's target is
      `sigmoid(cp / 400)` against Stockfish centipawns, so the network's output is already
      on a centipawn scale, and a better one than the material evaluation's. The margins
      still want checking, because the *distribution* of a network's evaluations differs
      from a material count's even on the same scale.
- [ ] Self-play data: positions from our own engine, labelled by our own search at fixed
      depth. In-domain and entirely our own. Train v3 on the combined set, A/B against v2.
- [ ] Input king buckets, `768 x N`, if the margin sweep and self-play data are done and
      there is still time. This reintroduces an accumulator refresh path, so it needs its
      own differential test before it can be trusted.
- [ ] **Correction history.** Genuinely absent from `search.py`: a running correction to the
      static evaluation, keyed on pawn structure and material, applied when search results
      systematically disagree with the static eval. It pairs particularly well with a
      network and is self-contained. Perhaps 15 to 30 Elo. Check its compile cost against
      the import headroom below before committing to it.
- [ ] **Buy back import headroom**, which every item above spends. `perft` in `movegen.py`
      and `material_eval` in `evaluate.py` are warmed at import but used only by tests;
      dropping those two warm-ups costs the test suite a one-off compile and buys platform
      budget. Measure the saving before and after with `make bench`.

Ship whichever version last won its match.

---

## Constraints and open questions carried out of the first day

### The import budget is the binding constraint on new features

`docs/superpowers/specs/2026-09-06-nnue-evaluation-design.md` has the full working. In
short: a real rated game log reports `Ready in 60.2 s` of a 90 s budget for the material
build, which imports in 24.8 s here, so the platform is **2.43x slower** at numba
compilation. The network build imports in 28.2 s here, so about 68.5 s there, about 76% of
the budget.

**That leaves about 8.8 s of local import time to spend.** Every jitted function added
spends from it, and missing the budget loses every game in the round. Check `make bench`
against this before adding anything that compiles.

### The soft time limit overshoots, and it is the most expensive class of bug

Found in the v4 validation log (`logs/AI Chessathon v4 log.log`), which reports
`slowest 14.4 s` and `slowest 11.5 s` on the two smoke games.

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
network, over `logs/Epoch Mate vs Edward.pgn`:

| | old | new |
| --- | --- | --- |
| overshoot mean / median / max | 1.78x / 1.85x / 3.35x | 0.71x / 0.79x / 1.36x |
| worst single move | 13.4 s, 84% of hard | 4.5 s, 45% of hard |
| clock after 20 of our moves | 16.5 s | 71.7 s |

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
