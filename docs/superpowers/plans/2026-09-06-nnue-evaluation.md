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
| `tools/calibrate.py` | Fits `SCALE` against the material eval | no |
| `tests/test_nnue.py` | Differential, mirror and overflow tests | no |

---

## Phase 1: runtime, on this machine, light compute

Proves the engine can run a network correctly and fast, using a randomly initialised net.
No training strength is expected or measured here. This phase is worthless if rushed and
everything after it depends on being able to trust the accumulator.

### Task 1: `nnue.py`, weights, features and forward pass

**Files:** Create `nnue.py`, `tests/test_nnue.py`

- [ ] Load `weights/net.npz` at import. Raise with a readable message if it is absent or
      any array shape disagrees with the compiled constants. No fallback.
- [ ] `L1`, `QA`, `QB`, `SCALE`, `BUCKETS` are module constants read from the npz before
      the jitted functions are defined, so numba compiles them as literals and the loop
      bounds are known at compile time.
- [ ] `feature(perspective, colour, piece, square) -> int32`, matching the spec exactly.
- [ ] `refresh(acc, ply, state, mail)` rebuilding both perspectives from the mailbox.
- [ ] `forward(acc, ply, state) -> int32`, SCReLU, int32 accumulation, bucket by popcount.
- [ ] Warm every jitted function at import with the real argument types.
- [ ] Test: evaluation of a position and of its colour-mirrored twin are exact negatives.
      This is the test that catches a perspective or sign error in the indexing.
- [ ] Test: `forward` after `refresh` equals a plain numpy reimplementation of the same
      arithmetic, so the jitted version is checked against something readable.

### Task 2: incremental update and its differential test

**Files:** Modify `nnue.py`, `tests/test_nnue.py`

- [ ] `apply(acc, ply, state, mail, move)` writing ply+1 from ply: one subtract and one
      add for a quiet move, plus the captured piece, the promotion piece swap, the castled
      rook and the en passant victim. Decoded from the parent state and mailbox.
- [ ] `copy(acc, ply)` for the null move.
- [ ] Test: over several hundred random playouts, at every ply, the incrementally updated
      accumulator equals `refresh` on the same position, element for element. This is the
      test the whole design rests on.
- [ ] Test: the update path is exercised for every move flag, castling, en passant,
      promotion and promotion-with-capture, rather than only whatever random play produced.

### Task 3: wire into the search

**Files:** Modify `search.py`, `evaluate.py`

- [ ] Add `acc` to `Work` as `np.zeros((STACK_PLIES, 2, L1), dtype=np.int16)`.
- [ ] `evaluate.py` exposes `evaluate(acc, ply, state) -> int32` delegating to `nnue`, and
      keeps `material_eval(state, mailbox)` for the tests and the snapshot opponent.
- [ ] Call `nnue.apply` at the five `make` sites, after `legal_after` passes; `nnue.copy`
      after `make_null`; `nnue.refresh` once per `think` at ply 0.
- [ ] Update the three `evaluate` call sites in `negamax` and `qsearch`.
- [ ] Test: a fixed-depth search from a set of positions returns the same score whether
      the evaluation came through the accumulator or through a refresh at every node.

### Task 4: measure the real cost

**Files:** Modify `tests/bench.py` if needed

- [ ] Snapshot the current engine as `snapshots/material` **before** any of this lands, so
      there is a fixed opponent to measure against.
- [ ] Record node rate before and after with a random net. Expect roughly 1.65 Mnps to
      1.30 Mnps. A larger drop means something in the path is not vectorising, and the
      cause is a non-contiguous numba type until proven otherwise.
- [ ] Record import time. Must stay well inside the 90 s budget.

---

## Phase 2: data and trainer, smoke tested here

### Task 5: `tools/extract.py`

**Files:** Create `tools/extract.py`, add a `data` target to the `Makefile`

- [ ] Stream the `.zst` without decompressing it to disk. Parse one JSON line per position.
- [ ] Take the deepest eval per position. **Negate `cp` when Black is to move**, because
      the file is White-relative. This was determined empirically, not assumed.
- [ ] Apply the spec's filters: not in check, best move not a capture, depth at least 12,
      `|cp| < 10000`. Mates saturate to +/- 12800.
- [ ] Write a packed binary, about 28 bytes per position, plus a small JSON sidecar
      recording counts, the filter settings and the source file's date, so a trained net
      can be traced to the data that produced it.
- [ ] Run on a small slice here to confirm correctness and throughput. The full pass is a
      desktop job.

### Task 6: `tools/train.py`

**Files:** Create `tools/train.py`, add a `train` target to the `Makefile`

- [ ] Device selection: CUDA, then MPS, then CPU. The desktop has an RTX 2060; this
      machine has MPS. The same script must run on both without edits.
- [ ] `768 -> L1 -> 1` per perspective with shared input weights, 8 output buckets,
      SCReLU, MSE in probability space against `sigmoid(cp / 400)`.
- [ ] Clamp output weights during training so the export-time overflow bound can pass.
- [ ] Checkpoint every epoch to `data/checkpoints/`, with a held-out loss logged per epoch.
- [ ] Smoke test here on the small slice: loss must fall, and the net must beat
      `snapshots/material` on a fixed-node match by a visible margin. A short run on a
      small slice will produce a weak net, and that is the expected result. What is being
      tested is that the pipeline is wired correctly end to end.

### Task 7: `tools/quantise.py`

**Files:** Create `tools/quantise.py`, add a `quantise` target to the `Makefile`

- [ ] Checkpoint to `weights/net.npz` at `QA`, `QB`, `SCALE`.
- [ ] **Compute the exact worst case from the weights being written**,
      `max over buckets of sum_i (QA^2 * |w_i|)`, and refuse to write unless it is below
      `2^31`. Do the same for int16 accumulator overflow using the real maximum feature
      weight and the maximum piece count. Report both margins.
- [ ] Report quantisation error: mean absolute difference between the float net and the
      quantised net over a sample of positions.

### Task 8: `tools/calibrate.py`

**Files:** Create `tools/calibrate.py`

- [ ] Fit `SCALE` so the net's centipawn output lines up with the material evaluation over
      a shared position set, so the six centipawn pruning margins in `search.py` keep
      approximately the meaning they were tuned for.

---

## Phase 3: the real net, on the desktop

### Task 9: full extraction and training

- [ ] `make data` on the desktop against the full 21.7 GB file.
- [ ] Train to convergence. Checkpoint every epoch so a long run can be cut short at any
      point without losing the day.
- [ ] Quantise every checkpoint, so there is always a shippable net on disk.

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
- [ ] Self-play data: positions from our own engine, labelled by our own search at fixed
      depth. In-domain and entirely our own. Train v3 on the combined set, A/B against v2.
- [ ] Input king buckets, `768 x N`, if the margin sweep and self-play data are done and
      there is still time. This reintroduces an accumulator refresh path, so it needs its
      own differential test before it can be trusted.

Ship whichever version last won its match.
