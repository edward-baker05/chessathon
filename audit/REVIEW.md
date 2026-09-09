# Current-engine review and improvement plan

2026-09-09. Reviewed the current checkout only. No git history, prior experiment notes,
snapshots, old game logs, or old checkpoints were consulted. The training metadata was
read after you specifically identified it. Production source and weights remain unchanged;
their SHA-256 fingerprints are in [source-hashes.json](/home/edwardb/Documents/chessathon/audit/source-hashes.json).

**There are real defects to fix, but the measurements do not establish that fixing them
recovers 500 Elo.** The foundation is substantially sound. The strongest immediate finding
is accidental loss of persistent game state. The larger strength project should investigate
search decisions and evaluation on difficult positions reached in play, using controlled
matches to decide what ships. Buying training compute is not the first step.

## What was reviewed and checked

Read all eight production Python modules, the training/extraction/quantization and measurement
tools, Makefile/configuration, the unchanged harness, and baseline agents. Verified the current
[agent contract](https://aichessathon.com/docs/agent-contract.md) and
[competition rules](https://aichessathon.com/docs/rules.md) directly: 120 s + 0.5 s, one CPU core,
2 GB memory, 90 s initialization, 50 MB uncompressed submission. Offline engine labeling is
allowed; a third-party engine or its lookup evaluations must not be shipped. Stockfish was used
only for offline experiments. All added files live under `audit/` and `tests/`, outside the
packager's production file selection. No harness source was changed.

Fresh results:

* **3,000 positions and 50,009 legal transitions** matched python-chess for legal move sets,
  every encoded state field, mailbox, and incremental Zobrist key. Incremental NNUE accumulators
  matched full rebuilds, and packed training features matched runtime features. All four move
  flags were exercised, including every promotion type when generated.
* Perft passed: start position depth 5 = **4,865,609**; Kiwipete depth 4 = **4,085,603**;
  the rook/pawn endgame fixture depth 4 = **43,238**. These are broad checks, not exhaustive proofs.
* The shipped network's conservative accumulator/output overflow bounds passed. Its forward
  pass matched an independent int64 reference exactly on all 43 benchmark positions.
* The current missing test suite was replaced only with this new audit suite: **5 passing test
  cases and 10 explicitly marked expected failures** reproducing current defects/approximations.
  The large randomized check is one passing case. Expected failures are not fixes.
* Ruff and strict mypy pass. `make gate`, with `CHESSATHON_INCREMENT_MS=100` to match its actual
  fast-game increment, passed both smoke games by checkmate. This establishes basic execution,
  not playing strength.
* A fresh process imported in **46.27 s**, with **975,216 KiB peak RSS** on this Ryzen 5 3600.
  This is below the numerical platform limits locally; it does not certify EPYC init time.
* A freshly packaged archive contained the eight production modules plus `weights/net.npz`:
  **646,371 bytes uncompressed**. The existing `submission.zip` was not overwritten or uploaded.

Details: [tests](/home/edwardb/Documents/chessathon/tests/test_fresh_audit.py),
[test output](/home/edwardb/Documents/chessathon/audit/tests.txt),
[import measurement](/home/edwardb/Documents/chessathon/audit/import.txt),
[gate output](/home/edwardb/Documents/chessathon/audit/gate.txt).

## Findings, in priority order

### 1. Continuing games are incorrectly reset after opponent double-pawn moves

**Confirmed; fix first.** [agent.py:104](/home/edwardb/Documents/chessathon/agent.py:104)
compares `board.ep_square` literally. The harness serializes with `board.fen()`, whose default
en-passant field includes only legal captures. A board produced by `push()` retains the raw
double-push target even when no capture is possible.

Reproduction: after our `e2e4` and their `e7e5`, the simulated opponent reply has `ep_square=e6`;
the incoming FEN has `ep_square=None`. `_continues_our_game()` returns false. `_track()` clears
the TT, ordinary/continuation histories, killers, counters, and repetition record. This affects
routine opening play, not just exotic positions.

In an eight-game, color-paired pilot at 65,536 counted nodes per move, normalized comparison
eliminated **17 false resets: 17 → 0**. Game score was **+1 =5 −2** for the fix, so this small
pilot **does not establish an Elo gain**. It does establish the unwanted state destruction.
The pilot used independent per-player Work/TT state and unchanged search/weights, but one
process and a node budget; it was not a platform-clock benchmark.

Fix: normalize the EP component by legal capturability, or remove cross-game detection from
the production interface because the contract already starts a fresh process per game.
Keep any multi-game local-test reset as an explicit operation. Preserve both colors' actual
positions in repetition history. Test opponent double pushes with absent, pinned, and legal EP.

### 2. Quiescence does not recognize important terminal draws

**Confirmed.** [search.py:427](/home/edwardb/Documents/chessathon/search.py:427) handles checkmate
when no check evasion exists, but non-check nodes can return stand-pat without detecting
stalemate. Recursive qsearch calls also lack repetition, insufficient-material, and fifty-move
checks. The checks in `negamax` protect entry through that function; they do not protect every
position subsequently reached inside qsearch.

Examples from the shipped network:

| Position | Correct result | Current qsearch |
|---|---:|---:|
| `7k/5K2/6Q1/8/8/8/8/8 b - - 0 1` | Stalemate, 0 | −3256 cp |
| `8/8/8/8/8/5k2/8/R6K w - - 100 1` | Fifty-move draw, 0 | +1641 cp |
| `8/8/8/8/8/5k2/8/6BK w - - 0 1` | Insufficient material, 0 | +217 cp |

Separately, [search.py:514](/home/edwardb/Documents/chessathon/search.py:514) tests the fifty-move
clock before checkmate, so a mating position at halfmove 100 is incorrectly valued as a draw.
The material-draw helper also omits same-color bishop-only dead positions.

Fix terminal adjudication consistently, with mate taking precedence. Add an efficient legal-move
existence check where stalemate is possible; measure its cost rather than generating all moves
at every leaf by default. Add search-through-transition tests as well as direct terminal tests.

### 3. Promotion gains are absent from SEE and qsearch delta pruning

**Confirmed.** [position.py:398](/home/edwardb/Documents/chessathon/position.py:398) values a
promoting mover as a pawn and never credits promotion material. An uncontested queen promotion
gets SEE 0 instead of +800 under the engine's own piece values.
[search.py:463](/home/edwardb/Documents/chessathon/search.py:463) gives an empty promotion target
only the default 100 cp capture gain, so delta pruning can discard a decisive promotion.

On `8/k1P5/2K5/8/8/8/8/8 w - - 0 1`, stand-pat is 1874. With alpha 2175, qsearch returns 1874
with pruning and 2734 without it. The latter is not a trusted chess score: `c8=Q` actually
stalemates, also demonstrating the terminal bug. The comparison proves the promotion branch
is discarded by a bound that omitted its gain.

Fix SEE's initial gain and arriving piece; make promotion-aware delta bounds or exempt
promotions. Quiet promotions currently receive quiet-history ordering, while capture promotions
are ordered using the pawn attacker. Give promotions explicit ordering. The capture generator
also includes queen promotions only: test underpromotions that avoid stalemate or give a uniquely
useful check, especially after fixing terminal scoring.

### 4. Quiet history is trained on moves that were not searched

**Confirmed by code and a capture-penalty test.**
[search.py:391](/home/edwardb/Documents/chessathon/search.py:391) calls its end index `quiet_end`,
but it is an index into the entire selected move list. Negative updates hit earlier captures,
pseudo-legal moves rejected for king safety, and moves skipped by pruning, as well as searched
quiets. This supplies false evidence to ordinary and continuation histories. Their piece/to
entries can subsequently influence genuinely quiet moves.

The cutoff classification `victim < 0` also treats EP and quiet promotions as quiet history
events, whereas other search paths use different classifications. Ordinary history uses
unbounded negative additions and occasional whole-side halving; continuation history clips.
These saturation dynamics are additional tuning concerns, not separately measured bugs.

Fix with a compact list of actually searched legal quiets, a shared move classification, and
bounded updates. Test a cutoff preceded by a capture, an illegal move, and a pruned quiet.
Only the searched legal quiets should receive a malus.

### 5. Root moves do not record the moved piece for continuation history

**Confirmed.** [search.py:774](/home/edwardb/Documents/chessathon/search.py:774) writes `played[0]`
but not `moved_piece[0]`. That slot remains zero after table clearing, so root king, knight,
bishop, rook, and queen moves are attributed to a pawn in continuation scoring/updates below
the root. Add the same assignment used in `negamax` and qsearch. The king-only reproduction
fails today. This fixes the data semantics; isolated match testing must still decide strength.

### 6. Root aspiration failures continue searching the remaining moves

**Measured inefficiency.** Once `local_alpha >= beta`,
[search.py:794](/home/edwardb/Documents/chessathon/search.py:794) does not break the root loop.
The aspiration attempt will have to widen regardless. On the 43-position suite at the largest
node budget, **5.15% of all counted nodes** were spent after this condition already held.

This is not an inverted-window bug: subsequent root searches still use valid null windows.
It is unnecessary work for establishing the aspiration failure, though that work can also
populate useful TT/history entries. Breaking immediately is therefore an experiment, not a
guaranteed 5.15% strength improvement. The measured move-quality effects were mixed.

### 7. SEE uses geometrical rather than legal recapturers

**Confirmed approximation with a pruning consequence.** In
`4k3/4n3/8/3p4/2B5/8/8/K3R3 w - - 0 1`, `Bc4xd5` wins a pawn: the knight on e7 is pinned to
the king by Re1. SEE returns **−230**, counting the illegal knight recapture, rather than +100.
Since qsearch drops moves with negative SEE, this can suppress a useful capture, not merely
misorder it. Pin-aware SEE or conservative exemptions should be tested against their cost.
A separate defended-king example passed; I did not count that initial suspicion as a defect.

### 8. Repetition keys distinguish an illegal en-passant right

**Confirmed but less frequent than finding 1.**
[position.py:83](/home/edwardb/Documents/chessathon/position.py:83) checks pawn geometry, not
whether an EP capture leaves its king safe. A pinned EP pawn can therefore change the hash
although the legal position is identical without that right. Test fixture:
`k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1`.
Normalize game-history keys correctly; assess whether sharing that normalization with every
incremental TT update is worth its runtime cost.

### 9. Some diagnostic numbers and existing measurement tools are misleading

* `negamax` counts a node, then qsearch counts the same entry again. At 1,048,576-node budgets,
  **18.57% of reported nodes** were these duplicated entries. This is primarily a measurement
  and node-budget definition issue. It does not imply an 18.57% wall-time speedup from deleting
  a counter increment. Keep node definitions fixed across A/B experiments, or use real time.
* **25.74% of nodes** were beyond the last committed iteration at the node cap. This is normal
  in part for iterative deepening. It motivates measuring completed first-root-move information
  and time allocation, rather than assuming every discarded partial iteration is avoidable.
* [tools/margins.py:56](/home/edwardb/Documents/chessathon/tools/margins.py:56) calls
  `clear_tables()`, which does not clear the TT, then performs `search_value` before the move
  test. Its move test inherits that search's TT/history work. This confounds independent
  fixed-node comparisons; changed moves and fewer nodes are not strength measurements.
* [tools/replay.py:46](/home/edwardb/Documents/chessathon/tools/replay.py:46) calls `search.think`
  directly and does not exercise agent game tracking or supply real repetition history.
  Its self-play fallback shares one clock between both sides. It is not a faithful integration
  or two-clock time-management test.
* The harness's fast increment is not automatically forwarded into `CHESSATHON_INCREMENT_MS`.
  Set that environment variable in the external test launcher. Do not edit the harness.
* `tests/bench.py` and `tests/match.py` remain absent; `make bench`/`make ab` are consequently
  not available. `make data` is mentioned but has no corresponding recipe. The trainer's
  `--seed` controls NumPy ordering but does not seed Torch initialization. Record full dataset
  and checkpoint provenance for new training runs; the existing metadata describes extraction,
  not which checkpoint/epoch produced this particular network.

### Other search risks worth testing, not claimed Elo losses

Early static/null pruning can return before establishing that a legal move exists. TT scores
do not include halfmove-clock or repetition-context information, so near-draw reuse deserves
targeted tests. Reductions and quiet pruning lack explicit protection for giving check, while
LMR can reduce quiet check evasions. Shallow null pruning has no verification. These are
selectivity tradeoffs; disabling them together would be a poor experiment.

## Fresh strength and performance evidence

Generated 43 positions from eight manually specified mainstream openings and fresh Stockfish
19 play. Early play randomly selected among up to three moves within 70 cp at 10,000 nodes;
later play selected its best move. Sampled several game phases. No previous games were used.
Each candidate/position/budget started with empty TT and histories. Three source-identical
copies except for the specified root changes were instrumented outside the production tree.

Stockfish selected a reference move at one million nodes. Every distinct candidate move and
that reference move were then evaluated separately with a fresh hash and 500,000 nodes,
restricted to that move. Loss is the difference from the best of those evaluated choices.
It is a noisy offline reference, not an exact game-theoretic value or a calibrated Elo measure.

| Counted node budget | Baseline median depth | Baseline mean loss | Root beta break | Beta break + root piece context |
|---:|---:|---:|---:|---:|
| 16,384 | 7 | 12.93 cp | 12.51 cp | 11.40 cp |
| 65,536 | 10 | 11.12 cp | 13.77 cp | 10.26 cp |
| 262,144 | 13 | 9.86 cp | 11.49 cp | 8.19 cp |
| 1,048,576 | 16 | 9.19 cp | 9.14 cp | 13.72 cp |

Neither variant consistently wins. Median loss is zero for almost every cell. The baseline
has no >=100 cp losses above the smallest budget. **This suite is too easy and too correlated
within eight openings to rank strong candidates confidently.** A 64× node increase only
reduces its mean loss by 3.74 cp. That is evidence to improve the test distribution, not proof
that more search is useless or that the network is the only bottleneck.

Baseline median reported throughput was about **807,000 nodes/s** locally. Measurements are
exploratory: some diagnostic processes overlapped on this machine, CPU frequency was not pinned,
and the target CPU differs. Do not interpret between-run wall times as a clean speed A/B.
Fixed-node move comparisons are independent of that timing noise.

Disassembly/IR confirms SEE allocates a 32-element gain array per call. A small warmed
start-position microbenchmark measured approximately 59 ns for quiet SEE, 87 ns for copy-make,
95 ns for move generation, and 265 ns for NNUE update plus forward. These are context-specific
microbenchmarks, not additive whole-search cost percentages. They make threshold SEE,
avoiding duplicate SEE work, and delaying evaluation until needed reasonable experiments.
They do not justify rewriting the position representation first.

Static NNUE/reference disagreement increased in lower-material positions in this sample
(about 58 cp mean absolute disagreement with 25–32 pieces versus 240 cp with 8–16 pieces).
This compares a static evaluator with a searched, potentially differently scaled Stockfish
19 score and is confounded by tactics and sample composition. It only motivates a properly
quiet, phase-stratified leaf-evaluation test. It does not prove a training bug.

Raw evidence: [positions](/home/edwardb/Documents/chessathon/audit/positions.json),
[summary](/home/edwardb/Documents/chessathon/audit/summary.json),
[reference evaluations](/home/edwardb/Documents/chessathon/audit/reference.json),
[tracking games](/home/edwardb/Documents/chessathon/audit/tracking-match.json),
[diagnostics](/home/edwardb/Documents/chessathon/audit/diagnostics.json).

### Fresh games against an external reference

Played both colors from the same newly specified Catalan opening, with the current engine
at **30 s + 0.2 s** and Stockfish 19 at **100,000 nodes per move**. The engine's actual Python
entrypoint and game tracking ran; all moves were legal and neither game flagged. Outcomes:
one loss and one draw. These are two diagnostic games at unequal resource policies, not an
estimate of either engine's competition rating. The referee checked python-chess outcomes
with draw claims; reference analysis time was outside the agent's clock.

Each agent move was compared with a fresh 300,000-node Stockfish search and a separately
searched restricted choice. The draw ended by repetition. In the loss, the position slipped
from approximately level to −123 cp by move 40, then worsened through several smaller errors.
At moves 43, 45, 47, and 48, the initial reference estimated losses of 33, 41, 67, and 84 cp.
The late 8,911 cp “error” was already in a lost rook ending and involved the artificial mate
score sentinel; it should not drive optimization priorities or inflate average centipawn loss.
Finite-node reference searches can disagree even when the chosen move is the same, so small
individual losses need confirmation.

This small sample motivates **endgame decisions, evaluation calibration, and effective search
depth** more than a claim of ubiquitous early tactical blunders. It cannot establish how often
these failures occur against leaderboard opponents. The early errors were retained for a
larger-budget re-search, with a stronger per-choice reference, as a development set only.
See [fresh games](/home/edwardb/Documents/chessathon/audit/reference-games.json) and
[error scaling](/home/edwardb/Documents/chessathon/audit/error-scaling.json).

The follow-up re-searched eight early errors with fresh TT/history at 65,536, 262,144,
1,048,576, and 4,194,304 counted nodes, and re-evaluated every candidate with Stockfish at
one million nodes. Mean reference losses were **10.0, 41.9, 34.5, and 38.0 cp**, respectively;
median completed depths were **9.5, 12.5, 15.5, and 20**. At move 43, the small search chose
`Kc5`; the three larger searches preferred `Ke3`, evaluated 37 cp worse. At move 48, the small
search chose `Ke3`; the larger searches remained around 97–98 cp worse. Other positions did
improve with deeper search.

These positions were selected for errors under the original policy, so the apparent advantage
of the smallest budget is selection-biased. The searches also deliberately lacked the live
game's cached history. **Do not infer that reducing search globally would improve Elo.** The
useful conclusion is narrower: increasing raw search budget alone does not reliably resolve
these particular failures. They are concrete starting points for distinguishing bad pruning,
misleading leaf evaluation, and threats outside the effective horizon.

## Where to look for the larger gap

The leaderboard gap is a useful target, not a decomposition. We have not inspected the leading
engine, played it under controlled conditions, or established rating uncertainty. There is no
defensible way yet to assign “200 Elo to evaluation, 150 to pruning, 150 to speed.” My priority
order is based on testability, cost, and the current architecture:

1. **Restore correct state and search semantics.** Address findings 1–5, with regression tests
   and individual A/Bs. Pin-aware SEE and exact EP history normalization follow. This creates
   a dependable baseline and removes avoidable failure modes; the pilot does not establish
   how much rating this will recover.
2. **Separate evaluation errors from selective-search errors.** Start with the eight retained
   positions, then expand to fresh failures. Record the engine PV and the stronger reference
   alternative. Re-search the relevant subtrees with one pruning/reduction mechanism disabled
   at a time, with fresh TT/history and bounded resource use. If the good line was cut before
   examination, investigate that pruning decision. If it was searched but systematically
   misvalued, compare its quiet leaves against new teacher labels, including king activity,
   rook activity, passed pawns, and material imbalance. A small full-width oracle checks
   implementation correctness; it does not substitute for these deeper targeted tests.
   This is the highest-value diagnostic step suggested by the new game evidence.
3. **Increase useful search per millisecond and improve ordering.** The current search eagerly
   generates and scores
   all moves, runs full SEE before trying many of them, and updates NNUE before children whose
   result may come from a TT cutoff or draw. Experiment with staged move selection, threshold
   SEE/SEE reuse, and lazy accumulator updates. Introduce a small qsearch TT only after terminal
   and bound semantics are covered. Test each at both equal nodes and equal wall time.
   Repair the history evidence, then test bounded
   history updates, continuous history-based LMR, and protections for forcing moves. Calibrate
   static pruning margins using errors observed at the actual nodes where they fire. Singular
   or other selective extensions are later candidates, after simpler changes have credible
   match results. Never rank pruning solely by reduced node count or nominal depth.
4. **Improve evaluation where it changes choices.** The shipped net is 768→512×2→1 with eight
   material-count outputs and QA 255. Feature transport and integer arithmetic passed. With
   270,411,213 extracted positions, a larger dataset alone is not a justified first purchase.
   Build an independent quiet-leaf holdout from fresh self-play, stratified by material,
   king exposure, passed pawns, imbalance, and tactical volatility. Measure calibration,
   move ranking, and float-versus-quantized disagreement. Then compare data reweighting and
   modest king-conditioned feature buckets against the same architecture trained on the same
   subset. Such features add relational capacity; the current net can represent interactions,
   so this is a capacity hypothesis, not a claim that it cannot understand king safety.
   Measure init time and peak compiler memory as well as inference speed: larger module-global
   weight arrays embedded by Numba may run into the initialization limits before the zip cap.
5. **Tune time allocation on real two-clock games.** Record time on stable and changing root
   choices, fail-low recovery, unfinished iterations, forced moves, and clock reserves by phase.
   The current policy cannot be judged from a single root benchmark. Preserve a conservative
   deadline and include Python entrypoint overhead. Test at 120+0.5 before accepting changes
   that won at short controls. Opening books are lower priority because starting positions are
   curated and unpublished; small permitted tablebases may be useful after endgame frequency
   and packaging costs are measured.

## A practical experiment sequence

**First half-day:** preserve the current source/weight fingerprint; turn the confirmed-defect
reproductions into passing tests one fix at a time. Establish correct game tracking and shared
terminal/promotion/history semantics. Re-run perft, differential state/NNUE checks, import/RSS,
and harness smoke tests. Keep performance changes separate from correctness changes so a
regression has an identifiable cause.

**Then build the selection test:** use at least 100–200 distinct fresh opening starts, played
with both colors. Mix middlegame and endgame starts and retain full PGN/FEN/time/score traces.
Use paired outcomes as the statistical unit; positions from one game are not independent
samples. Treat fixed-node tests as search diagnostics and wall-time matches as the shipping
criterion. A 200-game pilot can reject substantial regressions; small gains will usually need
more evidence. Do not treat a 51% result after 200 games as a win. Report confidence intervals
or a predeclared sequential likelihood test, including the draw model and stopping rule.

Freeze a development set and a separate final acceptance set. Mine difficult positions from
new games for development, but do not repeatedly tune against the held-out acceptance games.
Add synthetic rule/promotion/pin/zugzwang tests separately; they prove behavior, not Elo.
Use the original and repaired baseline plus at least one independently structured opponent;
Stockfish's configurable rating must not be equated to this leaderboard's rating scale.

**Run two experiment tracks within the next day:** search efficiency/selectivity changes
against frozen weights, and evaluation/data changes against frozen search. Start with inexpensive
local pilots; retain only changes that survive paired matches. Log source hash, net hash, seed,
opening set, side, node definition, actual clocks, completed depth, failures, and confidence
intervals. Confirm promising combinations separately, because gains need not add.

**Before the Sep 11 11:00 upload close:** reserve several hours for target-like import/memory
checks, a representative 120+0.5 match sample, packaging inspection, and platform validation.
The official validation log is authoritative. Freeze the best supported build, not the newest
experiment. If time is short, prefer a smaller number of adequately tested candidates over
many tiny, inconclusive sweeps.

## Local training capacity and cloud decision

The metadata reports **270,411,213** extracted positions. A short run of the actual current
training pipeline on the RTX 2060 processed 262,144 in-memory records in about **0.47 s** after
warm-up, roughly **560,000 positions/s**, at batch 16,384. Peak Torch allocation was about
489 MB. It performed forward/backward, AdamW, clamping, CPU feature unpacking, and transfer;
it saved no weights. See [train-probe.json](/home/edwardb/Documents/chessathon/audit/train-probe.json).

A straight throughput extrapolation is about **8 minutes per full-data epoch**, or roughly
80 minutes for ten epochs, before validation, checkpointing, whole-file I/O, and any larger-model
cost. This is a small cached-data throughput probe, not a measured full training run or a claim
about time to convergence. Benchmark a full epoch and a candidate feature layout locally first.

**No cloud spend is justified yet.** Rent compute only if a pilot already shows a credible
strength improvement and measured local throughput cannot complete the required training and
validation before the deadline. Fresh engine labeling or more independent match workers may
be a better use of money than retraining the current architecture on more similar positions.
The next proposal should specify the candidate, positions/epochs, measured throughput, total
runtime, provider quote, and a hard spending cap before purchase.

## Reproduce

From `/home/edwardb/Documents/chessathon`:

```sh
CHESSATHON_DEBUG=0 .venv/bin/python -m pytest tests/test_fresh_audit.py -q -rx -s
.venv/bin/python audit/build_variants.py
CHESSATHON_DEBUG=0 .venv/bin/python audit/measure.py bench --engine /tmp/chessathon-audit-variants/baseline --out audit/bench-baseline.json
CHESSATHON_DEBUG=0 .venv/bin/python audit/measure.py bench --engine /tmp/chessathon-audit-variants/root-beta --out audit/bench-root-beta.json
CHESSATHON_DEBUG=0 .venv/bin/python audit/measure.py bench --engine /tmp/chessathon-audit-variants/root-context --out audit/bench-root-context.json
.venv/bin/python audit/measure.py judge --stockfish /path/to/stockfish
CHESSATHON_DEBUG=0 .venv/bin/python audit/tracking_match.py
.venv/bin/python audit/summarize.py
CHESSATHON_INCREMENT_MS=100 make gate
```

The saved `positions.json` is the fixed suite for comparisons. `measure.py generate` deliberately
creates a new suite and should not be run between candidate measurements. Stockfish must remain
an external offline test executable. The synchronous UCI client exists because this sandbox's
socket restrictions prevented python-chess's threaded asyncio client from waking correctly.
