# Where the missing 500–600 Elo could be

**Audit date:** 10 September 2026. **Frozen source:** `7f20dd1c0603ae3aa377227c3184835f995880bd`.

## My diagnosis

**The strongest explanation is a ceiling in the combination of evaluation and selective search, reinforced by an unreliable improvement-selection process. I did not find an unfixed move-generation, feature-encoding, or arithmetic catastrophe that plausibly accounts for the whole gap.** The aggressive route is to improve the evaluator’s training objective and representation, make search allocate work according to reliable evidence, and repair the experiments used to choose between candidates. Another round of isolated constants and bit tricks is unlikely to get you there.

This conclusion is more specific than “NNUE could be better”:

1. **The shipped network fits low-material positions substantially worse than high-material positions, even on samples from its own training corpus.** Quantization is almost irrelevant to that difference. Its training objective also provides little incentive to distinguish different ways to convert large advantages. There is no independent leaf-distribution validation in the frozen training pipeline.
2. **Additional search has strongly diminishing returns on some actual mistakes.** In twelve deliberately difficult positions from your tournament games, increasing the budget from 262,144 to 8,388,608 nodes repairs several errors, but four positions still have at least an 80 cp deficit against the reference choice. The engine can reach nominal depths 20–26 and preserve the wrong preference.
3. **The improvement-selection tools contain consequential defects.** Nominal fixed-node matches still obey short clocks; the offline reference client can treat unfinished-search bounds as exact evaluations; the default opening set loses its source-game clustering. These can steer development toward the wrong architecture or search policy. These defects do not directly change the uploaded engine’s moves; they make improvement claims unreliable.
4. **The search spends work eagerly, then relies on aggressive pruning to recover depth.** Full-list generation and scoring, repeated full SEE, eager accumulator updates, weak use of history in reductions, and no selective extension other than check are a plausible substantial efficiency ceiling. The evidence does not yet establish a 2× speedup, and a speedup alone would not repair the persistent wrong preferences.

These mechanisms can interact: a misleading static evaluation orders the wrong move first, licenses pruning of alternatives, and receives more apparent confirmation as iterative deepening follows the same path. Treating the evaluator, ordering, and pruning as unrelated tuning projects misses that feedback loop.

**I cannot honestly allocate 600 Elo among these causes.** We do not have the leading engine’s source or a controlled match against its frozen build. Under the usual logistic Elo model, a 500–600 point deficit implies only about 5.3%–3.1% expected score against that opponent. Consistently losing contested positions is sufficient; it does not require a spectacular blunder every game. The remaining sections distinguish confirmed observations from hypotheses and give experiments that could falsify the hypotheses.

## Scope, isolation, and evidence

I made a `git archive` of the specified last commit into `/tmp/chessathon-elo-audit-7f20dd1`. All execution used that snapshot, not the changing production files. I did not modify the engine, weights, harness, Git index, branch, or another agent’s work. This deliverable occupies its own `audit/elo-gap-7f20dd1` directory. Source and probe copies use `.txt` suffixes so they do not add Python modules or lint targets to the active project.

The [complete source archive](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source-7f20dd1.tar.gz), [source hashes](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source-hashes.json), and linked source excerpts preserve the actual version reviewed. Any later changes by the other agent require a recheck before applying a finding to the current tree.

I read all eight production modules, training/extraction/quantization code, benchmark and match code, selectivity diagnostics, existing audit tests and reports, Makefile, packaging, and referee/clock code. Historical reports were used as leads, not treated as current truth. Many of their severe defects are already fixed in this commit.

Fresh work in this audit:

| Experiment | Scope | Result / interpretation |
|---|---|---|
| Existing independent audit suite | Frozen source and weights | **77 passed**, including perft, state/hash/NNUE differential checks, SEE, terminals, and tracking regressions. |
| Tournament-game screening | 23 Edward games, rounds 51–73; 434 sampled own-turn positions | Stockfish 19 analysis identified development cases; historical builds are not assumed identical to the snapshot. |
| Current-build error scaling | Twelve selected cases; 262k, 2m, 8m, and a reduced-selectivity comparison | Some mistakes respond to more work; several persistent wrong preferences remain. |
| Training-corpus sample | 65,536 uniformly sampled records, deterministic indices | Clear material-phase difference in model fit. This is **not an independent holdout**. |
| Checkpoint/quantization verification | Compare saved checkpoint arrays to shipped arrays; float and integer inference on the sample | Exact match to epoch 30; mean float/integer difference **4.52 cp** before the output clamp. |
| Basic conversion | Six specified endings, persistent agent state, 262k nodes/move, Stockfish defender at 50k | Five checkmates; bishop-and-knight ending drawn by the fifty-move rule. |
| Node-budget reproduction | Same position, requested 200k nodes, different clocks | Time limits truncate the nominal node experiment. |

I used the local binary at `/home/edwardb/Documents/stockfish/stockfish-linux-x86-64-universal`; its UCI identity reports **Stockfish 19**. It was used only offline. Reference searches used one thread and a 32 MiB hash. No reference engine, network, or table of reference evaluations was added to the submission.

The canonical [agent contract](https://aichessathon.com/docs/agent-contract.md) and [rules](https://aichessathon.com/docs/rules.md) were fetched directly on the audit date; fetched copies are retained with the evidence. They specify 120 s + 0.5 s, one EPYC core, 2 GB RAM, 90 s initialization, and a 50 MB uncompressed submission. They permit training your own network from engine-labelled positions and permit shipped opening books/tablebases, while prohibiting shipping or translating another engine or starting from its network. The plan below extends beyond the September 11 upload close as requested.

## 1. Evaluation is the leading large-gain hypothesis

### What is actually shipped

The evaluator is **768 piece-square-colour inputs → 512 neurons per perspective → a single scalar output**, with squared clipped ReLU and eight output buckets selected by piece count. Both perspectives share the feature transformer. It has about 402,000 learned parameters. See [nnue.py](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/nnue.py.txt:67) and [training model](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/tools--train.py.txt:59).

The shipped integer arrays exactly equal the quantized arrays from `data/checkpoints/epoch030.pt`, including transformer weights, biases, output weights and output biases. The checkpoint records 269,059,156 training positions and a holdout loss of 0.008602. The shipped file itself has no provenance field; the current exporter adds one, but these weights predate that addition. This matters because `logs/train-768-shuffled.log` describes a separate 20-epoch run. **Do not infer which model is playing from a training log or the latest trainer source.** This audit established the array match directly.

On the 65,536-record sample:

- Float loss: **0.008468** in the trainer’s sigmoid-MSE space.
- Quantized loss: **0.008479**.
- Float-to-integer absolute difference: mean **4.52 cp**, 99th percentile **19.41 cp**, before the final output clamp. One sample exceeds the runtime’s ±10,000 cp clamp; including that clamp changes the mean to **4.53 cp**, without changing the conclusion.

The existing bounds and transport tests pass. This substantially weakens the theory that the large gap comes from overflow, a perspective/sign mismatch, or quantization damage. It strengthens the case for improving the function that was learned.

### The weakness is visible within the corpus, not just against a different teacher

The following compares the shipped network directly with the packed labels in the local shuffled training file. `MAE` is restricted to labels with absolute value below 1,000 cp so mate sentinels do not dominate it.

| Pieces on board | Sample share | Sigmoid MSE | MAE on finite-range labels |
|---|---:|---:|---:|
| 2–5 | 4.46% | 0.01391 | 187 cp |
| 6–9 | 9.23% | 0.01760 | 187 cp |
| 10–13 | 10.30% | 0.01403 | 150 cp |
| 14–17 | 11.91% | 0.01144 | 121 cp |
| 18–21 | 14.26% | 0.00992 | 109 cp |
| 22–25 | 17.50% | 0.00643 | 90 cp |
| 26–29 | 19.07% | 0.00370 | 67 cp |
| 30–32 | 13.26% | 0.00136 | 40 cp |

This is strong evidence of uneven fit, **not a measured Elo loss or a proof of endgame generalization failure**. Phases differ in intrinsic difficulty and label distribution. However, the explanation cannot simply be that a new Stockfish version uses a different centipawn scale: these are the corpus’s own labels.

There are millions of low-material examples in a corpus of this size. “There are almost no endgames” would be wrong. The questions are which endgames, how useful their targets are, and how much representational capacity and gradient they receive.

### Three structural shortcomings to attack

**A. The data filter is a weak approximation to the engine’s leaf distribution.** [Extraction](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/tools--extract.py.txt:238) removes positions in check and positions whose teacher’s best move is a capture or promotion. It does not establish that the remaining positions are tactically quiet. Quiet checks, threats, defenses, zugzwang, and positions containing alternative captures survive. In particular, “teacher’s best move is quiet” does not mean “our quiescence search evaluates this position correctly.”

Meanwhile, runtime evaluation occurs at static-pruning nodes as well as quiescence leaves. Those nodes are selected by this engine’s own search, including artificial null-move subtrees. Their distribution is not the same as positions people requested evaluations for in a public database. A large general database can therefore leave systematic errors precisely where pruning needs trustworthy estimates. The [Lichess database description](https://database.lichess.org/#evals) describes the evaluation records; their suitability for this engine’s leaves must be measured separately.

**B. The objective does not train decision quality or conversion directly.** [The trainer](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/tools--train.py.txt:258) minimizes `MSE(sigmoid(output), sigmoid(label_cp / 400))`. It has no move-ranking objective, played-result target, teacher-confidence weighting, or special treatment of conversion. Mate labels become ±12,800 cp. A predicted +2,400 cp corresponds to about 0.9975 under that sigmoid; distinguishing it from still larger winning scores contributes very little loss. This is a property of the loss, not evidence that sigmoid training is inherently bad.

A network can fit “winning” while providing a poor gradient for king activity, pawn promotion, or avoiding a fortress. In a finite selective search, an attractive but unconvertible advantage can beat a smaller numerical score with a concrete route to victory. Learned draw recognition and evaluation calibration near the practical win/draw boundary are especially valuable.

**C. The feature transformer has to learn board relationships indirectly.** There is no king-conditioned feature index, explicit attack/threat feature, or explicit pawn-structure input. Kings are ordinary piece-square features. The nonlinear layer **can** represent interactions; it is false that this architecture cannot learn king safety. But a small shared transformer must spend capacity discovering relationships that a better feature basis could expose directly.

Eight output buckets do not provide eight independent hidden representations. Nor does enlarging the first-layer *input vocabulary* necessarily require enlarging every accumulator update: a sparse feature lookup can add parameters while keeping a 512-wide update. The general sparse-input/accumulator principle is explained in the [official NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#consideration-of-networks-size-and-cost). That is architectural background, not a recommendation to port its implementation.

### The aggressive improvement project

Build **a stronger evaluator with the same inference budget**, then spend extra inference budget only when match results justify it.

1. **Collect actual search leaves and near-pruning nodes.** Instrument a frozen engine offline to sample static evaluations, quiet leaves, and the boundaries at which RFP/futility/null decisions fire. Retain source game, material, side, depth, window, static value, and pruning decision. Include the teacher’s preferred alternative and this engine’s preferred alternative. De-correlate by game and position family.
2. **Relabel selectively with a consistent teacher.** Spend deeper analysis on contested endings, tactical uncertainty, score instability, and move-ranking disagreements. Use the public corpus for breadth, but give these engine-specific examples meaningful weight. Keep tactical and artificial-null examples distinguishable so they do not blindly contaminate a quiet-leaf objective.
3. **Train phase-balanced candidates.** Compare the existing architecture on improved data first. Report phase-conditional calibration and ranking, not just aggregate MSE. Give endgame draw boundaries and difficult material imbalances explicit coverage. Keep a separate evaluation set from later source games that no tuning run sees.
4. **Test relational features at controlled width.** A practical first family is 8 or 16 coarse king-location buckets, optionally with a consistently defined horizontal mirror. A naive `buckets × 768 × 512 × int16` transformer occupies about **6 MiB or 12 MiB**, respectively, before other arrays. It fits the upload cap on paper. Test refresh costs when a king changes bucket, first-move compilation, and compiler memory. Reuse or factor shared features where your own implementation makes that useful. A single failed feature experiment is not enough to reject the family unless training, initialization, node budget, and wall-time strength were all controlled.
5. **Compare objectives and quantization deliberately.** Try a modest auxiliary ranking term on candidate moves or teacher PV endpoints, and a bounded centipawn/residual term where sigmoid saturation hides useful differences. If adding a material/PSQT baseline, train the network to the residual; do not bolt a second material score onto a network already trained to predict the total. Measure float-versus-exported loss and playing strength after each architectural change.

The success criterion is better choices at a fixed search budget **and** better results at the real wall-time budget. A better training loss alone is insufficient. This is a plausible hundreds-of-Elo-scale research direction; no experiment in this audit measures that gain yet.

## 2. Search depth is overstating the amount of reliable chess examined

### What the current build does on historical failures

Screening selected every third fullmove, from moves 12–80, on Edward’s turn in 23 archived games. There were 434 positions. The first teacher pass used 100,000 nodes per unrestricted search and per differing played choice. This initial pass was used only to discover candidates. It used the snapshot’s UCI client, whose bound-handling defect was found during final verification, so its apparent losses must not be used as an error-rate estimate. The historical builds also differ from the snapshot.

I selected twelve large screened errors, at most two per game, and ran the frozen build from empty TT/history at four policies. Every resulting move, the original choice, and reference alternatives—including new choices found by the targeted ablations below—were then independently searched by Stockfish at one million nodes with cleared hash. **All numerical comparisons reported below were recomputed with a corrected client that retains the last completed exact-score report and rejects upper/lower bounds from unfinished iterations.** “Exact” describes the UCI bound type at that completed depth; it does not mean the true chess value is known. “Loss” below is the difference from the best of those tested choices, not a game-theoretic value.

| Frozen-build policy | Median completed depth | Mean reference loss | Cases losing at least 80 cp |
|---|---:|---:|---:|
| 262,144 nodes | 13 | 78.4 cp | 5 / 12 |
| 2,097,152 nodes | 18 | 64.5 cp | 5 / 12 |
| 8,388,608 nodes | 22 | 47.6 cp | 4 / 12 |
| Eight selective mechanisms disabled, 2,097,152 nodes | 7 | 60.3 cp | 3 / 12 |

The last policy still retains TT cutoffs, check extensions, and qsearch pruning. It is a comparison policy, not an exact oracle.

**More work matters.** For example, in the round-54 position below, the engine needs the larger allowance to stop choosing a drawing king move and find the reference’s winning bishop move. However, more work is not sufficient across the set. A 32× budget increase is an enormous engineering target, and it still leaves substantial errors.

These cases were selected for past errors, so the table must not be presented as a general strength curve. Also, the test is cold-root analysis: it does not recreate each historical game’s TT or repetition state. The point is to identify current reproducible weaknesses, not reconstruct an exact historical loss.

### Four concrete cases worth following into the tree

These headline move comparisons were additionally rechecked with **four million reference nodes per choice**, using completed exact-score reports.

**Winning attack missed until eight million nodes.**

`5r1k/1p6/p2p1n2/P2P1pbq/3P1p2/5R1p/7P/5QRK b - - 5 36`

At 2m nodes the engine reports depth 22, approximately +358 cp, and chooses `h8g7`; the reference restricted to that move scores **0 cp**. At 8m it chooses `g5h4`, scored **+192 cp** by the reference. The smaller search is not just uncertain: it is very optimistic about the wrong continuation. Investigate how that optimism is sustained and when the bishop move’s tactical point enters the tree.

**A practical draw-defense preference survives a huge budget.**

`4r1k1/1r4p1/7p/8/7P/3bBP2/6P1/4RNK1 w - - 1 51`

The engine chooses `f1g3` at 262k, 2m, and 8m nodes; the two larger searches report depth 24 and around −141 cp. The reference scores that choice **−147 cp**, while `e3f2` receives **0 cp** at its finite budget. The lower-selectivity comparison also chooses `f1g3`. This is a particularly good evaluation/distribution case; globally switching off pruning does not cure it at the tested allowance. The reference’s zero is evidence of a defensive resource, not proof of an exact fortress.

**A persistent missed castling resource.**

`r4r2/pb4kp/1p3qp1/1Bpp3P/4p3/P1P1P3/1PQ2PP1/R3K2R w KQ - 1 21`

At 2m and 8m the engine chooses `c2e2`, scored **−95 cp** by the reference, while `e1c1` scores **−3 cp**. The engine’s own score stays close to equality. The castling TT encoding bug from the old audit is already fixed, so invoking that old bug does not explain this result. Follow the actual castling line and its leaf evaluations.

**A search-policy case where shallow, broader work helps.**

`1r2r3/1q3pbk/Rp4p1/1Qp1pn1p/2N5/3PBPP1/1P2P1KP/2R5 b - - 2 27`

The normal search chooses `e5e4` at every tested node allowance; the reference scores it **−92 cp**. The lower-selectivity policy chooses `f5e3`, scored **−7 cp**. This is a direct reason to investigate selectivity and move allocation, although it is not evidence that removing all pruning is globally stronger.

Full FENs, chosen moves, scores, depths, TT-followed lines, and reference PVs are in [current-failures.json](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/current-failures.json). TT-followed lines are diagnostic continuations, not guaranteed exact completed PVs.

### Targeted ablations narrow the search hypothesis

I also ran nine policies at 2,097,152 nodes on four stubborn cases: remove each of the eight selective mechanisms individually, then remove RFP/futility/LMP together. The network and all other code stayed frozen.

In the round-64 move-27 position immediately above, removing normal-search SEE pruning selects `h5h4`; removing LMR selects `e8e6`. The stronger reference scores these **+18 cp** and **+6 cp**, against **−92 cp** for the baseline’s `e5e4`. This is a substantial, reproducible choice improvement in one contested position, not a measured Elo gain. The SEE switch does not disable qsearch’s SEE filter, and this does **not** prove an arithmetic bug in SEE or establish which descendant was wrongly pruned.

In the round-69 move-51 ending, every one of those policies still selects `f1g3`. In the round-59 move-36 case, all still select `b5b6`, which the one-million-node exact-score reference values at +80 cp against +188 cp for `d2e3`. The castling case changes some preferences under ablation but never finds castling at that budget. Those are strong reasons not to expect one global pruning switch to recover the missing strength.

See [selective-ablation.json](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/selective-ablation.json) and [four-million-node rechecks](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/exact-reference-4m.json).

### Why this search can lock onto the wrong answer

The frozen search uses RFP through depth 8, razoring through depth 3, futility through depth 6, and late-move/SEE pruning through depth 8. The margins were originally chosen with a material evaluator. RFP uses 75 cp per ply; futility uses `100 + 90 × depth`. The neural evaluator now has substantial structured error, particularly in endings. The code explicitly acknowledges the calibration mismatch. See [margin definitions](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/search.py.txt:119) and [pruning](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/search.py.txt:778).

Important interactions:

- Quiet pruning decides **before making the move**, so it does not know whether the move gives check or creates a forcing threat.
- LMR has no explicit `not checked` gate: it can reduce later quiet evasions. It also does not test whether the move gives check. A resulting check extension may offset part of a reduction, but that is not equivalent to deliberately protecting forcing moves.
- History values affect quiet ordering, but reduction size barely uses them: it recognizes special killer/counter score bands and `improving`, not a continuous measure of ordinary/continuation-history confidence.
- A reduced move is re-searched if it beats alpha. This is necessary but does **not** repair a false fail-low caused by the reduced search itself.
- Internal iterative reduction removes a ply when no TT move exists, without a subsequent “recover that ply if needed” step.
- Verified null move is used only at depth 10 or above. The pawn-only exclusion does not cover every zugzwang-prone low-material position.

None of these observations means each heuristic is a bug. Strong selective search needs approximations. The problem is applying several approximations using weak evidence and interpreting the resulting depth number as assurance.

### How to attack this aggressively

**Measure false pruning decisions at the nodes where they occur.** Sample suspected cutoffs, then re-search their alternatives offline with larger resources and independent reference analysis. Record whether the good move was generated, ordered late, discarded, reduced into a false fail-low, or searched and misvalued. That classification tells you whether to fix allocation or evaluation.

Test **coherent policy families**, not just one constant at a time: endgame-aware pruning, protected forcing moves, and confidence-based reductions. Include additive/subtractive and grouped ablations because mechanisms can mask each other. Keep equal-node diagnosis and equal-wall-time selection separate.

Specific implementation candidates are safer RFP/futility margins tied to measured error and phase; history-based LMR with explicit tests for checking moves/evasions; better capture ordering/history; a conservative learned correction to static evaluation from reliable quiet-search outcomes; verified low-material null pruning; and a narrowly implemented singular extension for credible, well-supported TT moves. Begin with the families that repair the recorded failures. Do not add further pruning simply because it increases nominal depth.

The existing `tools/failures.py` diagnostic also needs a conceptual correction: it counts cases where the less-selective engine’s **own score is higher**. A higher score is not necessarily a better move. It misses overoptimistic failures in the opposite direction and shares the same evaluator and qsearch blind spots. Its published “survival rate” cannot certify that selectivity is costing only a tiny amount of strength. See [failure comparison](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/tools--failures.py.txt:128).

## 3. Repair the experiment pipeline before believing another strength verdict

### Confirmed: “fixed nodes” still means “nodes or time, whichever stops first”

[The match launcher](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/tests--match.py.txt:178) defaults to 10,000 ms + 100 ms. Setting `--nodes` sets an environment variable but does not remove those clocks. [agent.py](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/agent.py.txt:75) forwards the node limit into `think`, and [_prepare / search_root](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/search.py.txt:1261) still install and enforce soft, stretch, and hard time limits. The untouched referee also continues deducting wall time.

On the same middlegame with a requested 200,000-node budget, using zero increment to isolate the clock effect, the fresh probe returned:

| Clock passed to `think` | Actual nodes | Completed depth |
|---|---:|---:|
| 10,000 ms | 89,185 | 10 |
| 5,000 ms | 50,407 | 9 |
| 2,000 ms | 12,538 | 7 |
| 1,000 ms | 8,667 | 6 |
| 3,600,000 ms | 200,705 | 10 |

The small overshoot of 200k follows the 2,048-node checking cadence. Runtime and therefore clipping amounts vary with the host; the mechanism does not. These specific timings came from low-priority audit execution with other work present, so they are **not** a platform speed benchmark.

Consequences:

- A slower, better evaluator can lose a supposed equal-node match because it receives fewer nodes.
- A faster but less effective search can appear to win an algorithm comparison because clocks stop the other side.
- Later phases can be tested at a different node allowance from earlier phases.
- Parallel “fixed-node” games can still be affected by contention because their clocks remain active.

**Fix in the testing layer, without editing `harness/`:** give true node-budget diagnostic games a deliberately nonbinding base clock; assert and log the per-move stopping reason and actual node count. Mate termination and maximum depth are valid early stops; hidden soft-clock stops are not. Separately run real two-clock matches to measure the speed-quality tradeoff.

### Confirmed: the offline reference client mistakes bounds for evaluations

[The frozen UCI client](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source/audit--uci.py.txt:65) replaces its saved result whenever it sees an `info ... pv` line. It does not reject `lowerbound` or `upperbound`. When a finite-node search stops during an aspiration re-search, its last PV report can be only a bound. Subtracting two such reports as if they were exact values invents a precise “centipawn loss” that the searches did not establish.

In the 32 distinct move judgements rerun for this audit’s selected cases, **26 final reports were bounds**. For example, the initial one-million-node reports for both `c2e2` and `e1c1` in the castling case were explicitly marked `lowerbound`. This is a widespread reference-measurement issue in this sample, not an exotic corner case.

I corrected the offline client in a separate probe file and reran every comparison used in the report. The corrected results retain the last completed exact-score iteration, record its depth/node count, and separately report whether the final message was a bound. The broad conclusions survive, but some precise values change. Correct this before reusing previous centipawn-loss dashboards, candidate rankings, or automatically generated training labels. Do not infer that the original Lichess training corpus has this bug: its extraction path is separate.

The production engine and repository UCI client remain untouched. The corrected diagnostic implementation is [uci_exact.py.txt](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/uci_exact.py.txt). Reference budget, UCI score type, actual completed depth, and repeated-search stability should all be retained in future experiments.

### The opponent and sample distribution are also limiting progress

The frozen match launcher defaults to `audit/positions.json`, the old 43-position set sampled along only eight generated source games. Its rows encode source games in identifiers such as `g0-p14`, but have no `cluster` field; the launcher falls back to treating each FEN as its own cluster. **The default run therefore does not actually obtain the independence its confidence-interval discussion assumes.** Use the purpose-built clustered opening set or rebuild the default with source-game clusters. Cycling the same starts to reach 200 games does not create 200 independent experiments.

The old easy suite had almost no large reference losses, and comparisons against near-identical engines can end in repetitions before exposing shared weaknesses. A test process can reject improvements in king safety or endgame understanding simply because it does not ask those questions often enough.

Use a development pool of genuinely different openings, difficult middlegames, and contested endgames, with paired colours and source-game clusters. Include an independent opponent as well as the last accepted build. Keep an untouched acceptance pool. Choose candidates using match outcomes with a predeclared sample/stopping policy; do not run many small tests and keep the luckiest result. For the requested large gains, focus first on changes producing clear effects, not resolving tiny increments.

Also fix the statistical reporting detail: the launcher’s point estimate weights pairs, while its interval is centered on an unweighted average of cluster means. Those estimate different quantities when clusters differ in size. Use one declared estimand and a matching clustered uncertainty calculation.

## 4. There is useful-search headroom, but no evidence for a board rewrite

The production search calls full move generation and scores every pseudo-legal candidate before examining its first move. Capture scoring runs full SEE. Pruning can then run SEE again on the same move. SEE allocates a gain array and computes legality-related exchange information. The search updates NNUE eagerly after making each legal child, even when that child immediately returns from a TT cutoff or repetition check. Quiescence has no TT probe/store policy.

These are repeated costs across enormous numbers of nodes, not isolated Python boundary overhead. They are credible places to look for a substantial increase in **useful decisions per second**. Their cumulative benefit must be profiled; the percentages from isolated microbenchmarks do not add.

My implementation order would be:

1. **Staged move selection:** validate and try a credible TT move before scoring all alternatives; defer quiet generation/scoring where a tactical move already cuts off.
2. **Threshold SEE and reuse:** pruning asks “is this above a threshold?”, not always for an exact full exchange value. Avoid paying for equivalent SEE twice. Preserve the recently fixed pin/promotion semantics.
3. **Lazy accumulator work:** carry move deltas and materialize the accumulator only when an evaluation is actually required. Ensure king-bucket refreshes and repeated searches remain correct if the evaluator grows.
4. **Improve quiet allocation:** bounded history updates and reduction confidence, then selective extensions. This can buy better chess without increasing raw nodes/s.
5. **A measured qsearch TT and check-evasion generator**, once their cost and correctness are demonstrated on actual search nodes.

The board layout itself is compact: sixteen uint64 fields plus a 64-byte mailbox per ply. Current perft and differential tests pass. The prior independent representation experiment found no full-search wall-time improvement from its native bit-scan variant, despite a faster move-generation microbenchmark. There is no supporting evidence for rewriting copy-make into unmake or replacing bitboards.

The initialization path is a more consequential architectural constraint than the zip’s current small size. It searches for magic constants at import, eagerly compiles a large search, and embeds read-only arrays into compiled functions. Older rated logs show roughly 63–80 s initialization against a 90 s limit. Save your own generated magic constants as readable Python data and investigate passing large model arrays as data rather than embedding them repeatedly, where that reduces compiler cost without harming vectorization. Measure this on the platform.

**Do not interpret this audit’s 131 s low-priority import as a measured platform failure.** These diagnostic processes ran alongside existing work, outside the platform’s isolated scheduling conditions. It does establish that I cannot certify initialization from these runs. Architecture growth must preserve the real init and memory budgets.

## 5. Endgame behavior: distinguish a broad weakness from a rare missing technique

Fresh persistent-state conversion tests at 262,144 nodes per move produced:

| Setup | Result | Plies played |
|---|---|---:|
| KRK, defending king in corner | Checkmate | 19 |
| KRK, defending king central | Checkmate | 37 |
| KQK, defending king in corner | Checkmate | 27 |
| KQK, defending king central | Checkmate | 25 |
| Two bishops and king vs king | Checkmate | 79 |
| Bishop, knight and king vs king | Fifty-move draw | 99 |

The referee’s claimable-draw behavior explains adjudication at 99 plies: the next move can establish the claim. These are six hand-selected diagnostic games, with a finite-budget Stockfish defender, not a comprehensive tablebase-backed conversion test.

The bishop-and-knight failure is real in the tested setup but is **not a credible explanation for 600 Elo**. More important are the contested rook/minor/pawn endings above, where one inaccurate preference can turn a defendable game into a loss or sacrifice a winning opportunity.

A small legal tablebase set, or your own transparent specialized evaluation for common endings, can remove entire classes of failure. Prioritize coverage by actual game frequency and byte cost. Treat such support as a complement to better general endgame evaluation. Do not spend the main strength budget solving a rare ending because it offers an easy, exact reproduction.

## Findings already repaired, and distractions to avoid

The earlier audit’s game-reset bug after double pawn pushes, TT castling/promotion packing bug, root continuation-piece omission, promotion SEE/delta errors, several SEE pin/recapture errors, and terminal/draw defects have fixes in this snapshot. The 77-test run exercises those repairs. Recommending them as new ways to recover hundreds of Elo would double-count completed work.

Some approximations remain: queen-only promotions in non-check qsearch; path-dependent TT scores around repetition; conservative twofold handling; imperfect fifty-move context away from the protected band; and unbounded negative ordinary-history updates. They deserve correctness work or measured policy tests. Nothing here establishes them as the dominant residual rating deficit.

I would not make the main project any of the following: rewriting the board representation, changing the sigmoid divisor in isolation, blindly increasing network width, replacing search with a GPU-style policy model under a one-core runtime, adding a generic opening book without measuring coverage of the curated starts, or shaving a few percent off bit operations. Those can consume the time needed for a materially stronger evaluator/search combination.

## An aggressive sequence with concrete decision gates

### Phase A — establish an experiment you can trust

**Deliverables:** frozen baseline; true node-budget mode with stop-reason assertions; a representative clustered development set and separate acceptance set; game-level traces containing time, source/net hash, depth, node count, score, and termination.

Retain all twelve failure cases as development fixtures. Expand from new games, concentrating on the first transition out of a contested position rather than the final blunder in an already lost game. Reconfirm large reference disagreements at higher budgets before optimizing against them.

**Gate:** identical fixed-node repetitions produce identical work/choices outside explicit early-stop conditions; equal-time matches use real two-clock accounting and controlled scheduling. New metrics do not silently reward higher self-evaluation or nominal depth.

### Phase B — run two substantial, controlled improvement tracks

**Evaluation track:** leaf/pruning-node data, phase-balanced training, matched-width relational features, independent validation and move ranking. Compare a small number of serious candidates with equal data exposure and reproducible seeds. Do not reject feature capacity solely because one undertrained or clock-confounded candidate loses.

**Search track:** trace missed alternatives, then test coherent pruning/reduction families; implement staged move selection and remove redundant hot work. Keep weights fixed so the source of improvements is identifiable.

**Gate:** candidate repairs a meaningful family of failures and earns a clear match advantage against the baseline across diverse starts. Any node-efficiency or static-loss claim must survive the wall-time tradeoff. Reject a candidate that wins only the fixtures it was designed around.

### Phase C — optimize the coupled system

Retrain on the stronger search’s leaf distribution; recalibrate pruning margins against its new evaluator. Verify that search and evaluation gains combine. Add selective extensions and endgame support where residual failures justify them. Increase model capacity or labelling effort only after a scaling experiment shows benefit.

Use real 120+0.5 validation, target initialization/RSS measurements, and platform validation before submitting a build. The full competition sample may be expensive; a fast match is a screening stage, not a replacement for that final check.

**The decisive question at each stage is whether the engine stops losing contested positions it currently misplays.** The evidence supports a larger evaluator/search project, not a promise that a hidden one-line fix recovers the entire gap.

## Reproduction and retained artifacts

- [Full committed source archive](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source-7f20dd1.tar.gz) and [hash manifest](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/source-hashes.json).
- [Fresh regression-test output](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/fresh-tests.txt).
- [Tournament screening script](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/rated_probe.py.txt), [reference results](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/rated-reference.json), and [copied PGNs](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/rated-pgn.tar.gz).
- [Completed-score rejudging script](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/exact_rejudge.py.txt), [exact reference scores](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/exact-reference.json), and [stronger rechecks](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/exact-reference-4m.json).
- [Targeted selectivity probe](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/selective_probe.py.txt) and [corrected ablation results](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/selective-ablation.json).
- [Current-build scaling script](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/failure_probe.py.txt) and [detailed results](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/current-failures.json).
- [Evaluation / clock / conversion probe](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/engine_probe.py.txt) and [results](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/engine-probe.json).
- [Checkpoint matching](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/checkpoint-match.json), [float/quantized comparison](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/quantization.json), and [comparison script](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/quantization_probe.py.txt).
- [Training sample indices](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/training-sample-indices.npy) and [sample predictions/labels](/home/edwardb/Documents/chessathon/audit/elo-gap-7f20dd1/evidence/sample-evaluations.npz).

For reproduction, extract the source archive into a separate directory, restore the probe scripts under an `elo-evidence` directory there, and adjust the local Python, Stockfish, checkpoint, and training-data paths. Execute the probes with the repository environment. They are offline diagnostics, not submission modules. `failure_probe.py` consumes `rated-reference.json`. Its original `judged` fields use the preliminary client; run `selective_probe.py` and then `exact_rejudge.py` with `uci_exact.py` to obtain the completed-score reference results. The delivered `current-failures.json` and `selective-ablation.json` have those corrected judgements merged in; their `*-preliminary.json` counterparts are superseded. `quantization_probe.py` consumes the sample indices created by `engine_probe.py`. The external dataset/checkpoints are not part of the committed archive.

The report’s performance observations are not isolated A/B measurements. Its independent-reference scores are finite-search estimates, and its difficult-position sample is deliberately selected. No Elo improvement was implemented or measured in this audit.
