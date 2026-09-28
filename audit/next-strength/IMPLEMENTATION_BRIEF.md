# Next strength campaign: independent diagnosis, then two substantial candidates

## Assignment

Take the current engine beyond the correctness sweep. Implement and evaluate a small number of changes aimed at a substantial strength improvement. Do not spend this task finishing a miscellaneous list of possible small optimizations, and do not finish with another diagnosis-only handoff if a candidate can be built and tested locally.

The strongest next bets are (A) coherent search selectivity, judged by independently assessed move quality, and (B) evaluation capacity/data fit, tested with a controlled king-conditioned network pilot. Neither is established as the source of hundreds of Elo. The leaderboard difference motivates the work; it does not prove an attainable gain from any particular change.

Read AGENTS.md and audit/HANDOFF.md. The user supplied and authorized that handoff and the selectivity results it describes. Do not read git history or unrelated old experiment logs. Use current source and these explicitly supplied findings. Preserve all existing correctness fixes. Do not edit harness/. Do not rewrite the board representation, import another engine's implementation, or use someone else's network. External Stockfish is an offline measurement/labeling tool only.

This brief was written after reading current tools/failures.py, tools/selectivity.py, training code and the supplied handoff. It does not independently certify the handoff's reported matches or tests. Run the current suite to establish your own starting point.

## 1. Correct the measurement before optimizing against it

This is a bounded tooling task, not a new general audit. tools/failures.py currently measures differences between two searches' own root scores. In particular, `gap = floor_score - shipped_score` and `survives = floor_score - equal_nodes_score > margin` do NOT establish move-quality losses. The searches can pick the same move and assign it different scores; the more optimistic search can be less accurate. Looking only at positive gaps also misses damaging overoptimism by the shipped search.

Its position sampler takes a prefix of games up to a position limit, discards checked positions, and stores only FENs. `fresh()` clears real game history. Those choices limit what the reported 15/300 and 4/300 rates mean. The equal-node re-search does not even retain its selected move in the result. Do not repeat those rates as independent error frequencies or use them to rule out checking-move issues globally. Equal-node work is also not equal wall time across different pruning policies.

Extend the existing tooling rather than creating several overlapping frameworks:

- Record the actual selected move, completed depth, nodes, elapsed time, score/bound, mechanism mask, source/net identity and whether the search aborted. Save every equal-budget candidate move.
- Sample by game/opening cluster, spread across games and phases instead of taking a file prefix. Retain checked positions as a separate stratum. Deduplicate positions, cap samples per game, and carry actual pre-root history into both searches. Separate cold-cache diagnostics from live-game replay.
- Start with roughly 200–300 development positions, including the existing four survivors, both signs of large score disagreements, positions with different moves, and an unbiased sample where the searches agree. The curated failures and random sample must remain separately labelled. Do not train or tune against the acceptance set.
- For each position, independently assess the union of candidate moves and a stronger reference move using the local Stockfish. Clear the reference hash per restricted-move analysis, use equal reference budgets and consistent side-to-move perspective, and recheck important disagreements at a larger budget. Single-PV scores from separate engine searches are noisy; disagreements that do not stabilize remain unresolved.
- A practical starting reference budget is 250k nodes per distinct candidate, escalating serious disagreements to 1m and then 4m as needed. Benchmark throughput first and reduce sample breadth before accepting unreliable labels. These are starting budgets, not claims of ground truth.
- Keep mates and non-mate scores separate. Verify claimed forced mates by an independent search where possible; a legal PV alone proves only one cooperative line. Do not average mate sentinels into centipawn loss.
- Store game history for the reference too where it affects adjudication. audit/uci.py currently sends `position fen`; passing a python-chess Board with a stack does not make that client send its history. Extend its protocol handling or use a client that does.

Report independently assessed move regret, severe-error counts, missed mates, and class breakdowns. Use game/cluster resampling for uncertainty. Static score differences between implementations are diagnostics, not regret. Save results to a new run directory; never overwrite audit/recheck/results.json or earlier evidence.

Deliver this first calibration before starting a long training or match run. It should identify which existing survivors are genuinely inferior decisions, not merely lower internal scores.

## 2. Search candidate: fix a general interaction, not four FENs

Use tools/selectivity.py's additive floor and pair masks. Preserve check extensions and the chosen TT policy when changing mechanism masks; set_pruning(False) is a different oracle configuration. Check which qsearch heuristics remain active under the eight-mechanism floor—do not call it fully unpruned unless that is true.

The supplied real-game lead is `8/6pk/4Kp2/7p/5P1P/3q4/3N4/2r5 b - - 0 1`. The handoff reports independent failures from RFP+futility, futility+LMP, and LMP+LMR. Confirm its complete FEN/history from the supplied results before recreating a live-game test; the FEN here specifies an empty clock/history for an isolated fixture.

Trace the earliest decision at which a independently confirmed good line becomes unavailable or too shallow. Log the position, move, actual remaining depth, window, static evaluation, checking/evasion status, and preceding reductions. Mechanism counts alone do not explain a missed line. A TT walk is not an authoritative completed principal variation: root search may not store a root entry, and descendant TT entries can be bounds or from different iterations. Add an explicit development PV/root-move trace where needed.

The candidate should address a recurring pattern across independently confirmed failures. Leading hypotheses:

- IIR and LMR compound, leaving important later moves below their threat horizon. Test a shared effective-depth policy or a targeted limit on compounded reductions.
- Static futility/RFP and move-count pruning independently erase every tactical resource. Test a coherent less-selective policy for demonstrably unreliable nodes, using information available at runtime rather than fixture-specific FENs or arbitrary material thresholds.
- Mate-threatening lines or check evasions are excessively reduced. Test a narrowly motivated protection if the independent data supports it. Do not blanket-exempt checks because a composed fixture improves, and do not rule them out using the old biased sampler.

Select one coherent candidate, not a grid of dozens of margin values. Compare baseline and candidate at equal time as well as equal nodes. More nominal depth and reproducing the floor at 25x work are not the objective. If the modification only fixes the original fixtures, fails to generalize, or loses badly at equal time, stop that branch and allocate the remaining effort to evaluation.

The floor itself is not a candidate to ship simply because it finds more tactics at equal depth. Likewise, do not expand this task into proving general path-dependent TT correctness unless fresh, frequent failures implicate it. The existing draw-context mitigation has known limits; the claim that one additional propagated bit would completely solve every repetition-context issue also requires a precise bound policy and tests.

## 3. Evaluation candidate: run the controlled pilot now

This work has been repeatedly deferred. It is a plausible route to a larger gain and should now receive an actual local experiment rather than another recommendation to build a holdout someday.

The current input is 768 piece-square-colour features feeding shared perspective accumulators, with material-count output buckets. That architecture can represent interactions, but king-conditioned features could make important relations easier to learn. There is no proof that capacity is the bottleneck, so keep an architecture-matched control.

First establish a small fresh holdout from independently generated games and quiet leaves. Split by game/opening cluster before any training, deduplicate against the selected training subset, and distinguish terminal states from evaluable positions. Include middlegame king exposure, rook endings, pawn races, material imbalance and king activity. Filter or separately label tactically unstable positions: a deep teacher score is not automatically an appropriate static target. Audit teacher score convention and phase-dependent calibration; minimize neither raw CP error across incompatible scales nor saturated sigmoid loss alone. Include candidate-move/quiet-leaf ranking and float-versus-quantized agreement.

Build one modest prototype: four own-king location buckets per perspective, for example a fixed 2x2 partition of the perspective-oriented board, applied to the existing 768 features. Keep hidden width, activation, output buckets, loss and training schedule otherwise matched initially. This is an experiment in king conditioning, not a request to copy a published engine's feature scheme. Explicitly document the orientation and bucket mapping.

Engineering requirements:

- Derive both kings from the packed piece records; the input records already contain their piece codes and squares. Verify offline feature extraction against runtime encoding.
- A king crossing its perspective's bucket boundary requires refreshing that accumulator; ordinary moves within the bucket remain incremental. The other perspective still needs the correct piece-square update. Test captures, all promotions, castling and king bucket transitions.
- Benchmark inference, update/refresh frequency, cold import and peak memory using representative arrays before a full training run. A net that compiles too slowly or loses too much search throughput is not useful regardless of validation loss. If necessary reduce buckets or width, with a matched control for the capacity change.
- Train a fresh current-architecture control and the king-bucket candidate on the same subset, labels, split and comparable schedule/updates. Include the shipped net as a third reference. Do not credit architecture for changes caused by retraining, different data or longer optimization. A small pilot can reject a candidate but may not establish its converged potential.
- Use a reproducible, phase-stratified subset of existing data to start, mixed with a bounded amount of fresh labels only if the holdout diagnoses a distribution gap. Do not regenerate hundreds of millions of labels or retrain on all data before checking whether the pilot learns and improves ranking.
- Measure a short actual training run on the RTX 2060 and set the pilot size to fit the available time. Save intermediate checkpoints and learning curves. The earlier throughput extrapolation is not a guarantee for this architecture or I/O path.
- Quantize with explicit overflow checks and compare quantized against float inference before games. Preserve the production net until a candidate has passed acceptance.

If king conditioning shows no credible improvement after a reasonable learning curve and control, report that and test at most one evidence-led data intervention with the current architecture—for example underrepresented rook/pawn endings. Do not search many network shapes on the same holdout.

## 4. Acceptance and resource discipline

There are two candidate tracks, not permission for unbounded concurrent processes. First measure the actual memory ceiling imposed on this task (including cgroup/container limits), not merely host free RAM. The supplied handoff reports startup failures despite apparently free host memory. Use one match worker initially; increase only after stable resident/peak measurements support it. Avoid concurrent training, reference labeling and agent compilations that invalidate timing or exceed memory.

Use immutable agent directories for every match, including the baseline; do not swap production files beneath a running process. Record full source and net hashes and all runtime switches. Perform a short startup/legality test before a long run. Preserve all raw game outcomes, failures and interruptions, marking invalid strength runs as invalid rather than deleting the evidence.

Reserve an independent set of fresh opening/game clusters for acceptance. Use paired equal-time games with a declared sample size and uncertainty calculation, not repeated peeking until the result looks positive. A first screening match should target a material regression or promising gain, not resolving single-digit Elo. The reported 80-game interval [-44,+35] does not exclude a substantial regression; it merely failed to establish one. Do not convert a throughput percentage into a fixed Elo number without measurement.

If both search and net candidates pass screening, test the 2x2 combinations (baseline/candidate search crossed with baseline/candidate net) to expose interactions, then spend remaining match budget on the best supported combination against baseline. Include an independently structured opponent if feasible; self-relative gains need not transfer to the leaderboard. Never equate Stockfish's configured rating with this competition's Elo scale.

Local GPU and CPU use are authorized. Cloud spend needs a concrete justification, measured local runtime, expected value and a spending cap approved by the user before purchase. No automatic uploads or replacement of the best-known submission.

The rules and contract must be fetched before relying on current limits or deadlines. This brief's author attempted both canonical URLs on 2026-09-10, but the web tool could not open them; no current deadline is certified here. Obtain that information early and reserve final time for packaging/platform validation. Judge observed local import time against the verified limits without treating local hardware as the platform.

## Deliverables and stopping decisions

Put new work in a uniquely named directory beneath audit/next-strength/; do not overwrite this brief or the prior handoffs. Keep a concise ongoing manifest so interruption does not destroy progress.

Deliver:

1. An independently assessed development set, with histories, candidate moves, reference budgets and unresolved cases.
2. A trace-based account of the frequent failure classes and one tested search candidate, or concrete evidence for stopping that track.
3. The matched evaluation pilot: training configuration, control/candidate checkpoints, holdout/ranking/quantization results, and inference/import costs. If infeasible, record the measured blocker rather than indefinitely deferring it.
4. Valid paired match results for viable candidates, with limitations and failures retained.
5. A best-supported build recommendation and exact reproduction commands. State separately what is correct, what is faster, and what wins games.

Do not use most of the task on instrumentation. Start the evaluation pilot once a small independent holdout and feasibility benchmark exist; expand labels only when they resolve a decision. If time is short, prioritize the independently judged failure sample and the matched evaluation pilot over minor history, castling, bit-scan or underpromotion tuning. Those smaller issues can remain documented unless they directly block a candidate or correctness validation.
