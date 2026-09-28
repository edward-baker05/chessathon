Review the current checkout and implement the remaining fixes below in separate, reviewable changes. Do not read git history or old experiment logs. Do not edit harness/. Preserve current source/weight fingerprints and leave evaluation weights unchanged during search tests. Read AGENTS.md and fetch the canonical rules if an implementation decision depends on competition constraints.

The current audit suite passes all 37 tests. Do not undo the fixes already present: game tracking/EP normalization, recursive qsearch draw checks and mate precedence, promotion gains, searched-quiet history lists, root moved-piece/static context, repetition parity, null halfmove reset, TT move packing, and dynamic SEE pin/king/promotion handling. The root fail-high break is deliberately disabled; do not enable it merely because an old audit suggested testing it.

Fresh reproductions are in audit/recheck/probe.py and results.json. Read those first. They are score/semantic tests, not an Elo estimate. Add assertions to the real regression suite, including tests that reach problematic nodes through legal moves and test multiple alpha/beta windows.

1. Fix terminal draw handling across main search, quiescence, and early returns.

The STALEMATE_PIECES=3 gate is incomplete. FEN `7k/7p/7p/7p/7P/8/8/K5R1 b - - 0 1` is a valid stalemate with four black pieces. Current qsearch returns -1777, and depth-one non-PV negamax with window [-777,-776] returns -1777. Both must recognize a draw.

The main search can also return before reaching the qsearch terminal guard. The near-fifty-move RFP example below demonstrates an actually incorrect cutoff through this path. Do not infer an incorrect alpha-beta bound merely because a narrow-window search returns a nonzero score on a draw: some such bounds are valid.

Establish terminal legality before score-returning pruning can misclassify a terminal position. Start with a correct baseline, then optimize legal-move existence using sound fast paths and early exit, rather than a piece-count assumption. Cover RFP, razor, null, TT interactions, and qsearch stand-pat; preserve mate precedence. Benchmark the whole search to quantify the cost. Do not trade known terminal correctness away based only on a small NPS comparison.

2. Make near-fifty-move search and TT reuse context-safe.

FEN `7k/8/8/8/8/8/8/KR6 w - - 99 1` has only non-mating reversible legal moves, so the depth-one result is a draw. Current non-PV narrow-window search can return +2200 via RFP with an empty TT. If the same board at halfmove zero was searched first, an exact TT score of +2106 can be reused at halfmove 99 instead. The probe also compares wide-window warm/cold searches to isolate TT reuse from RFP.

The position key excludes the halfmove clock. Keep repetition identity separate from any rule-clock-aware TT identity. Options include storing/checking rule-clock context alongside TT scores or using a separate context-aware TT key; allow reuse of safe move ordering/static information when rejecting a cached search score. Consider scores influenced by repetition separately: adding the halfmove clock alone does not solve path-dependent draws. Add warm-vs-cold tests in both clock directions, and real-history repetition tests.

Guard or adjust speculative pruning near the rule threshold; fixing TT reuse alone does not fix the empty-TT RFP example. Avoid an unexplained magic clock cutoff: document the correctness scope of the chosen approach and test just below/at the boundary, including pawn moves, captures, and mating moves.

3. Build a targeted selectivity diagnosis before adding more heuristics.

Current quiet futility/LMP/SEE pruning happens before child check status is known, so giving-check quiets have no protection. LMR can reduce quiet check evasions too. RFP runs through depth 8, futility through depth 6, LMP through depth 8, and shallow null fail-highs are unverified. These are hypotheses about strength, not independently proved bugs.

One synthetic diagnostic: `rrrrr2k/8/5KQ1/8/8/8/8/8 w - - 0 1` has Qg7 mate. At depth one, non-PV window [1210,1211], current search returns 210 via razoring while unpruned search returns 29999. The position is artificial; it demonstrates that a forcing line can be pruned, not its frequency in games. A fail-high of 210 in window [0,1] would be a valid bound and must not be labelled a failure.

Add independent development switches and counters for RFP, razor, null, futility, LMP, SEE pruning, LMR, and internal iterative reduction. Preserve other mechanisms while testing one switch. The existing set_pruning(False) also changes check extensions and TT cutoffs, so it is not a clean single-mechanism experiment. Record which mechanism removes a reference move, actual depth/window, check status, and principal variation. First test targeted protections for giving checks/evasions and low-material or near-draw nodes, then calibrate margins with fresh failures. Do not disable everything and call fewer errors at equal nominal depth a strength gain.

4. Address remaining smaller issues separately.

The capture generator still emits queen promotions only. Add all four promotion choices or an explicitly verified selective policy, including underpromotions that avoid stalemate and knight promotions with unique checks. Full move generation already supports these; ensure qsearch ordering and pruning retain them.

Ordinary history still has unbounded negative additions and asymmetric whole-side halving, although continuation history is clipped. Bounded/gravity updates are a tuning candidate; do not present a preferred formula as a proven bug fix. The main search also classifies castling as a capture because its flag is nonzero, while is_quiet correctly classifies it as quiet. Unify semantic classification, with any intentional reduction/pruning exemptions made explicit and tested.

5. Pursue evaluation capacity only with a diagnostic holdout.

The shipped weights are unchanged from the representation audit. No new evidence establishes a feature-transport or integer-arithmetic defect. If bad choices survive the selectivity experiments, collect fresh quiet leaves from those lines and compare them with stronger offline reference analysis. Stratify by material, passed pawns, king activity/safety, and rook endings. Separate tactical contamination and score-scale differences from evaluation errors. Compare float vs quantized outputs and move ranking before retraining. King-conditioned features/data reweighting are candidates, not a reason to replace the board representation or buy cloud compute yet.

Validation and reporting:

- Run existing tests plus the new transition/window/TT-context regressions, then ruff/mypy and required smoke checks. Do not change harness/.
- For representation/performance-only changes, require identical search decisions and node counts on fixed-node tests, then measure wall time. For pruning changes, fewer nodes or greater reported depth are not proof of strength.
- Use paired, independent opening starts and equal-time games for acceptance, with frozen weights and source fingerprints. Keep development failures separate from final acceptance openings. Retain cluster-aware uncertainty and the actual stopping rule. Do not compare old/new node-count definitions as equal work.
- Keep correctness fixes, search tuning, and network changes separate; test any final combination. Report what is confirmed, what is exploratory, and what remains untested. Do not claim that these fixes recover a particular amount of the leaderboard gap.
