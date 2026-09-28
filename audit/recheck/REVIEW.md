# Current-state reassessment

The board representation does not need replacing. The previous defect backlog has substantially shrunk: all **37 current audit tests pass**, including the differential state/NNUE checks, perft, current SEE oracle/examples, and TT move roundtrips. Production files were not edited in this reassessment. Source and weight fingerprints are in `source-hashes.json`; the weights match the previous representation audit.

## What is now addressed

Current source and passing tests cover game-tracking EP normalization, EP hash legality, qsearch repetition/material/fifty-move checks, mate precedence, initial promotion SEE gain, the searched-quiet malus list, root moved-piece context, node-count handover, and the two later SEE defects. The SEE implementation also checks king recaptures against attacks on the destination. Recent source changes additionally address TT move packing, root static evaluation context, history parity, and resetting the halfmove clock after a synthetic null move.

The audit findings were not all binary defects: root aspiration fail-high stopping was an experiment, and its current disabled setting should not be labelled an unfixed bug. Timing policy has changed, but this review did not measure its playing strength. Ordinary negative history updates remain unbounded; the qsearch stalemate fix remains partial; queen-only qsearch promotions remain.

## Highest-confidence remaining fixes

### 1. Terminal legality is still bypassed

`search.py:515` limits qsearch stalemate checks to at most three pieces on the side to move. The valid four-piece stalemate `7k/7p/7p/7p/7P/8/8/K5R1 b - - 0 1` returns **−1777** both in full-window qsearch and depth-one non-PV search with window [−777,−776]. An unpruned main search returns zero, which would fail high in that window rather than fail low.

Main-search early returns also precede legal-move detection; the rule-clock example below confirms a wrong cutoff through RFP. The probe includes a bare-king stalemate with a nonzero RFP return, but its negative search window makes that particular result a valid lower bound. It is not evidence of a wrong cutoff and should not be counted as one.

Fix legal-move existence before early score returns, then optimize the existence test. A piece-count heuristic does not establish that stalemate is impossible. Include child-transition tests and multiple windows; the original full-window terminal tests did not cover this.

### 2. Rule-clock context is absent from TT scores and speculative pruning

For `7k/8/8/8/8/8/8/KR6 w - - 99 1`, every legal move is reversible and none mates. Depth-one search should therefore score a draw under the engine's fifty-move convention.

Measured results:

| Search | Score |
|---|---:|
| Same board at halfmove zero, fresh full-window search | +2106 |
| Halfmove 99, reusing that exact TT entry, full window | +2106 |
| Halfmove 99, empty TT, full window | 0 |
| Halfmove 99, empty TT, window [−100,100] | +2200 |
| Halfmove 99, unpruned full window | 0 |

The first failure isolates clock-insensitive TT reuse. The second is RFP returning static evaluation before searching the imminent draw. Both need attention. Preserve repetition keys; a clock-aware TT identity or entry context is a separate design decision. Clock-aware TT handling alone does not fix pruning or repetition-path dependencies.

These two categories are confirmed scoring errors, but the synthetic cases do not quantify their frequency or Elo cost.

## Highest-potential investigation for larger gains

**Search selectivity versus evaluation remains the central unresolved question.** There is no evidence here that another collection of rare edge-case fixes adds hundreds of Elo.

The current search can prune quiet checking moves before making them and discovering check; it can reduce quiet check evasions; several static pruning mechanisms extend to substantial depths. These policies may work on average, so changing all of them at once would obscure the result. Add independent mechanism switches, a usable PV/decision trace, and counters indicating which mechanism skipped a promising line. The existing `set_pruning(False)` also changes check extensions and TT cutoffs and therefore does not isolate pruning effects at equal nominal depth.

A concrete synthetic example is `rrrrr2k/8/5KQ1/8/8/8/8/8 w - - 0 1`. Qg7 mates, but depth-one non-PV search in [1210,1211] returns **210** via razoring; unpruned search returns **29999**. This is a wrong fail-low, unlike returning 210 in [0,1], which is a valid fail-high. The artificial promoted-rook material makes this a diagnostic fixture, not evidence that this exact failure is common. Confirm relevance on newly collected game failures before choosing a broad pruning change.

Use individual ablations to identify whether the better move was excluded, reduced below its threat horizon, or searched and misvalued. If it is searched and still misvalued, inspect fresh quiet leaves with a stronger offline reference and build a phase-stratified holdout. The unchanged 768-feature network is a plausible source of further gains, but existing arithmetic/feature tests do not establish a network implementation bug. Diagnose before purchasing compute or replacing its architecture.

## Secondary candidates

- All promotion choices in qsearch, especially rook underpromotions that avoid stalemate and knight promotions with unique checks. Full move generation already supports them.
- Unified castling classification: `is_quiet` calls it quiet while the main loop's nonzero-flag test calls it a capture. This changes reduction/pruning eligibility and quiet counts. It may act as a useful exemption, but should be explicit rather than an inconsistent definition.
- Bounded ordinary-history updates, evaluated as a tuning change.
- Staged move selection/SEE reuse/lazy accumulator updates, one at a time after profiling actual search. The earlier bit-scan microbenchmark did not establish a whole-search speed improvement; do not use it to justify a representation rewrite.

## Next action

Implement the two confirmed draw-handling fixes with transition/window/context regressions. In parallel as a conceptual workstream, build selective-search diagnostics against frozen weights; do not conflate this with simultaneously changing the network. Then compare targeted pruning protections on fresh failure positions and accept candidates with paired equal-time games on independent held-out starts. Treat greater nominal depth, fewer nodes, and small reference-score differences as diagnostics rather than Elo.

The ready-to-send implementation brief is `CLAUDE_HANDOFF.md`. `probe.py`, `results.json`, and `tests.txt` preserve the new evidence. No production changes or strength matches were performed in this review, and no amount of the leaderboard gap has been assigned to a particular fix.
