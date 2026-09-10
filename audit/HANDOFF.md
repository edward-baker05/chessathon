# State of the draw-handling work, and where to go next

Written at commit `be21ddb`, on branch `sweep`. Weights are unchanged from the
representation audit; `audit/recheck/source-hashes.json` still matches
`weights/net.npz`. Only `search.py`, `movegen.py` and `tt.py` differ from the audited
tree, plus tests and two new tools.

## Commits, oldest first

| commit | what |
|---|---|
| `8a192de` | the audited working tree, committed unchanged so the rest is reviewable against it |
| `2640a08` | terminal legality before any score is returned |
| `225b708` | rule-clock context on cached scores |
| `e7d3b37` | no speculative pruning inside the fifty-move band |
| `e14a714` | one switch and one counter per selective mechanism |
| `b186341` | restore a note lost while splitting the two commits above |
| `d92020f` | keep the post-fix probe run beside the diagnostics |
| `3ac83e9` | close the reduction and depth-field holes in the band |
| `e673e7f` | ablate pruning in both directions, not just subtractively |
| `4031195` | one uncontaminated equal-time match |
| `be21ddb` | measure what selectivity costs on positions from real games |

77 tests pass. ruff and mypy are clean. `make gate` passes. `make zip` builds at
677 KB unzipped.

## Confirmed defects, fixed

1. **Terminal legality.** Quiescence looked for stalemate only where the side to move
   had three men or fewer. `7k/7p/7p/7p/7P/8/8/K5R1 b - - 0 1` is a valid stalemate
   with four and scored -1777 in every window tried. `movegen.has_free_move` is a sound
   one-sided test: with the side to move not in check, a piece that is neither the king
   nor standing on a rank, file or diagonal through its own king is unpinned, so one
   such piece with anywhere to go proves a legal move exists. En passant is excluded
   deliberately, being the one capture that can expose a king from off the pin lines.
   The exact generator decides the rest.
2. **Rule-clock context on cached scores.** The position key does not carry the halfmove
   clock, so a score searched at clock 0 came back at clock 99 where the same board is a
   draw. Entries now carry one bit of context and a score is reused only when the entry
   and the node reading it both sit outside a sixteen-ply band below the threshold. The
   move and static evaluation are properties of the position and keep being used.
3. **Speculative pruning inside the band.** With an empty table the same position still
   returned +2200 through reverse futility. Reverse futility, razoring, the null move,
   and the futility, late-move and static-exchange skips inside the move loop are all off
   inside the band.
4. **Internal iterative reduction inside the band.** Found by testing the boundary rather
   than the supplied fixtures. It is the only reduction with no re-search behind it. On
   `7k/8/8/8/8/8/8/KR6 w`, where the draw sits exactly `100 - halfmove` plies away, it
   returned +2368 at clock 94 depth 6, +2313 at 95 depth 5, +2254 at 96 depth 4, and was
   correct from 97 down, which is exactly where the reduction stops firing.
5. **The seven-bit depth field truncated.** A direct search at MAX_DEPTH with a checked
   root extends to 128 and came back as depth 0. It now saturates.

## What the band does not cover

The band is a mitigation with a stated scope, not general correctness.

- A node more than sixteen plies from the threshold whose own search shuffles all the way
  to it is still scored without the draw in view, and an entry made from it is marked
  context 0 and can be reused at any other clock outside the band. The pruning predicate
  `near_fifty_move` widens to the remaining depth and covers deep searches; the table
  predicate `rule50_context` is clock-only and cannot, because store and probe must
  compare the same function.
- **Repetition is not covered at all.** A node scored zero because its line repeated is
  stored as an ordinary exact score and can be handed to a line that does not repeat.
  Closing it needs a second bit and a way to carry it up from the node that saw the
  repetition. The gate in `search.py` says so beside itself, and
  `test_repetition_from_the_real_game_history_is_a_draw` pins current behaviour rather
  than fixing it.

## Measured strength

`audit/ab-drawfixes/`. Both engines frozen in `snapshots/draw-fixes` and
`snapshots/pre-draw-fixes`, untouched for the duration. Run length fixed before starting:
two chunks of twenty pairs on halves of the opening set sharing no cluster.

    +10 =59 -11 over 80 games, 49.375% on 40 pairs in 32 clusters
    Elo -4, 95% CI [-44, +35], zero agent failures
    throughput cost of the fixes: about 4%, paired and interleaved

This detects a large regression and finds none. It cannot resolve the 3 to 5 Elo a 4%
throughput cost is worth. **The build is a candidate, not a demonstrated improvement.**

At fixed depth 11 over the eight bench positions the search is bit-identical to the
audited tree: 925,557 nodes, same moves, same scores. The fixes change nothing outside
terminal and near-fifty-move positions.

## Selectivity: where the real lead is

`tools/selectivity.py` ablates in both directions. Subtractive alone cannot clear
pruning: if two mechanisms would each remove the same line, turning either off changes
nothing. Additive starts from all eight off, which still keeps the check extension and
the transposition policy that `set_pruning(False)` throws away.

`tools/failures.py` measures the frequency on played games. Over 300 positions from the
80 match games, at depth six:

| | | |
|---|---|---|
| floor beats shipped by over 50cp | 15 of 300 | 5.0% |
| of those, a quiet move giving check | 2 of 300 | 0.67% |
| gaps surviving the floor's own node budget | 4 of 300 | 1.33% |
| quiet moves giving check among the survivors | 0 | |

Equal nominal depth is not equal work: the floor spends 25x the nodes for the same depth.
Given that budget the shipped search closes 11 of the 15 gaps.

**Do not build a give-check exemption on the composed fixtures.** It targets 0.67% of
positions at equal depth and none of the gaps that survive equal work.

The lead worth taking is the pair interaction. On `8/6pk/4Kp2/7p/5P1P/3q4/3N4/2r5 b`, a
real game position where a forced mate is missed, no single mechanism causes the failure
and no single mechanism prevents it, but reverse futility with futility, futility with
late move pruning, and late move pruning with late move reductions each reproduce it
exactly. Margin tuning one at a time cannot find that.

Two of the four surviving gaps are missed forced mates in simple endgames, which may be a
separate weakness worth its own look.

## Untouched

Handoff items 4 and 5: all four promotion choices in the capture generator, bounded
ordinary-history updates, unified castling classification, and the evaluation holdout.
The root fail-high break remains deliberately disabled.

## Practical notes for whoever runs the next match

- Each agent process peaks at **1.03 GB resident**. Eight of them at four workers
  exhausted memory and pushed agents past the ninety second init budget, which is what
  produced fourteen unexplained startup losses in an earlier run. Two workers is safe on
  this machine.
- Imports across separate cores barely contend: 55.5 s alone, 62.7 s with six at once.
  Two agents on one core take 116 s, but the referee starts them sequentially.
- **A single agent import takes 56 s of the platform's 90 s budget** on this machine,
  alone on one core. That margin is thin on hardware we cannot test.
- The harness runs an agent from its directory in place with no copy. Never edit the
  working tree while a match is drawing players from it. Use frozen snapshot directories.
- `audit/recheck/probe.py` writes its results back over `results.json`. Copy it first.
- In a narrow window the unpruned oracle returns a bound, not a score. Compare bounds as
  bounds; a return of 210 is a correct fail-high in [0,1] and a wrong fail-low in
  [1210,1211].
