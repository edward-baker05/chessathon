# Next strength campaign, run 1

Written against `audit/next-strength/IMPLEMENTATION_BRIEF.md`. Every number below was
measured in this run; nothing is carried over from the previous handoff, and where this run
contradicts it, the contradiction is stated rather than smoothed over.

`MANIFEST.md` beside this file is the timestamped log of what happened when. This is the
argument.

## The short version

**The search is not where the strength is going.** Judged by Stockfish 19 on 320 positions
from 232 real game clusters at a million nodes a move, which is what a rated move on this
machine actually gets, the shipped search plays the reference's best move in **268 of 320**
positions and gives up **4.9 centipawns** of mean regret, 95% over game clusters [3.0, 7.0].
Regret is measured against the best of the moves actually put to the reference, so it rises
as more alternatives are scored: the same build on the same positions reads 239 and 6.9cp
once a second engine's moves join the set. Every comparison below is within one merged
pass, which is the only way two builds are comparable.

A coherent search candidate was built from an independently confirmed failure set, fixed 8
of 19 of those failures at fixed depth, and then made move quality **worse** on the whole
sample. It is reverted.

**Two of the four failures the previous run reported are not failures.** In both, the
shipped search and the unpruned floor play the same move, and it is the reference's best
move. The "gap" was two implementations disagreeing about what one move is worth.

**King conditioning is the one open lead, and it did not clear the bar.** A matched pilot pair
at 8 epochs on 100M positions put four own-king buckets 3.6% ahead on the trainer's own
holdout, 0.4% ahead on a fresh holdout, with no ranking gain and exactly 50.0% over 120
games. The same comparison at 12 epochs over all 269M positions puts it **6.6% ahead of its
own matched control and 7.1% ahead of the shipped network** on the fresh holdout, and 1.9
points better at picking the reference's top move. The pilot would have rejected it outright.
But it also costs **4.6%** of search throughput, is slightly worse on independently judged
move quality inside a search, and drew a 240-game acceptance match: 50.6%, Elo +4, 95% CI
[-27, +32]. **Nothing from this run ships.** The current build stands.

**The measurement built to answer these questions is the durable part of this run.**

## 1. What the rules actually say

Fetched from the canonical pages on 2026-09-10, at the start of this run, because the brief
could not open them and the numbers move:

| | |
|---|---|
| registration closes | **Sep 11 11:00**, the final Swiss that afternoon, live final Sep 12 |
| time control | 120 s + 0.5 s per move per side, wall time |
| init budget | 90 s, before the clock starts |
| hardware | one core of an AMD EPYC 9V74 at 2.60 GHz, 2 GB, no network, no GPU |
| size | <= 50 MB unzipped, 10 uploads per team per day |
| log | 8 KB kept, as the first 4 KB and the last 4 KB |
| networks | must be self-trained; labelling with an existing engine is allowed; starting from a published network is not |

That left **26.5 hours** from the start of this run, and it is the reason the evaluation
pilot is sized the way it is rather than the size the brief's most ambitious reading would
imply.

## 2. Correcting the measurement

`tools/failures.py` measured the difference between two searches' own root scores and
reported it as a loss rate. It is not one, for three independent reasons, and this run
replaced the tool rather than patching the report.

**A score difference is not a move-quality loss.** Two searches can select the same move and
disagree about what it is worth, and the more optimistic one can be the less accurate. Of
the four "surviving gaps" the previous run reported, two have the shipped search and the
floor playing the *same move*.

**Only one direction was looked at.** `gap = floor - shipped` counted the floor scoring
higher and never the reverse, so the case that actually costs games, the shipped search
being wrong in its own favour, could not appear. Measured properly on 320 positions, the
disagreement is symmetric: 28 above +50cp and 28 below -50cp.

**The sample was a file prefix with the histories thrown away.** Positions in check were
discarded, `fresh()` cleared the real game history, only FENs were stored, and the
equal-node re-search did not keep the move it chose. Both draw rules read history, so a
probe without it is not the position the game was in.

What replaced it:

- `tools/failures.py` samples by game and by phase across every available PGN, caps four
  positions per game, deduplicates on the position proper, keeps checked positions as their
  own stratum, and carries the real move history into every search. It records the selected
  move, completed depth, seldepth, nodes, elapsed seconds, score, whether the search aborted
  and which mechanisms fired, for the shipped search, the additive floor, an equal-node
  re-search and a fixed-node arm. Cold and warm cache regimes are separate and labelled.
- `tools/judge.py` asks Stockfish 19. Every move under test, including the reference's own
  choice, gets its own search **restricted to that move**, on a **cleared hash**, with the
  **real history** behind it, at an **equal node budget**. Scores read off one unrestricted
  search are not comparable, because the move the search believed in got most of the budget.
  Disagreements escalate 250k -> 1M -> 4M nodes and anything that does not stabilise is
  reported as unresolved, not as a finding.
- `audit/uci.py` gained the two things that made this possible: it now sends the game's
  moves rather than only the current FEN, and it ignores `lowerbound`/`upperbound` info
  lines, which are an unfinished iteration reporting which side of its window it fell out
  of and not a score. A node limit that lands mid-iteration otherwise returns a number off
  by the aspiration window.
- Mates are never averaged into a centipawn mean. `rank()` gives a total order over scores
  with no centipawn conversion anywhere: delivering mate beats every centipawn score and
  sooner beats later, being mated loses to every centipawn score and later beats sooner.
  Missed mates, walks into mate and slower mates are counted as their own classes.
- Uncertainty resamples game clusters, not positions. Four positions from one game are one
  game seen four times.

### The development set, judged

320 positions, 232 clusters, 290 quiet and 30 in check, at depth 6, cold cache.

| verdict on the shipped move | count | share |
|---|---|---|
| ok | 286 | 89.4% |
| material (50-100cp) | 7 | 2.2% |
| severe (>=100cp) | 13 | 4.1% |
| slower mate | 6 | 1.9% |
| missed mate | 2 | 0.6% |
| walked into mate | 6 | 1.9% |

Mean regret over the centipawn-comparable cases, 17.6cp, 95% over clusters [12.0, 24.3].
Nine positions never settled at 4M reference nodes and are recorded as unresolved.

Endgames carry 16 of the 27 real errors while having the *lower* mean centipawn regret
(8.3cp against the middlegame's 23.7cp), because the endgame errors are mate-class rather
than centipawn-class.

Those are cold-cache numbers, and a live game is not cold. Re-run over 80 positions with the
same side's two previous moves re-searched first and the table kept, which is the state a
game actually arrives in, 88.8% of shipped moves are judged fine against 89.4%, and mean
regret is 11.4cp, 95% [5.7, 18.2], against 17.6cp. It is a smaller and different sample, so
the two are not a paired comparison, but the direction is the expected one and the cold
figures quoted throughout are the pessimistic ones.

### The four previous survivors

Recovered from `audit/selectivity/failures-d6.json` and put to the reference. They carry no
history, because the previous tool stored none; that limits what they can show about either
draw rule and is why they are labelled `fixture` and kept apart from the sampled set.

| position | previous claim | what the reference says |
|---|---|---|
| `3r2k1/5ppp/Ppp5/...` | shipped 727cp, floor 1308cp | **same move**, `b3g3`, and it is the best move |
| `8/8/8/8/8/2q3k1/8/3K4 b` | shipped 4563cp, floor 29995cp | **same move**, `c3b2`, and it is the best move (mate in 3) |
| `8/6pk/4Kp2/7p/...` | shipped `d3d2`, floor `c1c7` | real, and small: mate in 4 instead of mate in 2. Still winning |
| `8/6pk/2N2p2/7p/...` | shipped `f3e3`, floor `c6a5` | the engine is *being* mated either way; it picks mate in 10 against over mate in 11 against |

**None of the four is a lost win or a lost draw.** The previous headline rested on score
differences that a move-quality measurement does not support.

## 3. The search candidate, built and rejected

### Attribution across confirmed failures, not fixtures

`tools/attribute.py` runs both ablation directions over every position the reference had
already confirmed the shipped move wrong in, and scores every move any variant produced
under the same restricted-move conditions, so "this mechanism broke it" means the reference
says the move got worse and not that the move changed.

Nineteen of the 27 are positions the additive floor itself gets right, so only those can
attribute anything.

| added alone to the floor | breaks | removed alone from the full search | fixes |
|---|---|---|---|
| reverse futility | **6** of 19 | reverse futility | **10** |
| razoring | 3 | late move reductions | 8 |
| late move pruning | 3 | late move pruning | 7 |
| null move | 2 | static exchange | 6 |
| futility, late move reductions | 1 each | null move | 4 |
| static exchange, internal iterative reduction | 0 | razoring, futility | 2 each |

Ten of nineteen break from a single mechanism and six only from a pair, so this is not
purely the pair-interaction story the previous handoff proposed. Reverse futility is the
single largest contributor, and **every one of its six is an ending**.

Measured on those six: static evaluations from +581 to +6334 against a margin of 75 a ply,
with reverse futility firing between 434 and 1923 times in one depth-6 search. The cutoff
returns the raw static evaluation, so the subtree collapses into "the static evaluation one
move down", which is no way to convert a won ending.

### The candidate

Two linked changes, both aimed at that, both using only information available at runtime:

1. withhold the cutoff below a piece count. Piece count is not an arbitrary threshold here:
   it is already the quantity the evaluation buckets its own output weights by, so a node
   the evaluation treats as a different regime is a node the margin was not calibrated on.
2. return `beta + (static - beta) / 3` rather than the raw static evaluation. The cutoff
   proves "at least beta" and nothing more; returning the static score returns an unproven
   number that is then stored as a lower bound at that depth and read back by later probes.

Both were implemented so that the defaults reproduce the old behaviour exactly, and the
bench stayed at **925,557 nodes** with identical moves and scores throughout.

### Why it is reverted

On the 19 confirmed failures, at fixed depth 6, the two together put **8 of 19** right and
the piece-count guard alone 6. That is a real effect on the thing it was built from.

At a 300k-node budget on the same 19, the baseline already gets **14 of 19** right and every
candidate is one position worse.

Widened to the whole 320-position sample, both builds probed at the same fixed node budget
and every move either produced scored **once against the same merged candidate set**, so
neither gets a lower bar than the other:

| build | best | <50cp | severe | mate class | mean cp | 95% over clusters |
|---|---|---|---|---|---|---|
| baseline | 250 | 58 | 6 | 6 | **5.7** | [3.7, 8.0] |
| candidate | 250 | 56 | 8 | 6 | **10.3** | [5.4, 16.3] |

and by phase the damage is exactly where the change was aimed:

| build | endgame | middlegame | opening |
|---|---|---|---|
| baseline | **5.7c** | 7.3c | 5.9c |
| candidate | **15.6c** | 5.8c | 4.3c |

The intervals overlap, so this is not a proven regression. What it is is a complete absence
of any gain, in the one phase the change was built to help, on the sample it was not
selected from. No match budget was spent on it. The reasoning and the numbers are left in a
comment beside the cutoff in `search.py` so the idea is not rebuilt from scratch.

### The depth trace, which says the same thing from the other side

`tools/depthsweep.py` runs both builds at every depth from 1 up on each confirmed failure
and asks the reference about every move either picks, so the answer is the shallowest depth
at which each first plays a move the reference is happy with. The difference between those
two depths is what selectivity actually costs on that position, in plies.

The shipped search gets there in **18 of 19** within depth 12, at a **median depth of 3**.
Where both get there, selectivity costs a **median of 0 plies**, mean 2.2, max 8.

Split by phase, the two classes are completely different things:

| class | positions | first good depth, shipped | plies selectivity costs |
|---|---|---|---|
| middlegame severe | 7 | 1 to 3 | **0 on every one** |
| endgame, mate-class | 8 | 7 to 10, one never inside 12 | 3 to 8 |

The middlegame errors are not pruning losses. The search has the right move at depth 3,
loses it at depth 6, and has it again at a real budget: that is non-monotonicity in a
narrow band, not a mechanism removing a line. Only the endings pay real plies, and they pay
them for mating technique specifically. `8/4Q3/3B2k1/3K4/8/8/8/8 w` costs eight plies and
`6k1/r2r1p2/5Q2/5R2/2B4p/4P1P1/8/6K1 w` the shipped search never solves inside depth 12
while the floor solves it at depth 4.

The depth-6 failure rate that motivated the whole track is therefore largely an artefact of
the diagnostic budget. A rated move on this machine gets about 2.5 seconds at roughly 500
knodes/s, so over a million nodes, enough for depth fifteen and up on endings this small,
which absorbs a three-to-eight ply cost. That is why recovering those plies did not pay for
itself anywhere else.

### The baseline at a rated budget

The same 320 positions, the same merged reference pass, with the search given a million
nodes a move instead of 300k:

| budget | best | <50cp | severe | mate class | mean cp | 95% over clusters |
|---|---|---|---|---|---|---|
| 300k nodes | 250 | 58 | 6 | 6 | 5.7 | [3.7, 8.0] |
| **1M nodes** | **268** | 43 | **4** | 5 | **4.9** | [3.0, 7.0] |

Endgame regret 6.4cp, middlegame 5.1cp, opening 4.0cp; positions in check 10.3cp against
4.9cp for quiet ones. At the budget it actually plays with, this search picks the
reference's best move in **84%** of real game positions, against a candidate set of its own
four searches. Whatever is costing this engine rating on the leaderboard, on this evidence
it is not the selectivity of its search.

## 4. The evaluation pilot

### The fresh holdout

The trainer's own holdout is the tail of the training file, split at a byte offset, drawn
from the distribution the weights were fitted to. It answers "did this net fit this file".
`tools/holdout.py` builds the other kind: 204 fresh Stockfish games played here out of the
competition's own opening clusters, positions filtered exactly the way `tools/extract.py`
filters training data (not in check, best move neither a capture nor a promotion), labelled
now at 300k nodes with the top four moves and their scores, so a net can be scored on
whether it *orders* moves and not only on how close its number is.

6,095 positions, 204 games, 70 clusters. The opening clusters were then split in half by
hash **before anything was selected on either**: 35 clusters of holdout positions to judge
nets on, 35 disjoint clusters reserved for acceptance games. The evaluation half is 3,045
positions, of which 258 carry a mate label and are reported separately.

### The architecture and its runtime cost

Four own-king buckets per perspective, a fixed 2x2 partition of the perspective-oriented
board: bit 2 of the oriented square is the file half and bit 5 the rank half, giving

    0  own half, files a-d      1  own half, files e-h
    2  far half, files a-d      3  far half, files e-h

applied to the existing 768 features, so 3072 inputs. `nnue.py` reads the bucket count from
the file it loads, so one runtime plays a plain 768 net and a king-conditioned one with no
edit; at one bucket the whole mechanism folds to a compile-time constant zero and the
indices are exactly what they were. A king crossing its own bucket boundary invalidates that
perspective's accumulator and only that one, and is the single case that cannot be
incremental; the other perspective still takes its ordinary piece-square update.

`tests/test_king_buckets.py` checks the two agreements this depends on, at one bucket and at
four: the offline feature indices the trainer learns from equal the ones the engine plays
with, and every legal move's incremental update equals a full refresh, across castling both
wings, en passant, all four promotions and kings crossing bucket boundaries. Both pass.

Measured before committing to it, same engine, same positions, 400k nodes a move:

| net | file | ft_weight | import | peak RSS |
|---|---|---|---|---|
| 768, 1 bucket | 0.5 MB | 0.8 MB | 65.5 s | 1051 MB |
| 3072, 4 buckets | 1.8 MB | 3.1 MB | 63.6 s | 738 MB |

Import and memory are unaffected and the file is nowhere near the 50 MB limit.

Throughput was then measured properly, on the shipped network against the full-scale
candidate, in three alternating rounds so that drift in machine load cancels rather than
lands on one of them. Both searched 4,816,89x nodes, the same tree to within three nodes,
so this is throughput and nothing else:

| net | round 1 | 2 | 3 | mean |
|---|---|---|---|---|
| 768, 1 bucket | 595.3 | 593.7 | 586.0 | **591.7 knps** |
| 3072, 4 buckets | 564.5 | 566.3 | 563.3 | **564.7 knps** |

**4.6% slower**, which is the bucket refresh plus a weight table four times the size going
through the same cache. That is worth a few Elo, so the architecture has to be worth more
than that before it is worth shipping.

### The matched pair

Both runs identical in data, subset, split, seed, width, output buckets, loss, batch size,
learning rate and schedule; the only difference is `--king-buckets`. The pilot took the
first 100M records of the already shuffled training file, 8 epochs each, run concurrently.
The shipped network is the third reference: the same architecture at one bucket, 30 epochs
over all 269M records.

Trainer's own holdout, which is the training distribution:

| | epoch 1 | 2 | 3 | 6 | 8 |
|---|---|---|---|---|---|
| 1 bucket | 0.010867 | 0.010195 | 0.009873 | 0.009458 | 0.009364 |
| 4 buckets | 0.010543 | 0.009885 | 0.009552 | 0.009086 | **0.008991** |

Four buckets is ahead at every epoch and the gap widens: **3.6% lower loss** at the end.

The fresh holdout, non-mate positions only:

| net | MAE | sigmoid MSE | top-1 move | pairwise |
|---|---|---|---|---|
| pilot, 1 bucket, 8 epochs | **148.5cp** | 0.00708 | 31.4% | 55.5% |
| pilot, 4 buckets, 8 epochs | 155.1cp | **0.00705** | 31.2% | 55.1% |
| shipped, 1 bucket, 30 epochs | 156.4cp | **0.00675** | 31.2% | 55.6% |

**The advantage does not travel.** 3.6% on the training distribution becomes 0.4% off it,
the centipawn error is worse rather than better, and the ordering of candidate moves does
not improve at all. Float and quantised agree to within 0.3cp on every figure, so nothing
here is a quantisation artefact.

The first reported centipawn errors from this holdout were around 1000cp and were nonsense:
8.5% of the labels are saturated mates at the training convention's own sentinel, and a net
saying +900 where the label says +12800 contributes 11,900 to a centipawn mean. Every
centipawn figure above excludes them, and the mate rows are reported on their own line.

### At full scale

Both runs repeated over all 269,059,156 records for 12 epochs, again identical but for
`--king-buckets`, run concurrently. The shipped network is 30 epochs over the same file at
one bucket, so its trainer holdout number is directly comparable to these.

| net | trainer holdout | fresh holdout, sigmoid MSE | MAE | top-1 move | pairwise |
|---|---|---|---|---|---|
| full control, 1 bucket, 12 epochs | 0.008784 | 0.00670 | 153.7cp | 32.9% | 55.5% |
| **full candidate, 4 buckets, 12 epochs** | **0.008258** | **0.00626** | 154.1cp | **33.3%** | 55.8% |
| shipped, 1 bucket, 30 epochs | 0.008602 | 0.00674 | 156.3cp | 31.4% | 55.3% |

Fresh-holdout figures are the quantised networks, which is what would actually ship; float
and quantised agree to within 0.5cp and 0.0004 of sigmoid MSE on every row.

At equal epochs four buckets is 6.0% better in-distribution and **6.6% better out of it**,
and it beats a network of the same shape given two and a half times the optimisation. On
mate-labelled positions, where the trainer's target is saturated and the sigmoid figure is
the only meaningful one, it is 0.00166 against the shipped network's 0.00184. It is also the
only net here that improves move ordering, by 1.9 points of top-1.

**The pilot was wrong, and it was wrong in the direction the brief warned about.** At 8
epochs on 100M positions the advantage was 0.4% and the match was dead level; at 12 epochs
on 269M it is 6.6%. A small pilot can reject a candidate without establishing its converged
potential, and this one nearly did.

### The pilot match, which agreed with the pilot holdout

The cleanest form of the question is the two pilot nets against each other: byte-identical
source in both agent directories, matched training, only `weights/net.npz` different. Frozen
read-only before the match and untouched while it ran, 60 pairs from the 33 reserved
acceptance clusters, 20000ms + 200ms, three workers each pinned to its own physical core,
run length fixed before the run started.

    +26 =68 -26 over 120 games
    score 50.000% on 60 pairs in 33 clusters
    Elo -0, 95% CI [-33, +38]
    no agent failed to start, crash, flag or play an illegal move
    terminations: checkmate 52, threefold_repetition 49, fifty_moves 13, insufficient 6

Dead level. The interval does not exclude a moderate gain and 60 pairs was never going to
resolve single-digit Elo, but nothing in the run points to one, it agrees with the fresh
holdout, and it disagrees with the trainer's own holdout. Against a measured 3% throughput
cost, that is a change that does not pay.

### No data intervention is justified either

The brief's suggested fallback was underrepresented rook and pawn endings. Measured over the
training file against the holdout:

| | training | holdout |
|---|---|---|
| twelve men or fewer | 21.6% | 26% |
| rook endings | 5.3% | 7.4% |
| pawn races | 2.5% | 1.9% |

Pawn races are if anything over-represented in training. The evaluation is genuinely weakest
in exactly these textures, at 0.72 to 1.59 of error per unit of label magnitude against 0.54
in the middlegame, but that is not a coverage problem, and no training run was spent chasing
one.

## 5. What is correct, what is faster, what wins games

**Correct.** The accumulator now carries an own-king bucket and every feature index is
checked against the trainer's, at both bucket counts, over every legal move of ten positions
chosen to cover every move flag. Three real bugs were found and fixed on the way, all of them
in the measurement rather than in the engine:

- the UCI client read `lowerbound`/`upperbound` info lines as scores, so a node limit landing
  mid-iteration returned a number off by the aspiration window, and it sent only the current
  FEN, throwing away the history both draw rules read;
- training checkpoints recorded `torch.__version__` as the object it really is, which
  `weights_only=True` refuses, making every checkpoint written since that field was added
  unreadable by the quantiser. It had never been noticed because the shipped network predates
  the field;
- `test_move_state_hash_accumulator_and_features` built its reference features at the default
  bucket count rather than the runtime's, which passed silently at one bucket and failed 509
  of 512 neurons at four. It caught the very thing it exists to catch, once there was
  something to catch.

`tests/bench.py --depth 11` totals **925,557 nodes** with identical moves and scores to the
audited tree, so none of the runtime work changed what the engine searches. 79 tests pass
against the shipped network and against the candidate, ruff and mypy are clean over 41 files,
`make gate` passes and `submission.zip` builds at 684 KB unzipped.

**Faster.** Nothing. No change here was made for speed, and the one candidate that would have
cost speed was rejected before its cost decided anything.

**Wins games.** Nothing. The search candidate lost on move quality before reaching a match.
The network candidate reached one and drew it: 50.6% over 240 games, Elo +4, 95% CI
[-27, +32]. The build that plays is the build that was already playing.

## 6. The build recommendation

### The rule, declared before the acceptance match reported

Three lines of evidence on the king-conditioned network point in two directions, so the
rule for what to do about it is written down here before the deciding number arrived, and
not adjusted afterwards.

**For it.** On the fresh holdout it is 6.6% better than its own matched control and 7.1%
better than the shipped network in the loss it was trained on, and it is the only net in
this run that improves move ordering, by 1.9 points of top-1 against the reference.

**Against it.** It is **4.6% slower**, measured cleanly in alternating rounds on identical
trees, which is worth a few Elo on its own. And on independently judged move quality inside
a search, at a million nodes a move on the same 320 positions with both builds' moves in one
merged reference pass, it is slightly **worse**: 228 best moves against 239, and 8.4cp of
mean regret against 6.9cp, with overlapping intervals. Being better at predicting a deep
search's score on a quiet position is not the same as being a better leaf evaluation inside
a search, and this run is a concrete instance of the two coming apart.

So: **ship it only if the acceptance match interval excludes 50% on the positive side.** A
change carrying a measured cost has to show a measured gain, not merely fail to show a loss.
Anything less and the shipped network stays, and the king-bucket direction is recorded as
the best-supported next experiment rather than as this run's answer.

### The acceptance match

120 pairs, 240 games, declared before the start and not extended. The 35 reserved opening
clusters, which nothing was selected on. Both agent directories frozen read-only, source
byte-identical, differing only in `weights/net.npz`. 20000ms + 200ms, three workers each
pinned to a physical core.

    +63 =117 -60 over 240 games in 254 minutes
    score 50.625% on 120 pairs in 35 clusters
    Elo +4, 95% CI [-27, +32]
    zero agent failures
    terminations: checkmate 123, threefold_repetition 81, fifty_moves 20, insufficient 16

**The interval contains equality, so by the rule above the candidate does not ship.** A run
this size could not have resolved single-digit Elo and was not meant to; it was sized to
catch a material change in either direction, and it found none. What it does exclude is the
gain the fresh holdout hinted at being large: an improvement worth more than +32 Elo is
outside this interval.

An earlier attempt at this match, at five workers, is kept as
`INVALID-match-acceptance-kb4-vs-shipped-5workers.*`. It scored 54.375%, and it is not
reported as a result because 25 of its 240 games ended in an agent missing the 90 s import
budget rather than in chess. The harness flags that itself. The failures were symmetric, so
they diluted rather than biased, which means the true figure was more likely above 54.375%
than below it; the clean re-run at three workers says 50.625%, which is the reminder that a
compromised number is not a conservative number, it is just not a number.

### What ships

**Keep the current build.** No change measured in this run earns a place in it.

The engine as it stands: the search, the accumulator change (behaviour-preserving, proven by
an unchanged 925,557-node bench), and `weights/net.npz` unchanged at the 30-epoch 768-feature
network. `submission.zip` builds at 684 KB unzipped, imports in **56.8 s** of the 90 s budget
on a quiet machine at a peak of 1048 MB against the 2 GB limit, `make gate` passes and 79
tests pass.

The import margin is the one number here that should worry someone. 56.8 s of 90 s on this
machine, and the platform's hardware is not this machine. Nothing in this run made it worse
(the four-bucket network imports in 56.9 s, indistinguishably), but nothing made it better
either, and it is the failure that costs a whole game rather than a few centipawns.

### What the next run should do

**Do not repeat the search track.** Two independent measurements now say the same thing: at
the budget a rated move actually gets, this search picks the reference's best move in 84% of
real positions with under 5cp of mean regret, and the one coherent candidate built from
confirmed failures made it worse. The remaining failures are mating technique in endings,
they cost a median of zero plies once the budget is realistic, and they are not where the
rating is.

**The king-conditioned network is the best-supported open lead, and it is not settled.** The
evidence for it is a 7.1% better loss and 1.9 points of top-1 on a fresh holdout; against it,
4.6% throughput, slightly worse judged move quality in search, and a level 240-game match.
The honest reading is that it is worth roughly nothing at 12 epochs and might be worth
something trained to convergence, since its advantage grew from 0.4% at 8 epochs on 100M
positions to 6.6% at 12 epochs on 269M and had not stopped growing. The experiment that would
settle it is 30 epochs at four buckets against the shipped network's own 30, and a match of
at least 400 pairs on more than 35 clusters. That is a day of GPU and a day of CPU.

**The gap between the two holdouts is the most useful thing this run found.** Three points of
in-distribution loss improvement bought nothing measurable in games. Any future evaluation
work should be judged on the fresh holdout and on judged move quality in search before it is
allowed near a match, because the trainer's own holdout has now been shown to mislead in both
directions on the same architecture.

## 7. Reproducing this

Stockfish is an offline measurement tool here and never ships. `SF` below is the path to it.

    SF=/path/to/stockfish-linux-x86-64-universal

**The development set and its judgement.**

    uv run python tools/failures.py --pgn 'audit/*.pgn' 'logs/*.pgn' \
        --depth 6 --per-game 4 --limit 320 --cache cold --budget 300000 \
        --out audit/next-strength/run1/probe-base-budget.json
    uv run python tools/judge.py --probe audit/next-strength/run1/probe-cold.json \
        --engine $SF --nodes 250000 --ladder 4 16 --workers 5 \
        --out audit/next-strength/run1/judged-cold.json
    uv run python tools/classify.py --judged audit/next-strength/run1/judged-cold.json \
        --out audit/next-strength/run1/classes-cold.json

**The four previous survivors**, as bare fixtures with no history:

    uv run python tools/failures.py --fens audit/next-strength/run1/survivors.json \
        --depth 6 --budget 300000 --out audit/next-strength/run1/probe-survivors.json
    uv run python tools/judge.py --probe audit/next-strength/run1/probe-survivors.json \
        --engine $SF --nodes 250000 --ladder 4 16 --workers 2 \
        --out audit/next-strength/run1/judged-survivors.json

**Mechanism attribution and the depth trace.**

    uv run python tools/attribute.py --classes audit/next-strength/run1/classes-cold.json \
        --engine $SF --depth 6 --nodes 1000000 \
        --out audit/next-strength/run1/attribution.json
    uv run python tools/depthsweep.py --classes audit/next-strength/run1/classes-cold.json \
        --attribution audit/next-strength/run1/attribution.json --engine $SF \
        --max-depth 12 --node-cap 3000000 --out audit/next-strength/run1/depthsweep.json

**The rejected search candidate.** The settings it needed were reverted, so reproducing it
means restoring `RFP_MIN_MEN` and `RFP_RETURN_SHARE` as described in the comment beside the
cutoff in `search.py`, then:

    uv run python tools/trial.py --classes audit/next-strength/run1/classes-cold.json \
        --attribution audit/next-strength/run1/attribution.json --only-attributable \
        --engine $SF --depth 6 --nodes 200000 \
        --config baseline --config "endgame:CHESS_RFP_MIN_MEN=12" \
        --config "bounded:CHESS_RFP_RETURN_SHARE=3" \
        --config "both:CHESS_RFP_MIN_MEN=12,CHESS_RFP_RETURN_SHARE=3" \
        --out audit/next-strength/run1/trial-rfp.json
    CHESS_RFP_MIN_MEN=12 CHESS_RFP_RETURN_SHARE=3 uv run python tools/failures.py \
        --pgn 'audit/*.pgn' 'logs/*.pgn' --depth 6 --per-game 4 --limit 320 \
        --cache cold --budget 300000 --out audit/next-strength/run1/probe-cand-budget.json
    uv run python tools/compare.py \
        --probe audit/next-strength/run1/probe-base-budget.json \
        --probe audit/next-strength/run1/probe-cand-budget.json \
        --name baseline --name rfp-candidate --arm budget --engine $SF \
        --nodes 1000000 --workers 4 --out audit/next-strength/run1/compare-rfp.json

**The holdout and the cluster split.**

    uv run python tools/holdout.py --engine $SF --games 300 --plies 140 --per-game 30 \
        --spread 70 --workers 6 --out audit/next-strength/run1/holdout.json
    uv run python audit/next-strength/run1/split.py

**The matched training pair.** Identical but for `--king-buckets`.

    uv run python tools/train.py --data data/train.shuffled.bin \
        --checkpoints data/pilot-kb1 --positions 100000000 --epochs 8 \
        --king-buckets 1 --seed 0
    uv run python tools/train.py --data data/train.shuffled.bin \
        --checkpoints data/pilot-kb4 --positions 100000000 --epochs 8 \
        --king-buckets 4 --seed 0
    uv run python tools/quantise.py --checkpoint data/pilot-kb4/epoch008.pt \
        --out audit/next-strength/run1/net-pilot-kb4.npz
    uv run python tools/evalnet.py --holdout audit/next-strength/run1/holdout-eval.json \
        --checkpoint data/pilot-kb1/epoch008.pt --checkpoint data/pilot-kb4/epoch008.pt \
        --checkpoint data/checkpoints/epoch030.pt --npz weights/net.npz \
        --out audit/next-strength/run1/evalnet-pilot-v2.json

**Matches.** Agent directories are frozen and read-only before the match starts and are not
touched while it runs. `taskset` keeps two concurrent matches on disjoint physical cores,
because `--pin` hands out the first cores of whatever affinity mask it is given and two
matches would otherwise claim the same ones.

    taskset -c 0,1,2,6,7,8 env CHESSATHON_INCREMENT_MS=200 CHESSATHON_DEBUG=0 \
      uv run python tests/match.py \
        --agent snapshots/run1-pilot-kb4 --opponent snapshots/run1-pilot-kb1 \
        --openings audit/next-strength/run1/openings-acceptance.json \
        --games 120 --base-ms 20000 --increment-ms 200 --workers 3 --pin \
        --json audit/next-strength/run1/match-pilot-kb4-vs-kb1.json \
        --pgn audit/next-strength/run1/match-pilot-kb4-vs-kb1.pgn

**Verification.**

    uv run pytest -q                       # 79 tests
    uv run python tests/bench.py --depth 11 # must total 925,557 nodes
    make gate
    make zip
