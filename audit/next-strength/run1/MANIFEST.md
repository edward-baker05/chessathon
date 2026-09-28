# run1 manifest

Started 2026-09-10 08:30 BST. Deadline verified live from the rules page the same morning:
registration closes **Sep 11 11:00**, i.e. 26.5 h from the start of this run. That is the
budget the whole brief has to fit inside, and it is why the pilot is sized the way it is.

Verified from https://aichessathon.com/docs/rules.md and /agent-contract.md on 2026-09-10:
time control 120 s + 0.5 s per move per side, 90 s init budget, one core of an EPYC 9V74 at
2.60 GHz, 2 GB RAM, <= 50 MB unzipped, 10 uploads per team per day, 8 KB of log kept as the
first and last 4 KB. Networks must be self-trained; labelling with an existing engine is
allowed; starting from a published net is not.

Starting point measured here, not taken from the handoff: 77 tests pass (90.9 s).

## Log

- 08:30 run directory created.
- 08:41 `probe-cold.json`: 320 positions, 232 game clusters, sampled by game and phase with
  real histories. 140 of 320 give three different moves across shipped / floor / equal-node.
  The score gap runs both ways: 28 above +50cp and 28 below -50cp. The old tool counted only
  the first of those and called it a loss rate.
- 08:46 `holdout.json` generation started: 204 fresh Stockfish games from the opening
  clusters, quiet leaves only, labelled at 300k nodes with the top four moves each.
- 09:20 nnue.py carries an own-king bucket, read from the shipped file so one runtime plays
  both shapes. `tests/bench.py --depth 11` still totals **925,557 nodes** with identical
  moves and scores, so the accumulator change is behaviour-preserving on the shipped net.
- 09:25 `tests/test_king_buckets.py` passes at 1 and 4 buckets: offline features equal
  runtime features, and every legal move's incremental update equals a full refresh,
  including kings crossing a bucket boundary.
- 09:35 training throughput on the RTX 2060, machine loaded: 296k positions/s at one bucket,
  264k/s at four. The pilot is sized from that: 100M positions, 8 epochs, everything else
  matched, both runs concurrent.
- 08:54 `judged-cold.json`: Stockfish 19 scored every candidate move restricted to that move,
  on a cleared hash, with the game's real history behind it, 250k nodes escalating to 1M and
  4M where anything was in doubt. The shipped move is fine in 89.4% of the 320. The other
  27 are 13 severe centipawn losses, 6 walks into mate, 2 missed mates and 6 slower mates.
  Mean regret 17.6cp, 95% over game clusters [12.0, 24.3]. Nine positions never settled.
- 08:59 `classes-cold.json`: of the 27, 18 are ones the additive floor gets right, so
  selectivity is implicated; 6 are ones neither the floor nor a bigger budget fixes, which
  are evaluation or horizon and no pruning change will recover them.
- 09:06 `attribution.json`: over the 19 the floor gets right, added alone to the floor,
  **reverse futility breaks 6, more than any other mechanism**, and every one of the six is
  an ending. Removing it alone from the full search fixes 10. Late move pruning dominates
  the pair interactions. Ten of nineteen break from one mechanism alone and six only from a
  pair, so this is not purely a pair story.
- 09:15 measured on those six: static evaluations of +581 to +6334 against a margin of 75 a
  ply, with reverse futility firing 434 to 1923 times in a single depth-6 search. The
  subtree collapses into the static evaluation one move down, which is why the technique in
  a won ending is poor.
- 09:20 `search.py` gains two settings, both defaulting to exactly the old behaviour, so the
  bench is still **925,557 nodes**: `CHESS_RFP_MIN_MEN` withholds the cutoff below a piece
  count, and `CHESS_RFP_RETURN_SHARE` divides the part above beta that it may claim.
- 09:22 opening clusters split in half by hash, before anything is selected: 35 clusters of
  holdout positions to judge the pilot on, 35 disjoint clusters reserved for acceptance.
- 09:30 pilot training, matched except for the feature set. The king-conditioned run is
  ahead on the trainer's own holdout at every epoch so far: 0.010543 against 0.010867 after
  one, 0.009885 against 0.010195 after two. That holdout is the tail of the same file, so it
  is a sanity signal and not the comparison; `holdout-eval.json` is.
- 09:17 another agent session is running on this machine (a Codex sandbox re-judging the old
  failure set with Stockfish, working out of /tmp/chessathon-elo-audit-7f20dd1). It is not
  mine and it is left alone, but it takes a core and it means every wall-time measurement
  from here on is taken on a shared machine. Matches below are pinned to disjoint physical
  cores for that reason, and the import timing is re-taken on a quiet machine before
  packaging rather than trusted from a loaded one.
- 09:35 `trial-rfp.json`, the first result that cuts against the search track. On the 19
  confirmed failures, at fixed depth 6, the two changes together put 8 of 19 right and the
  endgame guard alone 6. **At a 300k-node budget the baseline already gets 14 of the 19
  right and every candidate is one position worse.** The depth-6 failures are largely an
  artefact of the diagnostic budget, and the changes cost nodes without buying anything back
  at a budget a rated game actually gives. Widened to the whole 320-position sample next,
  because 19 selected positions cannot settle it either way.
- 09:30 `judged-survivors.json`, the four gaps the previous run reported as surviving the
  floor's own node budget, put to the reference:
  * `3r2k1/5ppp/...` and `8/8/8/8/8/2q3k1/8/3K4 b` — **both searches play the same move and
    it is the reference's best**. These were never move-quality failures; the whole "gap"
    was two implementations disagreeing about what the same move is worth.
  * `8/6pk/4Kp2/...` — real but small: mate in 4 instead of mate in 2. Still winning.
  * `8/6pk/2N2p2/...` — the engine is *being* mated either way, and picks mate in 10 against
    over mate in 11 against.
  **None of the four is a lost win or a lost draw.** The previous headline rate rested on
  score differences that a move-quality measurement does not support.
- 09:45 `compare-rfp.json`, the measurement that ends the search track. Both builds probed on
  the same 320 positions at the same 300k-node budget, then every move either build produced
  scored once against the same merged candidate set, so neither gets a lower bar than the
  other:

      build            best  <50cp  severe  mate class   mean cp   95% over clusters
      baseline          250     58       6           6      5.7c      [ 3.7,  8.0]
      rfp-candidate     250     56       8           6     10.3c      [ 5.4, 16.3]

  and by phase the candidate's damage is exactly where it was aimed: **15.6cp of endgame
  regret against the baseline's 5.7**. The intervals overlap, so this is not a proven
  regression; what it is is a complete absence of any gain, in the one place the change was
  built to help. The change is reverted, the bench is **925,557 nodes** again, and the
  reasoning and the numbers are left in a comment beside the cutoff so the idea is not
  rebuilt from scratch. No match budget is spent on it.
- 09:47 the baseline's own number from that table is the headline the search track produced:
  at a 300k-node budget it plays the reference's best move in **250 of 320** positions with
  **5.7cp** of mean regret, 95% over game clusters [3.7, 8.0]. The depth-6 failure rate that
  motivated this track is largely an artefact of the diagnostic budget.
- 09:52 pilot finished. Trainer holdout, eight matched epochs on the same 100M positions:
  **one bucket 0.009364, four buckets 0.009026**, a 3.6% lower loss, and the gap widened
  monotonically from epoch 1. Full-scale matched pair launched on all 269M positions.
- 10:05 runtime cost of the wider input layer, measured before committing to it. Same engine,
  same positions, 400k nodes a move, one net swapped for the other:

      net                 ft_weight   import   peak RSS   throughput
      768 (1 bucket)         0.8 MB    65.5s    1051 MB    555 knps
      3072 (4 buckets)       3.1 MB    63.6s     738 MB    539 knps

  Import and memory are unaffected; the file is 1.9 MB against 0.5 MB, nowhere near the
  50 MB limit. Throughput costs about **3%**, which is the bucket refresh plus a colder
  weight table, and 3% is worth a few Elo. So the king-conditioned net has to be worth more
  than that before it is worth shipping.
- 10:07 the fresh holdout says the pilot's advantage does not travel. Centipawn figures over
  non-mate positions only; 258 of 3045 holdout labels are mates and are reported apart:

      net                       MAE   sigmoid MSE   top-1 move   pairwise
      pilot 1 bucket, 8 epochs  148.5     0.00708        31.4%      55.5%
      pilot 4 buckets, 8 epochs 155.1     0.00705        31.2%      55.1%
      shipped, 30 epochs        156.4     0.00675        31.2%      55.6%

  On the trainer's own holdout, drawn from the training distribution, four buckets was 3.6%
  better. On this one, generated fresh from games the training file never saw, it is 0.4%
  better and orders candidate moves no better at all. That gap between the two holdouts is
  the whole reason the fresh one was built.
- 10:20 no data intervention is justified either. The brief's own example was underrepresented
  rook and pawn endings; measured over the training file against the holdout, rook endings
  are 5.3% of training against 7.4% of the holdout and pawn races are 2.5% against 1.9%,
  so pawn races are if anything over-represented. Positions of twelve men or fewer are 21.6%
  of training against 26% of the holdout. The evaluation's weakness in these textures is not
  a coverage problem, and no training run is spent chasing one.
- 10:22 a plumbing match was invalidated by my own mistake and is kept as
  `INVALID-match-pipeline-check.*`: I rewrote `snapshots/run1-baseline` while that match was
  still drawing players from it. All three agent directories now hold byte-identical source
  and differ only in `weights/net.npz`, and nothing touches them while a match runs.
- 10:25 the controlled architecture match starts: pilot 4 buckets against pilot 1 bucket,
  identical source, matched training, 60 pairs from the reserved acceptance clusters, three
  workers each pinned to a physical core. This is the cleanest form of the question, because
  the two nets differ in nothing but the feature set.
- 10:28 `depthsweep.json` is the trace the brief asked for, in the unit that transfers to a
  game. For each confirmed failure, the shallowest depth at which each build first picks a
  move the reference is happy with:
  * the shipped search gets there in **18 of 19** within depth 12, at a **median depth of 3**;
  * where both get there, selectivity costs a **median of 0 plies**, mean 2.2, max 8;
  * every middlegame severe error is found at depth 1 to 3 **at zero cost** — those are not
    pruning losses at all, they are the search being non-monotonic and losing at depth 6 a
    move it had at depth 3;
  * every position where selectivity genuinely costs plies is an **ending**, and all but one
    are mate-class. `8/4Q3/3B2k1/3K4/8/8/8/8 w` costs 8 plies, `7K/q7/8/4k3/8/8/8/8 b` 7,
    and `6k1/r2r1p2/5Q2/5R2/2B4p/4P1P1/8/6K1 w` the shipped search never solves inside depth
    12 while the floor solves it at 4.
  A rated move on this machine is over a million nodes, which on endings this small is depth
  fifteen and up, so a three-to-eight ply cost is usually absorbed. That is the same
  conclusion the equal-node comparison reached, arrived at independently.
- 12:35 the controlled architecture match finished. Identical source, matched training,
  60 pairs on 33 reserved acceptance clusters, 20000ms + 200ms, three pinned workers:

      +26 =68 -26 over 120 games, score 50.000%
      Elo -0, 95% CI [-33, +38], no agent failed to start, crash, flag or play an illegal move
      terminations: checkmate 52, threefold_repetition 49, fifty_moves 13, insufficient 6

  Dead level, on a run length fixed before it started. The interval does not exclude a
  moderate gain, but nothing in it points to one, and it agrees with the fresh holdout and
  disagrees with the trainer's own. Set against a measured 3% throughput cost, the pilot
  verdict on king conditioning is that it does not pay.
- 13:04 the baseline re-measured at a million nodes a move, which is what a rated move gets
  on this machine: **268 of 320** best moves and **4.9cp** mean regret, 95% over clusters
  [3.0, 7.0], against 250 and 5.7cp at 300k nodes. Four severe errors and five mate-class in
  320 positions. The search is in good shape at the budget it actually plays with.
- 13:22 the full-scale matched pair finished, twelve epochs each over all 269M positions, and
  it **reverses the pilot's verdict**. On the fresh holdout, quantised, which is what would
  actually ship:

      net                          sigmoid MSE     MAE   top-1 move   mate positions
      full control, 1 bucket           0.00670   153.7        32.9%          0.00249
      full candidate, 4 buckets        0.00626   154.1        33.3%          0.00166
      shipped, 1 bucket, 30 epochs     0.00674   156.3        31.4%          0.00184

  Four buckets is 6.6% better than its own matched control and 7.1% better than the shipped
  network, on positions generated fresh from games the training file never saw, and it picks
  the reference's top move 1.9 points more often. The pilot at 8 epochs on 100M positions
  showed 0.4% and no ranking gain. The brief's warning that a small pilot can reject a
  candidate without establishing its converged potential is exactly what happened here, and
  had the pilot been the whole experiment the candidate would have been wrongly dropped.
- 13:22 acceptance match declared **before it starts**, and not extended afterwards
  whatever it says: 120 pairs, 240 games, the 35 reserved acceptance clusters, 20000ms +
  200ms, five workers each pinned to a physical core, frozen read-only directories whose
  source is byte-identical and which differ only in `weights/net.npz`.
- 13:49 the warm-cache regime exercised, so the capability is measured rather than merely
  implemented. 80 positions, each preceded by re-searching the same side's two previous
  moves with the table kept, which is the state a live game arrives in: 88.8% of shipped
  moves judged fine against 89.4% cold, and mean regret 11.4cp, 95% [5.7, 18.2], against
  17.6cp cold. It is a smaller and different sample so the two numbers are not a paired
  comparison, but the direction is the expected one and the cold figures quoted elsewhere
  are, if anything, the pessimistic ones.
- 14:16 running the suite against the candidate network found a real bug, in the test that
  exists to find exactly this: `test_move_state_hash_accumulator_and_features` unpacked its
  reference features at the default one bucket while the runtime was reading four, so 509 of
  512 neurons disagreed. The bucket count now comes from the runtime. 79 tests pass against
  the shipped network and against the candidate.
- 16:10 the first acceptance run is **invalid and marked so**, by the harness's own check
  rather than by my judgement: 23 `both_failed` and 2 `init` out of 240 games, 10.4%, decided
  by an agent missing the 90 second import budget rather than by chess. Five workers is ten
  agents compiling numba kernels at once, and the machine was not idle: Rocket League alone
  was holding 7.8 GB, with Brave, Steam, Discord and another agent session taking most of the
  rest. The failures are symmetric so they favour neither side, but a void pair scores 1.0/2
  and every one of them is a forced draw sitting in the number. Kept as
  `INVALID-match-acceptance-kb4-vs-shipped-5workers.*` with its own README.
- 16:11 re-declared and relaunched at **three workers**, the setting that played 120 games
  with no failure at all earlier today, with the stagger doubled to 30 s. Same 120 pairs,
  same reserved clusters, same control, length fixed before the start.
- 17:00 an independent line on the candidate network, and it does not agree with the holdout.
  Both networks searched the same 320 positions at a million nodes a move, then every move
  either produced scored once against the same merged candidate set:

      net            best  <50cp  severe  mate class   mean cp   95% over clusters
      shipped         239     71       4           6      6.9c      [ 4.8,  9.3]
      king-bucket     228     83       5           4      8.4c      [ 5.5, 11.6]

  The candidate is slightly **worse** here, with overlapping intervals. This is at a fixed
  node budget, so it is not the 3% throughput cost; it is that a network better at predicting
  a deep search's score on quiet positions is not automatically a better leaf evaluation
  inside a search. Note also that the shipped net's absolute figures here (239, 6.9cp) are
  worse than the 268 and 4.9cp reported when it was measured alone: adding a second build's
  moves to the candidate set raises the bar for both. Absolute regret is only comparable
  within one merged pass.
- 17:08 throughput measured properly, three alternating rounds so load drift cancels, both
  nets searching the same 4,816,89x nodes: **591.7 knps at one bucket against 564.7 at four,
  a 4.6% cost**. Tighter and slightly larger than the 3% the pilot nets suggested.
- 17:29 the rule for what to do about the candidate network written into the report **before**
  the acceptance match reported, so that the decision cannot be fitted to the number: ship it
  only if the interval excludes 50% on the positive side. Three lines of evidence point two
  ways — better on the fresh holdout and at move ordering, 4.6% slower, and slightly worse on
  independently judged move quality inside a search — and a change carrying a measured cost
  has to show a measured gain rather than merely fail to show a loss.
- 20:25 the acceptance match, clean, at the declared length:

      +63 =117 -60 over 240 games, score 50.625% on 120 pairs in 35 clusters
      Elo +4, 95% CI [-27, +32], zero agent failures
      terminations: checkmate 123, threefold_repetition 81, fifty_moves 20, insufficient 16

  The interval contains equality. By the rule written down before the run, **the candidate
  network does not ship.** Its case rested on a 7% better fresh-holdout loss and 1.9 points
  of top-1; against that it is 4.6% slower, slightly worse on independently judged move
  quality inside a search, and now level over 240 games.
- 20:30 final verification on a quiet machine. Agent import **56.8 s** of the platform's 90 s
  with the shipped network and **56.9 s** with the candidate, so the wider input layer costs
  nothing there; peak resident 1048 MB against the 2 GB limit. `make gate` passes, ruff and
  mypy clean over 41 files, bench still **925,557 nodes**, `submission.zip` builds.
- 20:35 run closed. `weights/net.npz` is unchanged at `57ab14b4...`, the 30-epoch 768-feature
  network that was already playing. Nothing from this run ships.

  A note on version control: `audit/` is gitignored in this repository, so everything in this
  directory is on disk only. The source changes are all committed — the working tree matches
  HEAD exactly, and the committed `search.py` carries the rejected candidate as a comment
  rather than as code. Another agent session was committing to this branch during this run
  ("Updated gitignore", "Cleared git cache"), and its cache-clearing commits swept the new
  tools in at their final state; that was checked rather than assumed.

## Sep 11, run 2: epochs and bucket count

- 06:31 two matched 16-epoch runs over all 269M positions, identical but for the bucket
  count, both selecting `best.pt` on the fresh holdout rather than on the training file's
  own tail. Final quantised numbers on the fresh holdout, non-mate rows:

      net                                  sigmoid MSE   top-1   best epoch
      run2, 4 buckets                          0.00609   33.1%           10
      run2, 8 buckets                          0.00635   31.7%           13
      run1, 4 buckets, 12-epoch schedule       0.00626   33.3%           12
      shipped, 1 bucket, 30 epochs             0.00674   31.4%           30

  **Eight buckets is worse than four**, by 4.3% in loss and 1.4 points of top-1, while being
  2.4% *better* on the in-file holdout. The same divergence as the epoch axis: more capacity
  fits the training file better and generalises worse. Eight also overflowed int32 at QA 255
  and had to quantise at 181. The bucket axis is closed at four.
- 06:31 the epoch axis is closed too. Both run-2 curves reach their best fresh number at
  epoch 10 and 13 of 16 and drift up afterwards, while the in-file number improves
  monotonically to the end. Yesterday's 30-epoch shipped net is 2.1% better in-file and 6.2%
  worse on the fresh holdout than a 12-epoch retrain of the same architecture.
- 07:10 the best network measured in either run is run2 4 buckets at epoch 10: **9.6% better
  than the shipped network** on the fresh holdout and 1.7 points better at top-1. Its
  quantisation proof passes at QA 255 with only 0.3% of int32 headroom, which is safe by the
  tool's own worst-case bound but is the tightest margin any net here has had.
- 07:12 a 60-pair regression screen started, candidate against the shipped network, source
  byte-identical in both frozen directories. It cannot establish a gain at this length; it
  can catch a disaster. No swap is recommended on it either way.
- 07:40 the independent move-quality check, repeated on the better network, says the same
  thing it said yesterday. Both builds searching the same 320 positions at a million nodes,
  every move either produced scored once against one merged candidate set:

      net            best  <50cp  severe  mate class   mean cp   95% over clusters
      shipped         239     71       4           6      6.5c      [ 4.5,  8.9]
      run2 4 buckets  235     70       7           8      8.9c      [ 5.9, 12.3]

  Slightly worse, intervals overlapping, and now **replicated on two separately trained
  king-bucket networks**. The dissociation is the finding: a network 9.6% better at
  predicting a deep search's score on quiet positions picks slightly worse moves when a
  search actually uses it as a leaf evaluation. The fresh holdout is a better metric than the
  trainer's own, and it is still not measuring the thing that decides games.
- 08:37 the regression screen is **invalid and was stopped at 39 of 60 pairs**. Ten of 78
  games missed the 90 s import budget, and four of those were one-sided `init` failures that
  hand the opponent a win, so it was biased rather than merely diluted. Cause found while it
  ran: Steam's `fossilize_replay` shader pre-caching, three processes at ~93% of a core each
  plus Spotify, taking the machine to a load average of 13 on six physical cores. Nothing
  about either engine is implicated. Not re-run: too close to the deadline for another two
  hours of games, and the recommendation does not turn on it.
- 08:40 **Final recommendation: do not swap the network. Ship the build that is already
  playing.** The candidate is 9.6% better on the fresh holdout and 1.7 points better at
  top-1, and that is the only measurement that favours it. Against it: two independently
  judged move-quality comparisons, on two separately trained king-bucket networks, both put
  it slightly worse in search; a clean 240-game match on the previous version returned Elo
  +4, 95% CI [-27, +32]; and its quantisation proof passes at QA 255 with 0.3% of int32
  headroom, the tightest margin of any network here. A static-evaluation metric that has
  twice failed to convert is not grounds for changing what plays a locked final.

## Sep 11, why more capacity keeps not helping

- 09:05 Four capacity increases have now failed: king buckets, HalfKA, a wider L1, more
  epochs. That pattern is itself evidence, so the labels and the target were measured
  instead of the model.

  **The labels are excellent where it matters and poor where it does not.** 300 real training
  records were decoded back into positions, restricted to positions where neither king is on
  its home square so that the missing castling rights reconstruct exactly, and re-evaluated
  at 2M nodes:

      |deep score|   positions   stored label disagrees with a deep search by
      0-50                 115                                            7cp
      50-150                49                                           17cp
      150-400               51                                           73cp
      400-1000              77                                          175cp
      1000+                  8                                         1320cp

  Sign disagreements 1.0%, median disagreement 22cp. The convention and the extraction are
  sound. In balanced positions the labels are essentially exact.

  **The network is ten times worse than that floor in exactly those positions.** On the fresh
  holdout the shipped net's mean absolute error where |label| < 50 is **72cp**, against a
  7cp label floor, and it outputs an average magnitude of 77cp where the truth averages 16.
  It is systematically overconfident about balanced positions. That is why move ordering sits
  near chance at 31% top-1 and 55% pairwise: sibling moves differ by tens of centipawns and
  the evaluation has 72cp of noise.

  **The loss is spent on the noisy half.** Sigmoid-space squared error by band:

      |label|        share of positions   share of loss   label noise
      0-50                         33%             24%           7cp
      50-150                       18%             11%          17cp
      150-400                      17%             22%          73cp
      400-1000                     21%             40%         175cp
      1000+                        11%              4%     saturated

  Positions at |label| >= 150 are 48% of the data, take **65% of the loss**, and carry 73 to
  1320cp of label noise. A tenth of the file is the mate sentinel at +/-12800, where the
  sigmoid target is 1.0 to float precision and the gradient is zero: pure wasted compute.

  So the model is at the noise floor of the targets it is being given, over most of the
  gradient. Every capacity increase buys a better fit to that noise, which is precisely the
  signature observed throughout: the in-file holdout improves monotonically while the fresh
  holdout stalls or regresses, at 8 buckets against 4 and at 30 epochs against 12.

  **The next experiment is a target change, not an architecture change.** Filter or clamp the
  training labels to the well-labelled band, around |label| <= 600, which keeps roughly 80%
  of the file and removes the noisiest fifth along with the zero-gradient mate sentinels.
  The number to watch is not the loss, which will move for trivial reasons once the target
  distribution changes, but the mean absolute error where |label| < 50 and the top-1 rate.
