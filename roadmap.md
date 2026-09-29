Tier 1: an hour or two of code each, likely +10 to 30 Elo apiece

1. Use the transposition table in qsearch. qsearch (src/search.cpp:~302) never probes or stores the table, and at 2M nps most nodes are qsearch nodes. Probe for cutoffs, use the table's static eval for the stand-pat, and store the result at depth 0. This is probably the cheapest win you have.
2. Correction history. Keep a table indexed by [stm][pawn_key % N] (and optionally non-pawn keys) that learns how far the static eval is from the search result, and adjust static_eval before pruning. It has been one of the largest cheap gains in recent engines, and it partly offsets a network trained on another engine's evals.
3. History gravity and LMR driven by history.
   - update_history currently adds a bonus and halves the whole table on overflow. Switch to the gravity update, h += bonus - h*|bonus|/MAX.
   - Make the LMR reduction depend on history: reduction -= (history + cont_hist) / K. At the moment only the counter-move and killer bands change the reduction.
4. Capture history. Order captures by MVV + capthist[piece][to][victim] rather than pure MVV-LVA, and update it on cutoffs. Today update_history only runs for quiet moves (search.cpp:~480). Once that's in, apply a mild LMR to late bad captures as well.

Tier 2: about half a day, likely +20 to 40 Elo

5. Re-test singular extensions. Commit 86066cb dropped them because they "measured negative", but that was the Python engine, which searched far shallower. The C++ engine reaches depth 14 to 16 in milliseconds, which is where singular extensions pay off. Include double extensions and negative extensions. You'll need an excluded-move slot on the stack, and the table must not be written during the excluded search.
6. Small search tweaks: make the reverse futility margin depend on improving, let the null-move reduction depend on eval minus beta more finely, and add ProbCut. These are cheap to write, but expect +3 to 10 each, so they cost more SPRT time per Elo.

Tier 3: days of mostly CPU time, the largest single gain (+50 to 150 Elo across generations)

7. Self-play data generation, then training on eval and result together. This is the main path to a stronger engine, and it fits your hardware:
   - Add a datagen command to the engine: 8 to 10 random opening plies, then play the game out at about 5k nodes a move, recording the FEN, the score and the final result, and filtering captures and positions in check as extract.py already does.
   - At about 1.5M nps per thread on 12 threads, 100M positions is roughly 8 to 10 hours.
   - Train on λ·eval + (1-λ)·result in train.py. The game result is the signal your current data can't give you, and the positions come from the engine's own games rather than human analysis.
   - Start the first generation from the current net, then iterate. Each generation of net and data typically gains tens of Elo.

   This keeps you within "only ship a network you trained yourself".
8. SPSA tuning of the search constants. There are dozens of hand-picked margins: 75·depth, 200·depth, 100 + 90·depth, the LMP cap, the SEE thresholds, the LMR table, and so on. Expose them as UCI options and run SPSA with fastchess. Expect +15 to 40, but only after tiers 1 and 2 are in; otherwise you tune constants for code that is about to change.

Tier 4: only once self-play data exists

9. Network architecture:
   - King-bucketed inputs: codex/elo-recovery has a Python prototype, but it never reached the C++ engine.
   - A wider L1 of 768 or 1024.
   - A hidden layer after the accumulator, which only becomes affordable with a sparse or SIMD-friendly implementation.

   All of these need more and better data to pay off. On the current 270M eval-only positions they are likely to be flat.
