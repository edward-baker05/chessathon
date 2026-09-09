Equal-time paired match: the draw-handling fixes against the audited baseline.

agent     snapshots/draw-fixes      = commit 3ac83e9 (the four fixes)
opponent  snapshots/pre-draw-fixes  = commit 8a192de (the audited tree)
weights   identical in both; only search.py, movegen.py and tt.py differ

Two workers, --pin, 20000ms + 200ms. Run length was fixed before the run started:
two chunks of 20 pairs on halves of audit/ab-openings.json that share no cluster,
so the pairs pool without counting one source game twice. Two workers rather than
six because each agent process peaks at 1.03 GB resident, and eight of them
exhausted memory and pushed agent startup past the 90 second budget.

chunk-a.json   45.000% on 20 pairs, 15 clusters, +4 =28 -8
chunk-b.json   53.750% on 20 pairs, 17 clusters, +6 =31 -3

combined: +10 =59 -11 over 80 games
score 49.375% on 40 pairs in 32 clusters (1.2 pairs each)
terminations: checkmate 21, fifty_moves 6, insufficient_material 7, stalemate 1, threefold_repetition 45
agent failures (init, crash, flag, illegal): 0
95% CI [43.754%, 54.996%]
Elo -4   95% CI [-44, +35]

The interval contains equality.

A 4% throughput cost is worth roughly 3 to 5 Elo, which 40 pairs cannot resolve.
This run was sized to detect a large regression, not to measure the difference.
It found none, and no agent failed to start, crash, flag or play an illegal move.
