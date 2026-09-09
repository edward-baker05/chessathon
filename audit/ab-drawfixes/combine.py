"""Pool two disjoint match chunks and take the interval over cluster means.

The two chunks were run on halves of the opening set that share no cluster, so the pairs
pool without double counting a source game. The arithmetic is the same as tests/match.py:
average within a cluster first, then a plain 95% normal interval over the cluster means.
"""
import json
import math
import pathlib
import statistics
import sys

pairs: list[dict] = []
terms: dict[str, int] = {}
wins = draws = losses = 0
for path in sys.argv[1:]:
    data = json.loads(pathlib.Path(path).read_text())
    pairs.extend(data["pairs"])
    s = data["summary"]
    wins += s["wins"]
    draws += s["draws"]
    losses += s["losses"]
    print(f"{path.split('/')[-1]:14s} {s['score']:.3%} on {s['pairs']} pairs, "
          f"{s['clusters']} clusters, +{s['wins']} ={s['draws']} -{s['losses']}")

for pair in pairs:
    for termination in pair["terminations"]:
        terms[termination] = terms.get(termination, 0) + 1

by_cluster: dict[str, list[float]] = {}
for pair in pairs:
    by_cluster.setdefault(pair["cluster"], []).append(pair["score"] / 2.0)
units = [statistics.mean(v) for v in by_cluster.values()]
score = statistics.mean([p["score"] / 2.0 for p in pairs])
deviation = statistics.stdev(units) if len(units) > 1 else 0.0
error = 1.96 * deviation / math.sqrt(len(units))
low, high = max(0.0, score - error), min(1.0, score + error)

def elo(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)

print(f"\ncombined: +{wins} ={draws} -{losses} over {len(pairs) * 2} games")
print(f"score {score:.3%} on {len(pairs)} pairs in {len(units)} clusters "
      f"({len(pairs) / len(units):.1f} pairs each)")
print("terminations:", ", ".join(f"{k} {v}" for k, v in sorted(terms.items())))
failed = sum(v for k, v in terms.items() if k in {"init", "crash", "flag", "illegal"})
print(f"agent failures (init, crash, flag, illegal): {failed}")
print(f"95% CI [{low:.3%}, {high:.3%}]")
print(f"Elo {elo(score):+.0f}   95% CI [{elo(low):+.0f}, {elo(high):+.0f}]")
print("\nThe interval contains equality." if low <= 0.5 <= high else
      "\nThe interval excludes equality.")
