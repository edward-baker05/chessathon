"""An outside opinion about the moves `tools/failures.py` collected.

Two of our own searches disagreeing says nothing about which is right. This asks a third
party that is not ours and has no stake in the answer, and it asks it the one way that
makes the answers comparable: every move under test, including the reference's own choice,
gets its own search restricted to that move, on a cleared hash, with the same node budget
and the same history behind it. Scores taken from a single unrestricted search are not
comparable that way, because the move the search believed in got most of the budget and
the others got whatever was left.

    uv run python tools/judge.py --probe run1/probe.json --out run1/judged.json \
        --engine /path/to/stockfish --nodes 250000 --workers 6

What comes out per position is a score for each candidate, the best of them, and each
candidate's regret against that best. Regret is only ever in centipawns between two
centipawn scores. A mate is not a large number here: missing one, or walking into one, is
counted as its own class and never averaged into a centipawn mean.

Disagreements are re-asked at larger budgets. Single-PV scores from separate searches are
noisy, and a difference that does not survive four times and sixteen times the nodes is
reported as unresolved rather than as a finding. The budgets are a starting point chosen
against measured throughput, not a claim to ground truth.
"""

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audit"))

import chess  # noqa: E402
import chess.engine  # noqa: E402
from uci import Engine  # noqa: E402

from tools.failures import board_of  # noqa: E402

# Regret above this is a serious error, and it is also what makes a position worth
# re-asking at a larger budget. 50cp is roughly the difference an engine of this strength
# would trade a whole tempo for; below it, single-PV noise is a real competitor.
SEVERE_CP = 100
ESCALATE_CP = 50


def score_of(info: dict[str, Any], turn: chess.Color) -> dict[str, Any]:
    """One analysis result, with mates kept as mates.

    A mate converted to a big centipawn number averages into a mean and drags it wherever
    the sentinel happens to sit. Everything downstream branches on `kind` instead.
    """
    pov = info["score"].pov(turn)
    if pov.is_mate():
        return {"kind": "mate", "mate": pov.mate(), "cp": None}
    return {"kind": "cp", "cp": pov.score(), "mate": None}


def rank(score: dict[str, Any]) -> tuple[int, int]:
    """A total order over scores, sortable, with no centipawn conversion anywhere.

    Delivering mate beats every centipawn score and sooner beats later; being mated loses
    to every centipawn score and later beats sooner.
    """
    if score["kind"] == "mate":
        mate = int(score["mate"])
        return (2, -mate) if mate > 0 else (0, -mate)
    return (1, int(score["cp"]))


def regret(best: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    """How much worse `got` is than `best`, in the only terms that are comparable.

    `cp` is filled in only when both are centipawn scores. Everything else is a class:
    a mate that was available and not taken, a mate distance thrown away, or a move that
    hands the opponent a mate. Those are counted, never averaged.
    """
    if rank(got) >= rank(best):
        return {"cp": 0, "klass": "best"}
    if best["kind"] == "mate" and int(best["mate"]) > 0:
        if got["kind"] == "mate" and int(got["mate"]) > 0:
            return {"cp": None, "klass": "slower_mate",
                    "plies": int(got["mate"]) - int(best["mate"])}
        return {"cp": None, "klass": "missed_mate"}
    if got["kind"] == "mate" and int(got["mate"]) < 0:
        return {"cp": None, "klass": "into_mate"}
    if best["kind"] == "cp" and got["kind"] == "cp":
        return {"cp": int(best["cp"]) - int(got["cp"]), "klass": "cp"}
    # best is being mated and got is a centipawn score: ranked worse only if `rank` said
    # so, which it cannot have. Kept explicit rather than falling through silently.
    return {"cp": None, "klass": "incomparable"}


class Pool:
    """One Stockfish per worker thread, created once and reused."""

    def __init__(self, path: str, hash_mb: int) -> None:
        self.path = path
        self.hash_mb = hash_mb
        self.local = threading.local()
        self.engines: list[Engine] = []
        self.lock = threading.Lock()

    def engine(self) -> Engine:
        engine = getattr(self.local, "engine", None)
        if engine is None:
            engine = Engine(self.path)
            engine.configure({"Threads": 1, "Hash": self.hash_mb})
            with self.lock:
                self.engines.append(engine)
            self.local.engine = engine
        return engine

    def close(self) -> None:
        for engine in self.engines:
            engine.quit()


def restricted(engine: Engine, board: chess.Board, move: chess.Move,
               nodes: int) -> dict[str, Any]:
    """This one move, on a cleared hash, with the game behind it."""
    engine.new_game()
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=[move])
    score = score_of(info, board.turn)
    score["depth"] = int(info.get("depth", 0))
    score["pv"] = [m.uci() for m in info.get("pv", [])][:8]
    return score


def assess(row: dict[str, Any], pool: Pool, nodes: int, ladder: list[int]) -> dict[str, Any]:
    """Every candidate scored under identical conditions, escalated where it matters."""
    board = board_of(row)
    engine = pool.engine()

    # The reference's own preference, unrestricted, so the candidate set contains a move
    # neither of our searches proposed. Its score here is not used; it is re-measured under
    # the same restricted conditions as everything else.
    engine.new_game()
    top = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    reference = top["pv"][0] if top.get("pv") else None

    moves: list[chess.Move] = []
    for uci in row["candidates"]:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves and move not in moves:
            moves.append(move)
    if reference is not None and reference not in moves:
        moves.append(reference)

    out: dict[str, Any] = {"reference_move": reference.uci() if reference else None,
                           "legal_candidates": [m.uci() for m in moves], "rounds": []}
    budgets = [nodes, *[nodes * step for step in ladder]]
    verdicts: list[str] = []
    for budget in budgets:
        scores = {m.uci(): restricted(engine, board, m, budget) for m in moves}
        best_uci = max(scores, key=lambda uci: rank(scores[uci]))
        best = scores[best_uci]
        regrets = {uci: regret(best, score) for uci, score in scores.items()}
        out["rounds"].append({"nodes": budget, "best": best_uci,
                              "scores": scores, "regret": regrets})
        shipped = str(row["shipped"]["move"])
        hurt = regrets.get(shipped, {"cp": 0, "klass": "absent"})
        verdicts.append(verdict_of(hurt))
        # Nothing to re-ask if no candidate is materially behind the best.
        worst = max((r["cp"] or 0) for r in regrets.values())
        classes = {r["klass"] for r in regrets.values()}
        if worst < ESCALATE_CP and not (classes & {"missed_mate", "into_mate", "slower_mate"}):
            break
    out["verdict"] = verdicts[-1]
    # A finding has to survive a bigger budget to be a finding. One round means nothing was
    # in doubt; two or more that disagree means the reference did not settle it.
    out["resolved"] = len(verdicts) == 1 or verdicts[-1] == verdicts[-2]
    return out


def verdict_of(hurt: dict[str, Any]) -> str:
    """What the reference says about the move the shipped search actually played."""
    if hurt["klass"] in ("missed_mate", "into_mate", "slower_mate"):
        return str(hurt["klass"])
    cp = hurt["cp"] or 0
    if cp >= SEVERE_CP:
        return "severe"
    if cp >= ESCALATE_CP:
        return "material"
    return "ok"


def cluster_bootstrap(rows: list[dict[str, Any]], draws: int = 4000,
                      seed: int = 20260910) -> tuple[float, float, float]:
    """Mean cp regret with an interval that resamples games, not positions.

    Four positions from one game are one game seen four times. Resampling positions would
    report an interval about half as wide as the evidence supports.
    """
    import random
    by_cluster: dict[str, list[float]] = {}
    for row in rows:
        by_cluster.setdefault(str(row["cluster"]), []).append(float(row["_cp"]))
    keys = sorted(by_cluster)
    if not keys:
        return (0.0, 0.0, 0.0)
    means = [statistics.fmean(by_cluster[k]) for k in keys]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        pick = [means[rng.randrange(len(means))] for _ in range(len(means))]
        samples.append(statistics.fmean(pick))
    samples.sort()
    return (statistics.fmean(means), samples[int(0.025 * draws)], samples[int(0.975 * draws)])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--nodes", type=int, default=250_000)
    parser.add_argument("--ladder", type=int, nargs="*", default=[4, 16],
                        help="multiples of --nodes to escalate a disagreement to")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.out.exists():
        parser.error(f"{arguments.out} exists; evidence is never overwritten, pick a new name")
    probe = json.loads(arguments.probe.read_text())
    rows = probe["positions"][: arguments.limit or None]
    pool = Pool(arguments.engine, arguments.hash)
    print(f"{len(rows)} positions, {arguments.nodes:,} nodes per restricted move, "
          f"ladder {arguments.ladder}, {arguments.workers} workers")

    started = time.perf_counter()
    done = 0
    lock = threading.Lock()

    def work(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        result = dict(row)
        result["judgement"] = assess(row, pool, arguments.nodes, arguments.ladder)
        with lock:
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(rows)}  {time.perf_counter() - started:.0f}s", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=arguments.workers) as runner:
        judged = list(runner.map(work, rows))
    pool.close()

    report(judged)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "provenance": {
            "probe": str(arguments.probe), "probe_provenance": probe["provenance"],
            "engine": arguments.engine, "nodes": arguments.nodes,
            "ladder": arguments.ladder, "hash_mb": arguments.hash,
            "severe_cp": SEVERE_CP, "escalate_cp": ESCALATE_CP,
            "seconds": round(time.perf_counter() - started, 1),
        },
        "positions": judged,
    }, indent=1) + "\n")
    print(f"wrote {arguments.out}")
    return 0


def report(judged: list[dict[str, Any]]) -> None:
    """What the reference actually said, by stratum and by class."""
    counts: dict[str, int] = {}
    unresolved = 0
    cp_rows: list[dict[str, Any]] = []
    for row in judged:
        judgement = row["judgement"]
        verdict = str(judgement["verdict"])
        counts[verdict] = counts.get(verdict, 0) + 1
        if not judgement["resolved"]:
            unresolved += 1
        final = judgement["rounds"][-1]
        hurt = final["regret"].get(str(row["shipped"]["move"]))
        if hurt and hurt["klass"] in ("best", "cp"):
            cp_rows.append({**row, "_cp": hurt["cp"] or 0})

    print(f"\npositions judged              {len(judged)}")
    print(f"unresolved at the top budget  {unresolved}")
    print("verdict on the shipped move:")
    for name in ("ok", "material", "severe", "slower_mate", "missed_mate", "into_mate"):
        if counts.get(name):
            print(f"  {name:<14} {counts[name]:>4} ({counts[name] / len(judged):.1%})")
    if cp_rows:
        mean, low, high = cluster_bootstrap(cp_rows)
        print(f"\nmean regret of the shipped move, centipawn cases only ({len(cp_rows)} of "
              f"{len(judged)}): {mean:.1f}cp, 95% over games [{low:.1f}, {high:.1f}]")
    for key in ("stratum", "phase"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in judged:
            groups.setdefault(str(row[key]), []).append(row)
        print(f"\nby {key}:")
        for name, group in sorted(groups.items()):
            bad = sum(1 for r in group
                      if r["judgement"]["verdict"] not in ("ok", "material"))
            cps = [r["_cp"] for r in cp_rows if str(r[key]) == name]
            shown = f"{statistics.fmean(cps):.1f}cp" if cps else "-"
            print(f"  {name:<12} {len(group):>4} positions, mean regret {shown:>8}, "
                  f"{bad} severe or mate")


if __name__ == "__main__":
    raise SystemExit(main())
