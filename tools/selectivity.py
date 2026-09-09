"""Which selective mechanism removed the move.

`search.set_pruning(False)` answers "what does plain alpha-beta say here", and that is all
it answers: it also turns off the check extension and the transposition cutoffs, so when
its result differs from the real search, three things changed and the difference names
none of them. This runs the same position once per mechanism instead, with that one
mechanism off and every other one still running, so a difference names exactly one.

    uv run python tools/selectivity.py
    uv run python tools/selectivity.py --positions cases.json --json out.json
    uv run python tools/selectivity.py --root-depth 8

Two probes per case, because they answer different questions.

**Window probe.** One `negamax` call at a fixed depth and a fixed window, which is what an
interior node of a real search is handed. It is exact and it is cheap, and it is the form
the recheck fixtures are written in. A narrow window returns a bound, not a score, so a
number that differs from the oracle is only a failure when it is on the wrong side of the
window: a fail-high of 210 in [0,1] is correct and a fail-low of 210 in [1210,1211] is not.
The table says which.

**Root probe.** A whole iterative-deepening search to a fixed depth, which is what decides
the move the engine would actually play. Slower, and the one that says whether removing a
mechanism changes anything a game would notice.

Neither is a measurement of strength. A mechanism that removes a good move here may still
be worth its cost over a match; `tests/match.py` is the only thing in this repository that
can say so. What this produces is the list of candidates to take there.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import numpy as np  # noqa: E402

import bitboard as bb  # noqa: E402
import movegen  # noqa: E402
import position  # noqa: E402
import search  # noqa: E402
import tt  # noqa: E402

# Cases carry their own window because the window is half the question. `move` names the
# move the case exists to protect; leave it out and the oracle's own choice is used.
CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "quiet-mate",
        "fen": "rrrrr2k/8/5KQ1/8/8/8/8/8 w - - 0 1",
        "move": "g6g7",
        "depth": 1, "alpha": 1210, "beta": 1211, "is_pv": False,
        "note": "Qg7 mates. Artificial material, so this is a fixture and not a frequency.",
    },
    {
        "name": "quiet-mate-valid-window",
        "fen": "rrrrr2k/8/5KQ1/8/8/8/8/8 w - - 0 1",
        "move": "g6g7",
        "depth": 1, "alpha": 0, "beta": 1, "is_pv": False,
        "note": "The same node in a window the static score fails high in. Not a failure.",
    },
    {
        "name": "quiet-sacrifice-check",
        "fen": "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
        "move": "d6d1",
        "depth": 5, "alpha": 300, "beta": 301, "is_pv": False,
        "note": "...Qd1+ mates in three and lands on an empty square, so every quiet "
                "pruning sees a queen thrown away and none of them knows it gives check.",
    },
    {
        "name": "kiwipete",
        "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "depth": 7, "alpha": -200, "beta": -199, "is_pv": False,
    },
    {
        "name": "rook-endgame",
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "depth": 8, "alpha": 200, "beta": 201, "is_pv": False,
    },
    {
        "name": "near-fifty",
        "fen": "7k/8/8/8/8/8/8/KR6 w - - 99 1",
        "depth": 3, "alpha": -100, "beta": 100, "is_pv": False,
        "note": "Inside the rule-clock band, so the prunings are off by construction.",
    },
)


def fresh(work: search.Work) -> None:
    """An empty engine. Without this, case N inherits whatever case N-1 left behind."""
    tt.tt_clear(work.table)
    search.clear_tables(work)
    search.set_game_history([], work)
    search.set_pruning(True, work)
    search.set_mechanisms(None, work)


def principal_variation(board: chess.Board, work: search.Work, limit: int = 12) -> list[str]:
    """The line the table believes in, walked from the root and checked for legality."""
    line: list[str] = []
    seen: set[int] = set()
    walk = board.copy(stack=False)
    for _ in range(limit):
        position.encode(walk, work.state[3], work.mail[3])
        key = work.state[3][bb.KEY]
        if int(key) in seen:
            break
        seen.add(int(key))
        hit, _score, packed, _depth, _bound, _static, _rule50 = tt.tt_probe(work.table, key, 0)
        if not hit or int(packed) == 0:
            break
        try:
            move = chess.Move.from_uci(movegen.move_to_uci(int(packed)))
        except ValueError:
            break
        if move not in walk.legal_moves:
            break
        line.append(move.uci())
        walk.push(move)
    return line


def window_probe(case: dict[str, Any], work: search.Work) -> dict[str, Any]:
    """One negamax call at the case's own depth and window."""
    board = chess.Board(str(case["fen"]))
    search._prepare(board, 3_600_000, 0, 0, work)
    alpha, beta = np.int32(case["alpha"]), np.int32(case["beta"])
    score = int(search.negamax(work, 0, int(case["depth"]), alpha, beta, bool(case["is_pv"])))
    return {
        "score": score,
        "nodes": search.nodes(work),
        "fired": {k: v for k, v in search.mechanism_counts(work).items() if v},
        "pv": principal_variation(board, work),
    }


def root_probe(case: dict[str, Any], depth: int, work: search.Work) -> dict[str, Any]:
    """A whole search to a fixed depth, which is what picks the move in a game."""
    board = chess.Board(str(case["fen"]))
    move = search.think(board, 3_600_000, increment_ms=0, max_depth=depth, work=work)
    reached, _seldepth, score, nodes, _spent = search.last_search(work)
    return {
        "move": move,
        "score": score,
        "depth": reached,
        "nodes": nodes,
        "fired": {k: v for k, v in search.mechanism_counts(work).items() if v},
        "pv": principal_variation(board, work),
    }


def variants() -> list[tuple[str, str | None]]:
    """The runs to make for each case: the engine, the oracle, and one per mechanism."""
    rows: list[tuple[str, str | None]] = [("all on", None), ("oracle", "oracle")]
    rows.extend((f"-{name}", name) for name in search.MECHANISMS)
    return rows


def configure(mechanism: str | None, work: search.Work) -> None:
    if mechanism == "oracle":
        search.set_pruning(False, work)
        search.set_mechanisms(None, work)
    elif mechanism is None:
        search.set_pruning(True, work)
        search.set_mechanisms(None, work)
    else:
        search.set_pruning(True, work)
        search.without_mechanism(mechanism, work)


def contradicts(score: int, oracle: int, alpha: int, beta: int) -> bool:
    """Does this return value claim something the oracle rules out?

    The oracle is run through the same window, so it returns a bound too, not a number to
    be matched. A fail-high says the score is at least beta and a fail-low that it is at
    most alpha, and either can be far from the score and still be right. That is the trap
    in reading a table like this: a return of 210 is a correct fail-high in [0,1] and a
    wrong fail-low in [1210,1211], and only the second is a failure.
    """
    if oracle >= beta:
        return score < beta
    if oracle <= alpha:
        return score > alpha
    return score != oracle


def report_case(case: dict[str, Any], root_depth: int, work: search.Work) -> dict[str, Any]:
    board = chess.Board(str(case["fen"]))
    alpha, beta = int(case["alpha"]), int(case["beta"])
    print(f"\n{case['name']}  {case['fen']}")
    print(f"  depth {case['depth']}  window [{alpha},{beta}]  "
          f"{'PV' if case['is_pv'] else 'non-PV'}  "
          f"in check: {'yes' if board.is_check() else 'no'}  "
          f"reference: {case.get('move', '-')}")
    if case.get("note"):
        print(f"  {case['note']}")

    runs: dict[str, dict[str, Any]] = {}
    for label, mechanism in variants():
        fresh(work)
        configure(mechanism, work)
        runs[label] = window_probe(case, work)
        if root_depth:
            fresh(work)
            configure(mechanism, work)
            runs[label]["root"] = root_probe(case, root_depth, work)
    fresh(work)

    oracle = int(runs["oracle"]["score"])
    baseline = int(runs["all on"]["score"])
    reference = case.get("move")
    header = f"  {'run':<12} {'score':>8} {'bound':>6} {'nodes':>9}"
    if root_depth:
        header += f" {'root move':>10} {'root cp':>8}"
    print(header + "  fired / pv")
    for label, _mechanism in variants():
        row = runs[label]
        score = int(row["score"])
        verdict = "-" if label == "oracle" else (
            "wrong" if contradicts(score, oracle, alpha, beta) else "ok"
        )
        line = f"  {label:<12} {score:>8} {verdict:>6} {row['nodes']:>9,}"
        if root_depth:
            root = row["root"]
            mark = "*" if reference and root["move"] == reference else " "
            line += f" {root['move'] + mark:>10} {root['score']:>8}"
        fired = ",".join(f"{k}={v}" for k, v in row["fired"].items()) or "-"
        pv = " ".join(row["pv"][:6]) or "-"
        print(f"{line}  {fired} | {pv}")

    blamed: list[str] = []
    if contradicts(baseline, oracle, alpha, beta):
        blamed = [
            label for label, mechanism in variants()
            if mechanism not in (None, "oracle")
            and not contradicts(int(runs[label]["score"]), oracle, alpha, beta)
        ]
        if not blamed:
            blamed = ["no single mechanism; the bound needs more than one of them off"]
    print(f"  blamed: {', '.join(blamed) if blamed else 'nothing; the engine agrees'}")
    return {"case": case, "oracle": oracle, "runs": runs, "blamed": blamed}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--positions", type=Path,
                        help="JSON list of cases; the built-in list is used without it")
    parser.add_argument("--root-depth", type=int, default=0,
                        help="also run a whole search to this depth per variant")
    parser.add_argument("--json", type=Path, help="write the raw numbers here")
    arguments = parser.parse_args()

    cases: tuple[dict[str, Any], ...] = CASES
    if arguments.positions:
        loaded = json.loads(arguments.positions.read_text())
        cases = tuple({"depth": 6, "alpha": -50, "beta": -49, "is_pv": False, **case}
                      for case in loaded)

    work = search.WORK
    results = [report_case(case, arguments.root_depth, work) for case in cases]
    if arguments.json:
        arguments.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
