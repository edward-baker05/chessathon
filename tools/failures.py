"""Candidate moves from real games, recorded well enough for something else to judge.

The previous version of this file compared two searches' own root scores and called the
difference a loss. It is not one. Two searches can pick the same move and disagree about
what it is worth, and the more optimistic of them can be the less accurate; a positive
`floor - shipped` therefore names a disagreement and not an error, and looking only at
positive differences hides the case that costs games, which is the shipped search being
wrong in its own favour. Nothing here reports a rate of mistakes any more. It produces the
material `tools/judge.py` needs to have an outside opinion about who was right.

What it records, per position and per search: the move actually selected, the depth that
completed, nodes, elapsed seconds, the score and whether it was a bound, which mechanisms
fired, whether the search aborted, and the identity of the source and of the network. The
equal-budget re-search keeps its move, which the old one discarded.

Sampling is by game and by phase rather than by file prefix. Positions in check are kept
as their own stratum instead of being dropped: the quiet prunings are already off in
check, so those positions cannot show what selectivity costs, but they can and do show
whether check evasions are searched deeply enough, which is a separate question the old
sampler could not ask at all. Every sample carries the real history that produced it, so
repetition and the fifty-move clock mean in the probe what they meant in the game.

    uv run python tools/failures.py --pgn 'audit/*.pgn' --out audit/next-strength/run1/probe.json

Two cache regimes, never mixed in one run. `--cache cold` clears the table per position and
is a diagnostic. `--cache warm` first re-searches the same side's two preceding moves so
the table holds what a live game would have put there, which is the regime a game actually
plays in and the slower of the two.
"""

import argparse
import glob
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

import nnue  # noqa: E402
import position  # noqa: E402
import search  # noqa: E402
import tt  # noqa: E402
from bitboard import KEY  # noqa: E402

# Mechanism sets. `None` is the shipped engine, `()` is the additive floor: every selective
# mechanism off, but the check extension and the transposition policy still running, which
# is what `set_pruning(False)` throws away and why that switch is not used here.
SHIPPED: tuple[str, ...] | None = None
FLOOR: tuple[str, ...] = ()

_scratch_state, _scratch_mail = position.new_stacks()


def key_of(board: chess.Board) -> int:
    position.encode(board, _scratch_state[0], _scratch_mail[0])
    return int(_scratch_state[0][KEY])


def history_keys(board: chess.Board) -> list[int]:
    """Zobrist keys for every position of this game up to and including `board`.

    `agent.py` builds exactly this list and hands it to `search.set_game_history`, so a
    probe that builds it the same way sees the repetitions the game saw. Rebuilding from
    the root is cheap and keeps the two constructions independent of each other.
    """
    walk = board.root()
    keys = [key_of(walk)]
    for move in board.move_stack:
        walk.push(move)
        keys.append(key_of(walk))
    return keys


def phase_of(board: chess.Board) -> str:
    """Opening, middlegame or endgame, by ply and by material left on the board."""
    if board.ply() < 24:
        return "opening"
    men = len(board.piece_map())
    return "endgame" if men <= 12 else "middlegame"


def board_of(sample: dict[str, Any]) -> chess.Board:
    """A board with its real stack, rebuilt from a stored sample."""
    board = chess.Board(str(sample["root_fen"]))
    for uci in str(sample["history"]).split():
        board.push(chess.Move.from_uci(uci))
    return board


def cluster_of(game: chess.pgn.Game, path: Path, index: int) -> str:
    """What counts as one source for resampling: a game, not a position.

    Two positions from one game are not two independent draws, and neither are two games
    from one opening. The opening tag is preferred where a match wrote one, because that
    is the level the openings were built at.
    """
    for tag in ("Opening", "Event"):
        value = game.headers.get(tag, "")
        if value and value != "?":
            return f"{path.stem}:{value}"
    return f"{path.stem}:{index}"


def sample_games(
    paths: list[Path], per_game: int, seed: int, limit: int
) -> list[dict[str, Any]]:
    """Positions spread over games and phases, deduplicated, with their histories.

    Every game contributes at most `per_game`, drawn to spread over the phases it actually
    contains rather than from its opening. Positions in check are a stratum of their own.
    Duplicates are removed on the position proper, ignoring the move counters, so the same
    opening replayed in twenty games contributes its early positions once.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    chosen: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            index = 0
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                cluster = cluster_of(game, path, index)
                index += 1
                board = game.board()
                root_fen = board.fen()
                pool: list[dict[str, Any]] = []
                moves: list[str] = []
                for move in game.mainline_moves():
                    if not board.is_game_over(claim_draw=True) and board.legal_moves:
                        epd = board.epd()
                        if epd not in seen:
                            pool.append({
                                "cluster": cluster,
                                "source": path.name,
                                "root_fen": root_fen,
                                "history": " ".join(moves),
                                "fen": board.fen(),
                                "epd": epd,
                                "ply": board.ply(),
                                "phase": phase_of(board),
                                "stratum": "check" if board.is_check() else "quiet",
                            })
                    if move not in board.legal_moves:
                        break
                    moves.append(move.uci())
                    board.push(move)
                # Spread inside the game: take round-robin across the phases present, so a
                # long game does not contribute five middlegame positions and no ending.
                by_phase: dict[str, list[dict[str, Any]]] = {}
                for row in pool:
                    by_phase.setdefault(str(row["phase"]), []).append(row)
                for rows in by_phase.values():
                    rng.shuffle(rows)
                taken = 0
                while taken < per_game and any(by_phase.values()):
                    for key in sorted(by_phase):
                        if by_phase[key] and taken < per_game:
                            row = by_phase[key].pop()
                            if row["epd"] in seen:
                                continue
                            seen.add(str(row["epd"]))
                            row["id"] = f"{len(chosen):04d}"
                            chosen.append(row)
                            taken += 1
    rng.shuffle(chosen)
    return chosen[:limit] if limit else chosen


def fens_as_samples(path: Path) -> list[dict[str, Any]]:
    """Bare positions in the same shape a sampled one has, with an empty history.

    Kept deliberately separate from the sampled set and labelled `fixture`, because that is
    what it is: a position with no game behind it. Both draw rules read the history, so a
    fixture cannot show what either of them does, and a rate measured over fixtures is a
    rate over whatever was put in the file.
    """
    loaded = json.loads(path.read_text())
    rows = loaded["positions"] if isinstance(loaded, dict) else loaded
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        fen = str(row if isinstance(row, str) else row["fen"])
        board = chess.Board(fen)
        samples.append({
            "id": f"fixture-{index:03d}",
            "cluster": str(row.get("cluster", f"fixture-{index:03d}"))
            if isinstance(row, dict) else f"fixture-{index:03d}",
            "source": path.name, "root_fen": fen, "history": "", "fen": fen,
            "epd": board.epd(), "ply": board.ply(), "phase": phase_of(board),
            "stratum": "fixture",
        })
    return samples


def fresh(work: search.Work, keys: list[int]) -> None:
    """An empty engine that still knows the game it is in."""
    tt.tt_clear(work.table)
    search.clear_tables(work)
    search.set_pruning(True, work)
    search.set_mechanisms(None, work)
    search.set_game_history(keys, work)


def probe(
    board: chess.Board,
    keys: list[int],
    mechanisms: tuple[str, ...] | None,
    work: search.Work,
    depth: int = search.MAX_DEPTH,
    node_limit: int = 0,
) -> dict[str, Any]:
    """One search, and everything about it worth writing down."""
    search.set_mechanisms(mechanisms, work)
    started = time.perf_counter()
    move = search.think(board, 3_600_000, increment_ms=0, node_limit=node_limit,
                        max_depth=depth, work=work)
    reached, seldepth, score, nodes, _spent = search.last_search(work)
    return {
        "move": move,
        "score": score,
        "depth": reached,
        "seldepth": seldepth,
        "nodes": nodes,
        "seconds": round(time.perf_counter() - started, 4),
        "aborted": bool(work.ints[search.I_ABORT]),
        "fired": {name: count for name, count in search.mechanism_counts(work).items() if count},
    }


def warm(board: chess.Board, keys: list[int], depth: int, work: search.Work,
         moves_back: int = 2) -> int:
    """Re-search the same side's last few moves, so the table holds what a game left there.

    A cold table is not the regime a game plays in: by the time this position arrives the
    engine has already searched its two previous moves and kept the table. Replaying only
    the same side's recent moves costs a bounded amount and reproduces the part of the
    table that matters, which is the subtree this position sits inside.
    """
    stack = board.move_stack
    warmed = 0
    for back in range(2 * moves_back, 0, -2):
        if back > len(stack):
            continue
        earlier = board.root()
        for move in stack[:len(stack) - back]:
            earlier.push(move)
        if earlier.is_game_over(claim_draw=True) or not earlier.legal_moves:
            continue
        search.set_mechanisms(SHIPPED, work)
        search.set_game_history(keys[:len(keys) - back], work)
        search.think(earlier, 3_600_000, increment_ms=0, max_depth=depth, work=work)
        warmed += 1
    search.set_game_history(keys, work)
    return warmed


def run(sample: dict[str, Any], depth: int, cache: str, work: search.Work,
        budget: int = 0) -> dict[str, Any]:
    """Shipped, floor, and shipped again at the floor's own budget."""
    board = board_of(sample)
    keys = history_keys(board)
    row: dict[str, Any] = dict(sample)

    fresh(work, keys)
    if cache == "warm":
        row["warmed"] = warm(board, keys, depth, work)
    row["shipped"] = probe(board, keys, SHIPPED, work, depth=depth)

    fresh(work, keys)
    if cache == "warm":
        warm(board, keys, depth, work)
    row["floor"] = probe(board, keys, FLOOR, work, depth=depth)

    # Equal nominal depth is not equal work and it is not equal time either. This is the
    # comparison a game makes: the shipped search, no depth cap, given the number of nodes
    # the floor spent. Its move is kept, which the previous version threw away, so the
    # judge can be asked about it too.
    fresh(work, keys)
    if cache == "warm":
        warm(board, keys, depth, work)
    row["equal_nodes"] = probe(board, keys, SHIPPED, work, depth=search.MAX_DEPTH,
                               node_limit=int(row["floor"]["nodes"]))

    # A fixed node budget, the same one for every position and every build compared. The
    # equal-node arm above uses whatever the floor happened to spend, which is fine for
    # asking what selectivity costs and useless for comparing two builds against each
    # other, because the budget then moves with the position.
    if budget:
        fresh(work, keys)
        if cache == "warm":
            warm(board, keys, depth, work)
        row["budget"] = probe(board, keys, SHIPPED, work, depth=search.MAX_DEPTH,
                              node_limit=budget)

    candidates: list[str] = []
    for name in ("shipped", "floor", "equal_nodes", *(("budget",) if budget else ())):
        move = str(row[name]["move"])
        if move not in candidates:
            candidates.append(move)
    row["candidates"] = candidates
    row["differs"] = len(candidates) > 1
    # A diagnostic, deliberately not called a loss. It is the difference between two
    # implementations' opinions of the same position, in both directions.
    row["score_gap"] = int(row["floor"]["score"]) - int(row["shipped"]["score"])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pgn", nargs="+", help="paths or globs")
    parser.add_argument("--fens", type=Path,
                        help="a JSON list of FENs, or of objects with a `fen` field, probed "
                             "instead of sampling games. A bare FEN carries no history, so "
                             "repetition and the fifty-move clock mean less in the probe "
                             "than they did in the game the position came from")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--per-game", type=int, default=4)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--cache", choices=("cold", "warm"), default="cold")
    parser.add_argument("--budget", type=int, default=0,
                        help="also search every position to this fixed node count, which is "
                             "the arm two builds can be compared on")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.out.exists():
        parser.error(f"{arguments.out} exists; evidence is never overwritten, pick a new name")

    if not arguments.pgn and not arguments.fens:
        parser.error("one of --pgn or --fens is required")
    paths: list[Path] = []
    if arguments.fens:
        samples = fens_as_samples(arguments.fens)
    else:
        paths = sorted({Path(hit) for pattern in arguments.pgn for hit in glob.glob(pattern)})
        if not paths:
            parser.error(f"no files matched {arguments.pgn}")
        samples = sample_games(paths, arguments.per_game, arguments.seed, arguments.limit)
    net = hashlib.sha256(nnue.WEIGHTS_PATH.read_bytes()).hexdigest()[:12]
    print(f"{len(samples)} positions from {len(paths)} files, depth {arguments.depth}, "
          f"{arguments.cache} cache, net {net}")

    work = search.WORK
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(samples):
        rows.append(run(sample, arguments.depth, arguments.cache, work, arguments.budget))
        if (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(samples)}  {time.perf_counter() - started:.0f}s",
                  flush=True)

    strata: dict[str, int] = {}
    for row in rows:
        strata[str(row["stratum"])] = strata.get(str(row["stratum"]), 0) + 1
    differing = [row for row in rows if row["differs"]]
    print(f"\npositions                     {len(rows)}  {strata}")
    print(f"clusters                      {len({r['cluster'] for r in rows})}")
    print(f"the three searches differ     {len(differing)}")
    print(f"floor scored higher by >50cp  {sum(1 for r in rows if r['score_gap'] > 50)}")
    print(f"floor scored lower by >50cp   {sum(1 for r in rows if r['score_gap'] < -50)}")
    print("These are disagreements between two implementations. Which one is right is not a "
          "question this file can answer; run tools/judge.py on the output.")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "provenance": {
            "depth": arguments.depth, "cache": arguments.cache, "seed": arguments.seed,
            "budget": arguments.budget,
            "per_game": arguments.per_game, "limit": arguments.limit,
            "files": [str(p) for p in paths] or [str(arguments.fens)],
            "net_sha256_12": net,
            "l1": nnue.L1, "buckets": nnue.BUCKETS,
            "seconds": round(time.perf_counter() - started, 1),
        },
        "positions": rows,
    }, indent=1) + "\n")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
