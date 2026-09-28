"""What the selective mechanisms cost, on positions from real games.

`tools/selectivity.py` explains one position. This measures how often the explanation
matters, which is the question a composed fixture cannot answer. Two mating fixtures showed
that razoring, futility and late move pruning each remove a quiet move that gives check;
neither showed how often a real game contains one.

For each sampled position it runs the shipped search and then the same search with all
eight selective mechanisms off, at the same fixed depth. Everything off still keeps the
check extension and the transposition policy, so the floor is the real search minus
selectivity rather than plain alpha-beta.

The floor is a reference, not a truth. It is the same depth and the same evaluation, so
where it scores a position higher than the shipped search does, selectivity gave that up
at this depth; a deeper search might give it back, and a better evaluation might say the
line was never worth anything. What the floor cannot be is systematically biased in favour
of the shipped search, which is what makes the rate worth reading.

    uv run python tools/failures.py --pgn game.pgn --depth 6 --stride 8 --limit 200

Reports the rate at which the floor beats the shipped search by more than a margin, and
what kind of move the floor preferred: quiet or not, giving check or not. The last of those
is the hypothesis under test, since the quiet prunings decide before the move is made and
therefore before its check status is known.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

import search  # noqa: E402
import tt  # noqa: E402


def fresh(work: search.Work) -> None:
    tt.tt_clear(work.table)
    search.clear_tables(work)
    search.set_game_history([], work)
    search.set_pruning(True, work)


def searched(board: chess.Board, depth: int, mechanisms: tuple[str, ...] | None,
             work: search.Work) -> tuple[str, int, int]:
    """Best move, score and node count at a fixed depth, with these mechanisms enabled."""
    fresh(work)
    search.set_mechanisms(mechanisms, work)
    move = search.think(board, 3_600_000, increment_ms=0, max_depth=depth, work=work)
    _reached, _seldepth, score, nodes, _spent = search.last_search(work)
    return move, score, nodes


def positions(paths: list[Path], stride: int, limit: int) -> list[str]:
    """Every stride-th position from these games, in check positions left out.

    A node where the side to move is in check has the quiet prunings switched off already,
    so it cannot show what they cost.
    """
    found: list[str] = []
    for path in paths:
        with path.open() as handle:
            while len(found) < limit:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                board = game.board()
                for index, move in enumerate(game.mainline_moves()):
                    if (index % stride == 0 and not board.is_check()
                            and not board.is_game_over()):
                        found.append(board.fen())
                    board.push(move)
        if len(found) >= limit:
            break
    return found[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pgn", type=Path, nargs="+", required=True)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--stride", type=int, default=8,
                        help="sample one position in this many, so one game is not the run")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--margin", type=int, default=50,
                        help="centipawns the floor must beat the shipped search by")
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()

    work = search.WORK
    sample = positions(arguments.pgn, arguments.stride, arguments.limit)
    print(f"{len(sample)} positions at depth {arguments.depth}, "
          f"floor beats shipped by more than {arguments.margin}cp counts as a loss\n")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, fen in enumerate(sample):
        board = chess.Board(fen)
        shipped_move, shipped_score, shipped_nodes = searched(board, arguments.depth, None, work)
        floor_move, floor_score, floor_nodes = searched(board, arguments.depth, (), work)
        move = chess.Move.from_uci(floor_move)
        row = {
            "fen": fen,
            "shipped": shipped_move, "shipped_cp": shipped_score, "shipped_nodes": shipped_nodes,
            "floor": floor_move, "floor_cp": floor_score, "floor_nodes": floor_nodes,
            "gap": floor_score - shipped_score,
            "changed": shipped_move != floor_move,
            "quiet": not board.is_capture(move) and move.promotion is None,
            "gives_check": board.gives_check(move),
        }
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(sample)}  {time.perf_counter() - started:.0f}s")

    # Equal nominal depth is not equal work: the floor searches tens of times the nodes to
    # reach the same depth. So every position the floor won at equal depth is re-run with
    # the shipped search given the floor's own node budget, which is the comparison an
    # actual game makes. A gap that survives that is a gap selectivity really costs.
    lost = [row for row in rows if row["gap"] > arguments.margin]
    for hit in lost:
        fresh(work)
        search.set_mechanisms(None, work)
        rematch = chess.Board(str(hit["fen"]))
        search.think(rematch, 3_600_000, increment_ms=0, node_limit=int(str(hit["floor_nodes"])),
                     max_depth=search.MAX_DEPTH, work=work)
        reached, _seldepth, score, _nodes, _spent = search.last_search(work)
        hit["equal_nodes_cp"] = score
        hit["equal_nodes_depth"] = reached
        hit["survives"] = int(str(hit["floor_cp"])) - score > arguments.margin
    changed = [row for row in rows if row["changed"]]
    quiet_checks = [row for row in lost if row["quiet"] and row["gives_check"]]
    print(f"\npositions                          {len(rows)}")
    print(f"floor picked a different move      {len(changed)} ({len(changed)/len(rows):.1%})")
    print(f"floor scored more than {arguments.margin}cp higher  {len(lost)} "
          f"({len(lost)/len(rows):.1%})")
    if lost:
        print(f"  of those, the floor's move was quiet          "
              f"{sum(1 for r in lost if r['quiet'])}")
        print(f"  of those, it gave check                       "
              f"{sum(1 for r in lost if r['gives_check'])}")
        print(f"  of those, it was a quiet move giving check    {len(quiet_checks)} "
              f"({len(quiet_checks)/len(rows):.2%} of all positions)")
        worst = sorted(lost, key=lambda r: -int(r["gap"]))[:6]
        print("\n  largest gaps:")
        for row in worst:
            tag = ("quiet check" if row["quiet"] and row["gives_check"]
                   else "quiet" if row["quiet"] else "capture")
            gap, shipped_cp, floor_cp = row["gap"], row["shipped_cp"], row["floor_cp"]
            print(f"    {gap:>6}cp  shipped {row['shipped']} {shipped_cp:>6}"
                  f"  floor {row['floor']} {floor_cp:>6}  {tag:<12} {row['fen']}")
    if lost:
        survivors = [row for row in lost if row["survives"]]
        print(f"\n  given the floor's own node budget the shipped search closes "
              f"{len(lost) - len(survivors)} of those {len(lost)} gaps;"
              f" {len(survivors)} survive ({len(survivors)/len(rows):.2%} of all positions)")
        for row in survivors:
            tag = ("quiet check" if row["quiet"] and row["gives_check"]
                   else "quiet" if row["quiet"] else "capture")
            print(f"    floor {row['floor_cp']:>6} at depth {arguments.depth}"
                  f"   shipped {row['equal_nodes_cp']:>6} at depth "
                  f"{row['equal_nodes_depth']:<3} {tag:<12} {row['fen']}")
    shipped_total = sum(int(r["shipped_nodes"]) for r in rows)
    floor_total = sum(int(r["floor_nodes"]) for r in rows)
    print(f"\nnodes: shipped {shipped_total:,}, floor {floor_total:,} "
          f"({floor_total / max(shipped_total, 1):.0f}x)")
    if arguments.json:
        arguments.json.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
