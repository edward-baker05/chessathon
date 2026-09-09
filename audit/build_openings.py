"""Build the opening set the A/B match plays from.

Both engines are deterministic at a fixed node budget, so the same opening replays the
same two games and adds nothing at all. An A/B over N pairs therefore needs N distinct
openings, and `audit/positions.json` only has 43 of them.

Three sources, deduplicated by position rather than by FEN, so the halfmove clock and the
move number cannot smuggle the same position in twice:

* the curated start position in each rated game's PGN header, which is the real thing the
  platform starts its games from,
* six and twelve plies into each of those games, for positions of a shape the engine
  actually reaches,
* the audit suite, generated separately from mainstream openings and engine play.

Positions already decided by material, or already down to an endgame, are dropped: a game
that starts won measures nothing about either build.

    uv run python audit/build_openings.py
"""

import json
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "audit" / "ab-openings.json"

# Enough material left that the opening is still an opening, and close enough that neither
# side starts winning.
MIN_PIECES = 16
MAX_MATERIAL_EDGE = 2
VALUES = ((chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5),
          (chess.QUEEN, 9))


def material(board: chess.Board, colour: chess.Color) -> int:
    return sum(len(board.pieces(piece, colour)) * value for piece, value in VALUES)


def consider(board: chess.Board, cluster: str, seen: dict[str, tuple[str, str]]) -> None:
    if board.is_game_over() or not board.is_valid():
        return
    if chess.popcount(board.occupied) - 2 < MIN_PIECES:
        return
    if abs(material(board, chess.WHITE) - material(board, chess.BLACK)) > MAX_MATERIAL_EDGE:
        return
    # Keyed on the position and the side to move, not the FEN: two games can reach one
    # position with different clocks and they are still one opening.
    seen.setdefault(f"{board.board_fen()} {board.turn}", (board.fen(), cluster))


def main() -> int:
    seen: dict[str, tuple[str, str]] = {}
    for path in sorted((ROOT / "logs").glob("*.pgn")):
        with path.open() as handle:
            game = chess.pgn.read_game(handle)
        if game is None:
            continue
        # One cluster per source game. The header position and the positions six and twelve
        # plies into it are the same game seen three times, not three independent draws,
        # and a match that treats them as independent reports an interval that is too
        # narrow for the evidence it has.
        cluster = path.stem
        board = game.board()
        consider(board.copy(), cluster, seen)
        for ply, node in enumerate(game.mainline(), 1):
            board.push(node.move)
            if ply in (6, 12):
                consider(board.copy(), cluster, seen)

    positions = json.loads((ROOT / "audit" / "positions.json").read_text())["positions"]
    for row in positions:
        # The audit suite grew 43 positions out of eight opening lines, so its own ids
        # carry the cluster: "g3-p14" is the fourth game line.
        consider(chess.Board(row["fen"]), "audit-" + row["id"].split("-")[0], seen)

    rows = sorted(seen.values())
    OUT.write_text(json.dumps(
        {"positions": [{"fen": fen, "cluster": cluster} for fen, cluster in rows]}, indent=1
    ) + "\n")
    clusters = {cluster for _, cluster in rows}
    print(f"{len(rows)} openings in {len(clusters)} independent clusters -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
