"""Generate training positions from our own games, labelled by our own search.

The Lichess file is analysis positions: tactical, critical, and drawn from a distribution
no search of ours ever visits. This produces the other kind, positions our engine actually
reaches, and it is entirely our own work at every stage, which is what the rules require.

The point is not self-play. The point is the gap between how the games are played and how
they are labelled. Games run at a small node budget, because their job is variety and
reachability, and every kept position is then labelled by a much deeper search. A label
that is stronger than the play is what makes the data teach the network something it did
not already know; labelling at the strength it plays at would only write down its own
opinions and train it back towards them.

Records are the same 32 bytes `tools/extract.py` writes, so `tools/shuffle.py` and
`tools/train.py` take the output with no changes, and a self-play file can be concatenated
onto the Lichess one and shuffled together.

    uv run python tools/selfplay.py --games 2000 --out data/selfplay.bin
    cat data/train.bin data/selfplay.bin > data/mixed.bin
    uv run python tools/shuffle.py --data data/mixed.bin --out data/mixed.shuffled.bin

Not shipped: tools/ never reaches the zip.
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from tools.dataset import MAX_PIECES, pack  # noqa: E402
from tools.extract import MATE_SCORE, PIECE_CODE  # noqa: E402

# Long enough that endgames are reached and labelled, short enough that a drawn shuffle of
# pieces does not eat a worker. The competition's own cap is 600 plies.
PLY_CAP = 300
# Plies of random legal play on top of the opening, before the engine takes over. Without
# it a deterministic engine plays the same game from each opening every time, and 21
# openings would produce 21 games however many are asked for.
RANDOM_PLIES = (2, 8)
# Once a game is this far decided for this many labelled plies in a row, the rest of it is
# dropped. Playing on manufactures won endgames: the first smoke run was 32.7% decisive
# against the Lichess file's 11.5%, and those are the positions the network is already
# worst at, where sigmoid(score / 400) has saturated and there is little left to learn.
# Adjudicating is how the mix is controlled at the source rather than by throwing away
# records afterwards.
ADJUDICATE_PLIES = 4

_engine: object = None


def _worker_init(engine_dir: str) -> None:
    """Import the engine once per worker. numba compilation lands here, not per game."""
    global _engine
    sys.path.insert(0, engine_dir)
    import search

    _engine = search


def _encode(board: chess.Board) -> tuple[int, list[int]] | None:
    """Occupancy and piece codes in ascending square order, as `pack` wants them."""
    occupancy = 0
    codes: list[int] = []
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue
        occupancy |= 1 << square
        codes.append(PIECE_CODE[piece.symbol()])
    return None if len(codes) > MAX_PIECES else (occupancy, codes)


def _play(board: chess.Board, rng: random.Random, nodes: int) -> list[tuple[str, chess.Move]]:
    """One game from `board`, returning each position and the move played in it."""
    search = _engine
    assert search is not None
    search.clear_tables()  # type: ignore[attr-defined]
    visited: list[tuple[str, chess.Move]] = []
    for _ in range(PLY_CAP):
        if board.is_game_over(claim_draw=True):
            break
        # A small random share of moves, so the engine's own determinism does not collapse
        # every game from one opening into the same line once the random prefix is over.
        if rng.random() < 0.02:
            move = rng.choice(list(board.legal_moves))
        else:
            uci = search.think(  # type: ignore[attr-defined]
                board, 3_600_000, increment_ms=0, node_limit=nodes
            )
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
        visited.append((board.fen(), move))
        board.push(move)
    return visited


def _label(fen: str, move: chess.Move, depth: int) -> tuple[bytes, int] | None:
    """One position to one record and its score, or None if it is not one to train on.

    The filters are `tools/extract.py`'s, for the same reasons and so that a mixed file is
    filtered consistently: a position in check is not a position the evaluation is asked
    about, and one whose best move wins material is one the quiescence search resolves for
    itself, so training on it teaches the evaluation to count the tactic twice.
    """
    board = chess.Board(fen)
    if board.is_check():
        return None
    if board.is_capture(move) or move.promotion is not None:
        return None
    encoded = _encode(board)
    if encoded is None:
        return None
    occupancy, codes = encoded

    search = _engine
    assert search is not None
    # From the side to move's point of view, which is what negamax returns at the root and
    # what the record stores, so unlike the Lichess file there is no sign convention to get
    # wrong here.
    score = int(search.search_value(board, depth))  # type: ignore[attr-defined]
    score = max(-MATE_SCORE, min(MATE_SCORE, score))
    return pack(occupancy, codes, board.turn == chess.BLACK, score), score


def _run_games(payload: tuple[int, list[str], int, int, int]) -> tuple[bytes, int, int]:
    """One batch of games, played and labelled. Runs in a worker process."""
    seed, openings, nodes, depth, adjudicate = payload
    rng = random.Random(seed)
    out = bytearray()
    visited = 0
    for opening in openings:
        board = chess.Board(opening)
        for _ in range(rng.randint(*RANDOM_PLIES)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over(claim_draw=True):
            continue
        decided = 0
        for fen, move in _play(board, rng, nodes):
            visited += 1
            labelled = _label(fen, move, depth)
            if labelled is None:
                continue
            record, score = labelled
            # Counted over labelled plies rather than played ones, so a run of captures
            # in a won position does not reset it.
            decided = decided + 1 if adjudicate and abs(score) > adjudicate else 0
            if decided > ADJUDICATE_PLIES:
                break
            out += record
    return bytes(out), visited, len(out) // 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT,
                        help="directory holding the engine and the weights to play with")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "selfplay.bin")
    parser.add_argument("--games", type=int, default=1000)
    # Small. These games buy reachable positions, not good ones; the label is where the
    # quality goes, and a node budget here that competes with the label depth is spent on
    # the wrong half of the run.
    parser.add_argument("--nodes", type=int, default=20_000, help="node budget per move in play")
    parser.add_argument("--depth", type=int, default=9, help="fixed depth for the label")
    parser.add_argument("--adjudicate", type=int, default=1000,
                        help="stop a game once it is this decided, in centipawns; 0 to play on")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    from tests.openings import OPENINGS

    rng = random.Random(arguments.seed)
    openings = [rng.choice(OPENINGS) for _ in range(arguments.games)]
    # One batch per worker chunk rather than one per game, so the engine import is paid
    # once per worker instead of being amortised badly across a long queue.
    per_batch = max(1, arguments.games // (arguments.workers * 4))
    batches = [
        (arguments.seed + index, openings[at:at + per_batch], arguments.nodes, arguments.depth,
         arguments.adjudicate)
        for index, at in enumerate(range(0, len(openings), per_batch))
    ]

    print(f"{arguments.games} games from {arguments.engine.name}, {arguments.workers} workers, "
          f"play at {arguments.nodes:,} nodes, label at depth {arguments.depth}, "
          f"adjudicate at {arguments.adjudicate or 'never'}")
    started = time.perf_counter()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    visited = kept = 0
    with (
        arguments.out.open("wb") as sink,
        mp.Pool(
            arguments.workers, initializer=_worker_init, initargs=(str(arguments.engine.resolve()),)
        ) as pool,
    ):
        for done, (packed, seen, wrote) in enumerate(
            pool.imap_unordered(_run_games, batches), start=1
        ):
            sink.write(packed)
            visited += seen
            kept += wrote
            elapsed = time.perf_counter() - started
            print(f"\rbatch {done}/{len(batches)}: {kept:,} positions kept of {visited:,} "
                  f"visited, {kept / max(elapsed, 1e-9):,.0f}/s", end="", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\n{arguments.out} ({arguments.out.stat().st_size:,} bytes, {kept:,} positions) "
          f"in {elapsed:.0f}s")
    arguments.out.with_suffix(".meta.json").write_text(json.dumps({
        "source": "selfplay",
        "engine": str(arguments.engine),
        "games": arguments.games,
        "play_nodes": arguments.nodes,
        "label_depth": arguments.depth,
        "adjudicate_cp": arguments.adjudicate,
        "positions": kept,
        "mate_score": MATE_SCORE,
    }, indent=2) + "\n")
    print("shuffle before training: "
          f"uv run python tools/shuffle.py --data {arguments.out} --out <shuffled>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
