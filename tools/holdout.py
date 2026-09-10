"""A small evaluation holdout the training data has never seen.

The trainer's own holdout is the tail of the same file, drawn from the same distribution
and split at a byte offset. It answers "did this net fit this file" and it cannot answer
"is this evaluation any good", which is the question a strength change turns on. This
builds the other kind: fresh games played here, positions filtered the way the training
extractor filters them, and labels asked for now rather than read out of the same dump.

Three things make it usable as a holdout rather than as more training data.

**Split by game, before anything is trained.** Every position carries the cluster of the
opening it came from. Positions from one game are one game seen several times, and a split
that cuts across games leaks.

**Filtered the way training is filtered.** `tools/extract.py` drops positions in check and
positions whose best move captures or promotes, because quiescence resolves those and a
static evaluation trained on them learns to double count tactics. The same two filters run
here, so the holdout measures the function the net was actually asked to learn.

**Labelled with the ranking, not only the score.** Every position stores the reference's
top few moves with their scores, so a net can be scored on whether it orders moves the way
the reference does and not only on how close its number is. A net can halve its centipawn
error and order candidate moves worse.

    uv run python tools/holdout.py --engine /path/to/stockfish --games 120 \
        --out audit/next-strength/run1/holdout.json

Positions where the reference's top two moves are far apart are labelled unstable rather
than dropped. A deep search score is not automatically a sound static target, and a mean
taken over positions that are really tactics says more about the tactics than the net.
"""

import argparse
import json
import random
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

# Matches tools/extract.py, so a label from here means the same thing as a label from the
# training dump. A mate is saturated rather than dropped, for the same reason: at this
# scale sigmoid(12800/400) is 1 to well beyond float precision, so the exact value only has
# to be past saturation.
MATE_SCORE = 12800
# Two moves this far apart make the position a tactic rather than a quiet evaluation.
UNSTABLE_CP = 200


def phase_of(board: chess.Board) -> str:
    if board.ply() < 24:
        return "opening"
    return "endgame" if len(board.piece_map()) <= 12 else "middlegame"


def texture(board: chess.Board) -> list[str]:
    """Coarse tags for the position types the holdout is supposed to contain.

    Reported per tag, so a net that is fine on average and hopeless in rook endings does
    not get to hide behind the average.
    """
    tags: list[str] = []
    pieces = board.piece_map()
    kinds = [piece.piece_type for piece in pieces.values()]
    heavy = sum(1 for k in kinds if k in (chess.QUEEN, chess.ROOK))
    minors = sum(1 for k in kinds if k in (chess.BISHOP, chess.KNIGHT))
    pawns = sum(1 for k in kinds if k == chess.PAWN)
    if heavy == 2 and all(k in (chess.ROOK, chess.PAWN, chess.KING) for k in kinds):
        tags.append("rook_ending")
    if minors == 0 and heavy == 0:
        tags.append("pawn_race")
    balance = sum(
        (1 if piece.color else -1) * {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}[piece.piece_type]
        for piece in pieces.values()
    )
    if abs(balance) >= 2:
        tags.append("imbalance")
    for colour in (chess.WHITE, chess.BLACK):
        king = board.king(colour)
        if king is None:
            continue
        rank = chess.square_rank(king) if colour else 7 - chess.square_rank(king)
        if rank >= 3 and pawns >= 8:
            tags.append("king_exposed")
        if rank >= 3 and pawns < 8:
            tags.append("king_active")
    return tags or ["plain"]


def quiet_leaf(board: chess.Board, best: chess.Move) -> bool:
    """The two filters tools/extract.py applies, so the holdout is the same kind of thing."""
    if board.is_check():
        return False
    return not board.is_capture(best) and best.promotion is None


def play(engine: Engine, opening: dict[str, Any], plies: int, nodes: int,
         rng: random.Random, spread: int) -> list[dict[str, Any]]:
    """One game from one opening, keeping every position it passed through.

    Moves are chosen from the reference's top few rather than always its best, with the
    spread widening as the game goes on. A game of best moves is one game; a game with a
    little noise in it reaches material imbalances and endings that a drawn main line
    never does, which is most of what the holdout is supposed to contain.
    """
    board = chess.Board(str(opening["fen"]))
    seen: list[dict[str, Any]] = []
    engine.new_game()
    for ply in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        lines = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=3)
        top = lines[0]["score"].pov(board.turn).score(mate_score=MATE_SCORE)
        # The window widens as the game goes on. Kept narrow at the start, the opening
        # stays sane; widened later, the games reach the endings and the material
        # imbalances that a well played draw never visits, which is most of what a
        # holdout for an evaluation is supposed to contain.
        window = spread // 2 if ply < 8 else spread + spread * ply // 60
        choices = [line for line in lines
                   if line["score"].pov(board.turn).score(mate_score=MATE_SCORE) >= top - window
                   and line.get("pv")]
        if not choices:
            break
        seen.append({"fen": board.fen(), "ply": board.ply()})
        board.push(rng.choice(choices)["pv"][0])
    return seen


def label(engine: Engine, fen: str, nodes: int, multipv: int) -> dict[str, Any] | None:
    """Score and move ordering for one position, in the training label's own convention."""
    board = chess.Board(fen)
    engine.new_game()
    lines = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=multipv)
    lines = [line for line in lines if line.get("pv")]
    if not lines:
        return None
    best_move = lines[0]["pv"][0]
    if not quiet_leaf(board, best_move):
        return None
    ranked = [
        {"move": line["pv"][0].uci(),
         "cp": line["score"].pov(board.turn).score(mate_score=MATE_SCORE),
         "mate": line["score"].pov(board.turn).mate()}
        for line in lines
    ]
    top = int(ranked[0]["cp"])
    if abs(top) > MATE_SCORE:
        return None
    second = int(ranked[1]["cp"]) if len(ranked) > 1 else top
    return {
        "fen": fen,
        "cp": top,
        "mate": ranked[0]["mate"],
        "ranked": ranked,
        "unstable": bool(abs(top - second) >= UNSTABLE_CP),
        "phase": phase_of(board),
        "tags": texture(board),
        "men": len(board.piece_map()),
    }


class Pool:
    def __init__(self, path: str, hash_mb: int) -> None:
        self.path, self.hash_mb = path, hash_mb
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--engine", required=True)
    parser.add_argument("--openings", type=Path, default=ROOT / "audit" / "ab-openings.json")
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--plies", type=int, default=90)
    parser.add_argument("--play-nodes", type=int, default=40_000)
    parser.add_argument("--label-nodes", type=int, default=300_000)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--spread", type=int, default=60)
    parser.add_argument("--per-game", type=int, default=14)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.out.exists():
        parser.error(f"{arguments.out} exists; evidence is never overwritten, pick a new name")
    openings = json.loads(arguments.openings.read_text())["positions"]
    rng = random.Random(arguments.seed)
    rng.shuffle(openings)
    chosen = openings[: arguments.games]
    pool = Pool(arguments.engine, arguments.hash)
    started = time.perf_counter()
    print(f"{len(chosen)} games from {len({o['cluster'] for o in chosen})} clusters, "
          f"{arguments.workers} workers")

    def one_game(job: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
        index, opening = job
        engine = pool.engine()
        local = random.Random(arguments.seed + index)
        seen = play(engine, opening, arguments.plies, arguments.play_nodes, local,
                    arguments.spread)
        # Spread the sample over the game rather than taking its opening.
        local.shuffle(seen)
        rows: list[dict[str, Any]] = []
        for row in seen:
            if len(rows) >= arguments.per_game:
                break
            labelled = label(engine, str(row["fen"]), arguments.label_nodes, arguments.multipv)
            if labelled is None:
                continue
            labelled["cluster"] = opening["cluster"]
            labelled["game"] = f"{opening['cluster']}#{index}"
            rows.append(labelled)
        return rows

    done = 0
    lock = threading.Lock()

    def wrapped(job: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal done
        rows = one_game(job)
        with lock:
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(chosen)} games  "
                      f"{time.perf_counter() - started:.0f}s", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=arguments.workers) as runner:
        batches = list(runner.map(wrapped, enumerate(chosen)))
    pool.close()

    positions: list[dict[str, Any]] = []
    seen_fen: set[str] = set()
    for batch in batches:
        for row in batch:
            key = " ".join(str(row["fen"]).split()[:4])
            if key in seen_fen:
                continue
            seen_fen.add(key)
            positions.append(row)

    tags: dict[str, int] = {}
    phases: dict[str, int] = {}
    for row in positions:
        phases[str(row["phase"])] = phases.get(str(row["phase"]), 0) + 1
        for tag in row["tags"]:
            tags[tag] = tags.get(tag, 0) + 1
    print(f"\n{len(positions)} positions, {len({r['game'] for r in positions})} games, "
          f"{len({r['cluster'] for r in positions})} clusters")
    print(f"unstable  {sum(1 for r in positions if r['unstable'])}")
    print(f"phases    {phases}")
    print(f"tags      {tags}")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "provenance": {
            "engine": arguments.engine, "openings": str(arguments.openings),
            "games": arguments.games, "plies": arguments.plies,
            "play_nodes": arguments.play_nodes, "label_nodes": arguments.label_nodes,
            "multipv": arguments.multipv, "spread": arguments.spread,
            "per_game": arguments.per_game, "seed": arguments.seed,
            "mate_score": MATE_SCORE, "unstable_cp": UNSTABLE_CP,
            "seconds": round(time.perf_counter() - started, 1),
        },
        "positions": positions,
    }, indent=1) + "\n")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
