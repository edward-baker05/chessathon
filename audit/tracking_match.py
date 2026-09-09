"""Paired pilot isolating normalized-EP game tracking, with unchanged search/weights.

One Work per player, cleared at each new game; unchanged python-chess adjudication.
This is a node-limited strength pilot, not a platform-clock or process-isolation test.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import position  # noqa: E402
import search  # noqa: E402
import tt  # noqa: E402

OPENINGS = [
    'e4 e5 Nf3 Nc6 Bb5 a6',
    'e4 c5 Nf3 d6 d4 cxd4',
    'd4 d5 c4 e6 Nc3 Nf6',
    'd4 Nf6 c4 g6 Nc3 Bg7',
]


def signature(board: chess.Board, fixed: bool) -> tuple[Any, ...]:
    ep = board.ep_square if not fixed or board.has_legal_en_passant() else None
    return board.board_fen(), board.turn, board.castling_rights, ep


def main() -> None:
    works = [search.WORK, search.new_work(tt.new_table())]
    results = []
    for opening in OPENINGS:
        for fixed_white in (True, False):
            board = chess.Board()
            for san in opening.split():
                board.push_san(san)
            board = chess.Board(board.fen())  # Curated starting positions have no history.
            seen: list[list[int]] = [[], []]
            replies: list[chess.Board | None] = [None, None]
            resets = [0, 0]
            traces = []
            for work in works:
                tt.tt_clear(work.table)
                search.clear_tables(work)
            for _ in range(400):
                if board.outcome(claim_draw=True) is not None:
                    break
                index = int(board.turn != fixed_white)  # Fixed player always index zero.
                work = works[index]
                incoming = chess.Board(board.fen())
                continues = False
                previous = replies[index]
                if previous is not None:
                    target = signature(incoming, index == 0)
                    for move in list(previous.legal_moves):
                        previous.push(move)
                        continues = signature(previous, index == 0) == target
                        previous.pop()
                        if continues:
                            break
                if not continues:
                    resets[index] += int(previous is not None)
                    seen[index] = []
                    tt.tt_clear(work.table)
                    search.clear_tables(work)
                position.encode(incoming, work.state[0], work.mail[0])
                seen[index].append(int(work.state[0, 12]))
                search.set_game_history(seen[index], work)
                started = time.perf_counter()
                uci = search.think(incoming, 3600000, increment_ms=0, node_limit=65536, work=work)
                elapsed = time.perf_counter() - started
                move = chess.Move.from_uci(uci)
                assert move in board.legal_moves
                traces.append({'fen': board.fen(), 'move': uci, 'fixed': index == 0,
                               'depth': int(work.ints[search.I_DEPTH]), 'seconds': elapsed})
                board.push(move)
                replies[index] = board.copy(stack=False)
                position.encode(board, work.state[0], work.mail[0])
                seen[index].append(int(work.state[0, 12]))
            outcome = board.outcome(claim_draw=True)
            result = 0.5 if outcome is None or outcome.winner is None else float(
                outcome.winner == fixed_white)
            results.append({'opening': opening, 'fixed_white': fixed_white, 'fixed_score': result,
                            'termination': str(outcome), 'resets_fixed_baseline': resets,
                            'moves': traces})
            (ROOT / 'audit' / 'tracking-match.json').write_text(json.dumps(results, indent=2)+'\n')
            print(f'{len(results)}/8: fixed score {result}, resets {resets}', flush=True)
    # Verify code paths used the same weights and node budget; no claim of Elo from eight games.
    assert all(np.isfinite(t['seconds']) for r in results for t in r['moves'])


if __name__ == '__main__':
    main()
