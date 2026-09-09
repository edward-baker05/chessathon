"""Fresh diagnostic games against a fixed-node offline teacher; not an Elo calibration."""

import json
import sys
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
from uci import Engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402
import search  # noqa: E402


def main() -> None:
    sf = Engine(sys.argv[1])
    sf.configure({'Threads': 1, 'Hash': 64})
    results: list[dict[str, Any]] = []
    agent.INCREMENT_MS = 200
    for ours in (chess.WHITE, chess.BLACK):
        board = chess.Board()
        for san in ['d4', 'Nf6', 'c4', 'e6', 'Nf3', 'd5', 'g3', 'Be7', 'Bg2', 'O-O']:
            board.push_san(san)
        board = chess.Board(board.fen())
        agent._last_reply = None
        sf.configure({'Clear Hash': None})
        clock = 30000.0
        record: dict[str, Any] = {'ours_white': ours, 'start': board.fen(), 'moves': []}
        for _ in range(300):
            if board.outcome(claim_draw=True) is not None:
                break
            fen = board.fen()
            if board.turn == ours:
                before = time.perf_counter()
                uci = agent.get_move(fen, int(clock))
                elapsed = (time.perf_counter() - before) * 1000
                clock -= elapsed
                if clock < 0:
                    record['flag'] = True
                    break
                clock += 200
                depth, seldepth, value, nodes, _ = search.last_search()
                sf.configure({'Clear Hash': None})
                best = sf.analyse(board, chess.engine.Limit(nodes=300000))
                sf.configure({'Clear Hash': None})
                chosen = sf.analyse(board, chess.engine.Limit(nodes=300000),
                                    root_moves=[chess.Move.from_uci(uci)])
                best_score = best['score'].pov(ours).score(mate_score=10000)
                choice_score = chosen['score'].pov(ours).score(mate_score=10000)
                record['moves'].append({'fen': fen, 'move': uci, 'ours': True,
                                        'depth': depth, 'seldepth': seldepth, 'score': value,
                                        'nodes': nodes, 'ms': elapsed, 'clock_ms': clock,
                                        'reference_move': best['pv'][0].uci(),
                                        'reference_score': best_score,
                                        'choice_score': choice_score,
                                        'loss_cp': max(0, best_score - choice_score)})
            else:
                info = sf.analyse(board, chess.engine.Limit(nodes=100000))
                uci = info['pv'][0].uci()
                record['moves'].append({'fen': fen, 'move': uci, 'ours': False})
            board.push_uci(uci)
            (ROOT / 'audit' / 'reference-games.json').write_text(
                json.dumps([*results, record], indent=2)+'\n')
        record['outcome'] = str(board.outcome(claim_draw=True))
        results.append(record)
        (ROOT / 'audit' / 'reference-games.json').write_text(json.dumps(results, indent=2)+'\n')
        errors = sorted((r for r in record['moves'] if r['ours']),
                        key=lambda r: r['loss_cp'], reverse=True)
        print(f"ours_white={ours} {record['outcome']}; losses {[r['loss_cp'] for r in errors[:8]]}",
              flush=True)
    sf.quit()


if __name__ == '__main__':
    main()
