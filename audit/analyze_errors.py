"""Search-scaling probe on early errors mined from the new reference games."""

import json
import sys
from pathlib import Path

import chess
import chess.engine
from uci import Engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import search  # noqa: E402
import tt  # noqa: E402


def main() -> None:
    games = json.loads((ROOT / 'audit' / 'reference-games.json').read_text())
    positions = [m for g in games for m in g['moves']
                 if m['ours'] and m['loss_cp'] >= 30 and m['reference_score'] > -450]
    sf = Engine(sys.argv[1])
    sf.configure({'Threads': 1, 'Hash': 64})
    result = []
    for row in positions:
        board = chess.Board(row['fen'])
        record = dict(row)
        record['scaling'] = []
        choices = {row['move'], row['reference_move']}
        for cap in (65536, 262144, 1048576, 4194304):
            search.clear_tables()
            search.set_game_history([])
            tt.tt_clear(search.WORK.table)
            move = search.think(board, 3600000, increment_ms=0, node_limit=cap)
            record['scaling'].append({'limit': cap, 'move': move, 'search': search.last_search()})
            choices.add(move)
        record['judged'] = {}
        for move in sorted(choices):
            sf.configure({'Clear Hash': None})
            info = sf.analyse(board, chess.engine.Limit(nodes=1000000),
                              root_moves=[chess.Move.from_uci(move)])
            record['judged'][move] = info['score'].pov(board.turn).score(mate_score=10000)
        result.append(record)
        (ROOT / 'audit' / 'error-scaling.json').write_text(json.dumps(result, indent=2)+'\n')
        print(f"{len(result)}/{len(positions)}: {row['fen']} "
              f"{[(r['move'], record['judged'][r['move']]) for r in record['scaling']]}",
              flush=True)
    sf.quit()


if __name__ == '__main__':
    main()
