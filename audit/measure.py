"""Fresh positions and measurements. Offline analysis only; never packaged.

Examples (from repository root):
  .venv/bin/python audit/measure.py generate --stockfish /path/to/stockfish
  .venv/bin/python audit/measure.py bench --engine . --out audit/baseline.json
  .venv/bin/python audit/measure.py judge --stockfish /path/to/stockfish
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
from uci import Engine

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / 'audit' / 'positions.json'
OPENINGS = [
    'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7',
    'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6',
    'd4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O',
    'd4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O',
    'e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7',
    'e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6',
    'c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6',
    'd4 Nf6 c4 e6 Nf3 d5 g3 Be7 Bg2 O-O',
]


def generate(path: str) -> None:
    rng = random.Random(20260909)
    engine = Engine(path)
    engine.configure({'Threads': 1, 'Hash': 64})
    result: list[dict[str, Any]] = []
    for game, opening in enumerate(OPENINGS):
        board = chess.Board()
        for san in opening.split():
            board.push_san(san)
        engine.configure({'Clear Hash': None})
        for ply in range(120):
            if board.is_game_over(claim_draw=True):
                break
            if ply in (4, 16, 32, 52, 76, 104):
                result.append({'id': f'g{game}-p{board.ply()}', 'fen': board.fen()})
            info = engine.analyse(board, chess.engine.Limit(nodes=10000),
                                  multipv=3 if ply < 8 else 1)
            best = info[0]['score'].pov(board.turn).score(mate_score=10000)
            choices = [line for line in info
                       if line['score'].pov(board.turn).score(mate_score=10000) >= best - 70]
            board.push(rng.choice(choices)['pv'][0])
        print(f'generated opening {game + 1}, total {len(result)}', flush=True)
    SUITE.write_text(json.dumps({'stockfish': engine.id, 'seed': 20260909,
                                'positions': result}, indent=2) + '\n')
    engine.quit()


def bench(directory: str, output: str) -> None:
    sys.path.insert(0, str(Path(directory).resolve()))
    started = time.perf_counter()
    import numpy as np

    import nnue
    import search
    import tt

    loaded = time.perf_counter() - started
    positions = json.loads(SUITE.read_text())['positions']
    result: dict[str, Any] = {'import_seconds': loaded, 'engine': str(Path(directory).resolve()),
                             'weights_sha256': hashlib.sha256(
                                 nnue.WEIGHTS_PATH.read_bytes()).hexdigest(),
                             'l1': nnue.L1, 'qa': nnue.QA, 'positions': []}
    for row in positions:
        board = chess.Board(row['fen'])
        record = dict(row)
        record['searches'] = []
        for limit in (16384, 65536, 262144, 1048576):
            tt.tt_clear(search.WORK.table)
            search.clear_tables()
            search.set_game_history([])
            search.set_pruning(True)
            before = time.perf_counter()
            move = search.think(board, 3_600_000, increment_ms=0, node_limit=limit)
            elapsed = time.perf_counter() - before
            assert chess.Move.from_uci(move) in board.legal_moves
            depth, seldepth, score, nodes, _ = search.last_search()
            entry = {'limit': limit, 'move': move, 'depth': depth, 'seldepth': seldepth,
                     'score': score, 'nodes': nodes, 'seconds': elapsed}
            if search.WORK.ints.size > 10:
                entry['counters'] = search.WORK.ints[10:].tolist()
            record['searches'].append(entry)
        record['static'] = int(nnue.forward(search.WORK.acc, 0, search.WORK.state[0]))
        result['positions'].append(record)
        Path(output).write_text(json.dumps(result, indent=2) + '\n')
        print(f"{row['id']}: {[(x['depth'], x['move']) for x in record['searches']]}",
              flush=True)
    # An independent int64 forward reference over all root positions.
    errors = []
    for row in positions:
        search._prepare(chess.Board(row['fen']), 3600000, 0, 0, search.WORK)
        acc = search.WORK.acc[0].astype(np.int64)
        stm = int(search.WORK.state[0, 8])
        count = chess.Board(row['fen']).occupied.bit_count()
        bucket = (count - 2) // nnue.BUCKET_DIVISOR
        active = np.concatenate([acc[stm], acc[1 - stm]]).clip(0, nnue.QA)**2
        total = int((active * nnue.OUT_WEIGHT[bucket].astype(np.int64)).sum())
        expected = (total // nnue.QA + int(nnue.OUT_BIAS[bucket])) * nnue.SCALE // (
            nnue.QA * nnue.QB)
        expected = min(nnue.EVAL_LIMIT, max(-nnue.EVAL_LIMIT, expected))
        errors.append(int(nnue.forward(search.WORK.acc, 0, search.WORK.state[0])) - expected)
    result['integer_forward_max_error'] = max(abs(x) for x in errors)
    Path(output).write_text(json.dumps(result, indent=2) + '\n')


def judge(path: str) -> None:
    engine = Engine(path)
    engine.configure({'Threads': 1, 'Hash': 64})
    runs = {p.stem: json.loads(p.read_text()) for p in (ROOT / 'audit').glob('bench-*.json')}
    positions = json.loads(SUITE.read_text())['positions']
    result: list[dict[str, Any]] = []
    for index, row in enumerate(positions):
        board = chess.Board(row['fen'])
        moves = {x['move'] for run in runs.values()
                 for x in run['positions'][index]['searches']}
        engine.configure({'Clear Hash': None})
        info = engine.analyse(board, chess.engine.Limit(nodes=1000000))
        best_move = info['pv'][0].uci()
        moves.add(best_move)
        entry = dict(row)
        entry['reference_move'] = best_move
        entry['reference_score'] = info['score'].pov(board.turn).score(mate_score=10000)
        entry['choices'] = {}
        for move in sorted(moves):
            engine.configure({'Clear Hash': None})
            info = engine.analyse(board, chess.engine.Limit(nodes=500000),
                                  root_moves=[chess.Move.from_uci(move)])
            entry['choices'][move] = info['score'].pov(board.turn).score(mate_score=10000)
        result.append(entry)
        (ROOT / 'audit' / 'reference.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f"judged {index + 1}/{len(positions)}", flush=True)
    engine.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['generate', 'bench', 'judge'])
    parser.add_argument('--stockfish', default='stockfish')
    parser.add_argument('--engine', default=str(ROOT))
    parser.add_argument('--out', default=str(ROOT / 'audit' / 'bench-baseline.json'))
    args = parser.parse_args()
    if args.mode == 'generate':
        generate(args.stockfish)
    elif args.mode == 'bench':
        bench(args.engine, args.out)
    else:
        judge(args.stockfish)
