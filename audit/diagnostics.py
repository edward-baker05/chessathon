"""Additional reproducible current-source diagnostics; no historical data used."""

import json
import sys
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
import movegen  # noqa: E402
import nnue  # noqa: E402
import position  # noqa: E402
import search  # noqa: E402
import tt  # noqa: E402


@njit(cache=False)
def micro(work: Any, repeats: int, mode: int) -> int:
    total = 0
    for i in range(repeats):
        move = work.moves[i % 20]
        if mode == 0:
            total += position.see(work.state[0], work.mail[0], move)
        elif mode == 1:
            total += movegen.generate(work.state[0], work.moves, 0)
        elif mode == 2:
            position.make(work.state[0], work.mail[0], work.state[1], work.mail[1], move)
            total += int(work.state[1, 11])
        elif mode == 3:
            nnue.apply(work.acc, 0, work.state[0], work.mail[0], move)
            total += nnue.forward(work.acc, 1, work.state[1])
        else:
            search.score_moves(work, 0, 0, 20, np.int32(0))
            total += work.scores[i % 20]
    return total


def main() -> None:
    work = search.WORK
    result: dict[str, Any] = {'positions': []}
    fens = [
        '7k/5K2/6Q1/8/8/8/8/8 b - - 0 1',
        '8/8/8/8/8/5k2/8/R6K w - - 100 1',
        '8/8/8/8/8/5k2/8/6BK w - - 0 1',
        '8/k1P5/2K5/8/8/8/8/8 w - - 0 1',
        '4k3/4n3/8/3p4/2B5/8/8/K3R3 w - - 0 1',
    ]
    for fen in fens:
        board = chess.Board(fen)
        search._prepare(board, 3600000, 0, 0, work)
        search.set_game_history([])
        tt.tt_clear(work.table)
        search.clear_tables()
        record: dict[str, Any] = {'fen': fen,
                                  'outcome': str(board.outcome(claim_draw=True))}
        record['static'] = int(nnue.forward(work.acc, 0, work.state[0]))
        record['qsearch'] = int(search.qsearch(work, 0, np.int32(-32000), np.int32(32000)))
        count = movegen.generate(work.state[0], work.moves, 0)
        record['see'] = {movegen.move_to_uci(int(m)): int(position.see(
            work.state[0], work.mail[0], m)) for m in work.moves[:count].copy()}
        if fen.startswith('8/k1P5'):
            record['delta'] = []
            for alpha in (record['static'] + 301, record['static'] + 501):
                for pruning in (True, False):
                    search._prepare(board, 3600000, 0, 0, work)
                    search.set_pruning(pruning)
                    value = int(search.qsearch(work, 0, np.int32(alpha), np.int32(alpha + 1)))
                    record['delta'].append({'alpha': alpha, 'pruning': pruning, 'value': value})
        result['positions'].append(record)
    search.set_pruning(True)
    search._prepare(chess.Board(), 3600000, 0, 0, work)
    movegen.generate(work.state[0], work.moves, 0)
    result['micro_ns'] = {}
    for mode, name in enumerate(('see_quiet', 'generate_startpos', 'copy_make',
                                 'nnue_update_and_forward', 'score_20_quiets')):
        micro(work, 100, mode)
        durations = []
        for _ in range(5):
            started = time.perf_counter_ns()
            micro(work, 100000, mode)
            durations.append((time.perf_counter_ns() - started) / 100000)
        result['micro_ns'][name] = durations
    result['clock'] = []
    for left in (120000, 30000, 5000, 1000, 500, 300, 100):
        tt.tt_clear(work.table)
        search.clear_tables()
        before = time.perf_counter()
        move = agent.get_move(chess.STARTING_FEN, left)
        result['clock'].append({'left_ms': left, 'elapsed_ms': (time.perf_counter() - before)*1000,
                                 'budget': search.budget_of(), 'search': search.last_search(),
                                 'move': move})
    result['compiled_see_allocates'] = 'NRT_MemInfo_alloc' in position.see.inspect_llvm(
        position.see.signatures[0])
    result['net'] = {'l1': nnue.L1, 'buckets': nnue.BUCKETS, 'qa': nnue.QA, 'qb': nnue.QB}
    (ROOT / 'audit' / 'diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
