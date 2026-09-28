"""Make disposable, instrumented copies of current source without touching the engine.

Counters 10..15: qnodes, nodes after root beta breach, root calls, window retries,
nodes at last completed iteration, negamax-to-qsearch transitions.
"""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = Path('/tmp/chessathon-audit-variants')


def replace(source: str, before: str, after: str) -> str:
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after)


def build() -> None:
    source = (ROOT / 'search.py').read_text()
    source = replace(source, 'INT_SLOTS = 10', 'INT_SLOTS = 16')
    source = replace(source, '    work.ints[I_NODES] = 0',
                     '    work.ints[10:] = 0\n    work.ints[I_NODES] = 0')
    source = replace(source, '    if depth <= 0:\n        return qsearch',
                     '    if depth <= 0:\n        work.ints[15] += 1\n        return qsearch')
    start = source.index('def qsearch(')
    end = source.index('\n\n@njit', start)
    section = replace(source[start:end], '    work.ints[I_NODES] += 1',
                      '    work.ints[10] += 1\n    work.ints[I_NODES] += 1')
    source = source[:start] + section + source[end:]
    source = replace(source, '            for index in range(base, count):\n',
                     '            for index in range(base, count):\n'
                     '                root_before = work.ints[I_NODES]\n'
                     '                work.ints[12] += 1\n')
    source = replace(source, '                if work.ints[I_ABORT] != 0:\n'
                     '                    break\n',
                     '                if local_alpha >= beta:\n'
                     '                    work.ints[11] += work.ints[I_NODES] - root_before\n'
                     '                if work.ints[I_ABORT] != 0:\n                    break\n')
    source = replace(source, '            if score <= alpha:\n',
                     '            if score <= alpha:\n                work.ints[13] += 1\n')
    source = replace(source, '            if score >= beta:\n',
                     '            if score >= beta:\n                work.ints[13] += 1\n')
    source = replace(source, '        work.ints[I_DEPTH] = depth\n',
                     '        work.ints[14] = work.ints[I_NODES]\n'
                     '        work.ints[I_DEPTH] = depth\n')
    beta = replace(source, '                        local_alpha = value\n',
                   '                        local_alpha = value\n'
                   '                        if local_alpha >= beta:\n'
                   '                            break\n')
    history = replace(beta, '                work.played[0] = move\n',
                      '                work.played[0] = move\n'
                      '                work.moved_piece[0] = mail[np.int64(move & 63)]\n')
    # The tried list must exclude illegal/pruned moves as well as captures. Keep this
    # follow-on experiment to the isolated root-context fix rather than half fixing history.
    variants = {'baseline': source, 'root-beta': beta, 'root-context': history}
    for name, code in variants.items():
        directory = DEST / name
        directory.mkdir(parents=True, exist_ok=True)
        for path in ROOT.glob('*.py'):
            shutil.copy2(path, directory / path.name)
        shutil.copytree(ROOT / 'weights', directory / 'weights', dirs_exist_ok=True)
        (directory / 'search.py').write_text(code)
    paths = [*ROOT.glob('*.py'), ROOT / 'weights' / 'net.npz']
    (ROOT / 'audit' / 'source-hashes.json').write_text(json.dumps({
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }, indent=2) + '\n')


if __name__ == '__main__':
    build()
