"""Quantised evaluations from the engine's own runtime, for tools/evalnet.py.

The point of running this out of process is that it loads `nnue` exactly as `agent.py`
does and evaluates through the same jitted forward pass. A second implementation of the
int16 arithmetic written inside the comparison tool could agree with the trainer and
disagree with the thing that actually plays, which is the failure it exists to catch.

Reads a JSON list of FENs on stdin, prints one JSON line. `CHESSATHON_WEIGHTS` chooses the
network. Also reports what importing cost and how much memory the process peaked at, which
are the two runtime numbers a wider input layer moves and a validation loss does not show.
"""

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_started = time.perf_counter()
import chess  # noqa: E402

import nnue  # noqa: E402
import position  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - _started


def main() -> int:
    fens = json.loads(sys.stdin.read())
    state, mail = position.new_stacks()
    accumulator = nnue.new_accumulator(2)
    scores: list[int] = []
    for fen in fens:
        position.encode(chess.Board(fen), state[0], mail[0])
        nnue.refresh(accumulator, 0, state[0], mail[0])
        scores.append(int(nnue.forward(accumulator, 0, state[0])))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(json.dumps({
        "cp": scores,
        "import_seconds": IMPORT_SECONDS,
        "peak_mb": peak,
        "king_buckets": nnue.KING_BUCKETS,
        "l1": nnue.L1,
        "qa": nnue.QA,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
