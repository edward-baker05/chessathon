"""The harness entrypoint for the C++ engine: get_move relayed to build/engine over UCI.

`uv run python -m harness.play --white cpp` plays it like any other agent directory. The
engine does its own game tracking, as agent.py does for the Python engine, so this is only
a pipe: a bare FEN in, a move out.
"""

import os
import subprocess
from pathlib import Path
from typing import IO

ENGINE = Path(__file__).resolve().parent / "build" / "engine"

# The same knobs agent.py reads, so the harness can drive either engine the same way.
INCREMENT_MS = int(os.environ.get("ENGINE_INCREMENT_MS", "500"))
NODE_LIMIT = int(os.environ.get("ENGINE_NODE_LIMIT", "0"))

if not ENGINE.exists():
    raise FileNotFoundError(f"{ENGINE} is missing; build it with `make -C cpp`")

_engine = subprocess.Popen(
    [str(ENGINE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
)


def _pipe(stream: IO[str] | None) -> IO[str]:
    if stream is None:
        raise RuntimeError("the engine process exposed no pipe")
    return stream


def _send(line: str) -> None:
    stdin = _pipe(_engine.stdin)
    stdin.write(line + "\n")
    stdin.flush()


def _await(prefix: str) -> str:
    for line in _pipe(_engine.stdout):
        if line.startswith(prefix):
            return line.strip()
    raise RuntimeError(f"the engine exited with code {_engine.wait()}")


_send("uci")
_await("uciok")
_send("isready")
_await("readyok")


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation. Our colour is the side to move in `fen`."""
    _send(f"position fen {fen}")
    command = (
        f"go wtime {time_left_ms} btime {time_left_ms} winc {INCREMENT_MS} binc {INCREMENT_MS}"
    )
    if NODE_LIMIT:
        command += f" nodes {NODE_LIMIT}"
    _send(command)
    return _await("bestmove").split()[1]
