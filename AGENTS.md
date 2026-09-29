# Working in this repo

A chess engine. `agent.py` exposes `get_move(fen, time_left_ms) -> str` (UCI) and is the engine's
entry point; the search, evaluation and tooling live in the modules beside it. There are no
external rules or limits on it: time control, memory, dependencies and file size are all up to
the project.

## Layout

- `agent.py` game tracking and move validation around `search.think`.
- `search.py`, `position.py`, `movegen.py`, `bitboard.py`, `tt.py` the jitted search.
- `nnue.py`, `evaluate.py`, `weights/net.npz` the evaluation.
- `cpp/` the same engine ported to a C++ UCI binary; see `cpp/README.md`. It searches the same
  tree as the Python engine node for node, and `tests/test_cpp.py` keeps it that way, so a
  search or evaluation change made in one has to be made in both.
- `harness/` local referee, clock and subprocess protocol. Edit it freely.
- `tools/`, `tests/`, `lichess/`, `baselines/` training, tests, the lichess bridge and opponents.
- `audit/`, `logs/` historical measurements. Read-only records; do not rewrite them.

## Working notes

- The process starts once per game and stays alive between moves. Module state survives to the
  next move in the same game, so `agent.py` detects a new game and clears its tables.
- numba functions are warmed once at import so compilation stays off the game clock. Keep new
  jitted functions warmed with the argument types the real calls use.
- Never name a file after a module you import (`chess.py`, `types.py`, `random.py`).
- Only ship a network you trained yourself.

## Verify

```
make test      # pytest
make play      # one game against a baseline
make arena     # 20 fast games against a baseline, with a score
make gate      # ruff, mypy, and two games that have to finish cleanly
make bench     # import time and search speed
make -C cpp    # build the C++ engine; `perft` and `bench` targets too
```

Judge strength changes with `make ab` over hundreds of fixed-opening games, not a handful of
games.

## Style

Python 3.12, type-annotated, ruff and mypy strict clean. Match the comment density and naming of
the surrounding code.
