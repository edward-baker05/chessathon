# The engine in C++

A port of the Python engine (`search.py`, `position.py`, `movegen.py`, `bitboard.py`, `tt.py`,
`nnue.py`) to a single UCI binary, with `weights/net.npz` embedded in it.

It is a port rather than a rewrite: it makes the same decisions in the same order and hashes
with the Python engine's own Zobrist keys, so a fixed-node search here visits the same tree
as one there, with the same move, depth, score and node count. `tests/test_cpp.py` holds it to
that, and it is the check to run after changing either engine's search.

```
make -C cpp                 # build cpp/build/engine for this machine
make -C cpp perft           # move generator against known node counts
make -C cpp bench           # node rate over tests/bench.py's positions
uv run pytest tests/test_cpp.py
```

The build needs g++ (or clang++) with C++20, and `uv` to export the network: the npz is
deflate-compressed, so `tools/export_cpp.py net` flattens it to a blob the build links into
`.rodata`. `make ARCH=x86-64-v3` builds a binary that runs on any AVX2 machine rather than
only this one.

## Playing it

- The harness: `cpp/agent.py` relays `get_move` to the binary, so `cpp` is an agent directory
  like any other: `uv run python -m harness.play --white cpp --black .`
- fastchess and `tools/sprt.py`: the binary is a native UCI engine that carries its network,
  so `make sprt DEV=cpp/build/engine` plays it against HEAD directly.
- Anything else that speaks UCI: `id name chessathon-cpp`, with a `Hash` option in MB.

Beyond UCI it answers `perft N`, `eval`, `d` and `bench [nodes]`, and runs its command-line
arguments as one command, so `cpp/build/engine bench` works.

## Where it deliberately differs

- Game tracking. Like `agent.py`, a bare `position fen` one legal move on from our last reply
  continues the game, and anything else starts a new one and clears the tables. `agent.py`
  compares raw en passant squares, and python-chess records one after every double push
  while a FEN only shows it when the capture is legal, so any double push the opponent makes
  that cannot be taken reads there as a new game: the transposition table, the history
  tables and the repetition history are all thrown away mid-game. This port compares
  positions the way the Zobrist key does, en passant only where it is capturable.
- `position ... moves ...` is used as the repetition history directly, and `ucinewgame`
  clears the tables.
- `go movetime` searches for that long, as UCI means it. The zygote hands a movetime to the
  Python agent as a whole clock, which spends a small share of it.
- The search runs on its own thread, so `stop` and `go infinite` work.
- Beyond 2048 plies the repetition history keeps the most recent positions rather than the
  earliest.
