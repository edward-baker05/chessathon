# Chessathon

A UCI chess engine in C++: bitboard move generation, iterative-deepening PVS with a shared
transposition table (Lazy SMP), and an NNUE evaluation whose network is embedded in the binary.
It began as a Python/numba engine, kept on the `python-old` branch.

## Building

```
make                  # build/engine, for this machine
make ARCH=x86-64-v3   # a binary that runs on any AVX2 machine
make perft            # the move generator against known node counts
make bench            # node rate over a fixed set of positions
```

Needs g++ (or clang++) with C++20, and `uv`: the network is deflate-compressed in
`weights/net.npz`, so `tools/export_net.py` flattens it to a blob the build links into the
binary. `make PYTHON=python3` uses another interpreter with numpy for that step.

The engine speaks UCI on stdin/stdout; see [docs/uci.md](docs/uci.md) for the commands,
options and how it tracks a game.

## Development

```
make test                          # pytest, against the built engine
make gate                          # ruff, mypy, perft and pytest
make fastchess-setup               # build fastchess, fetch the opening book
make sprt [BASE=<ref>] [DEV=<ref|binary>]
                                   # SPRT under fastchess; builds the working tree and HEAD
make train / quantise              # train and ship a new network
make lichess-setup / lichess       # play on lichess through lichess-bot
```

## Layout

```
src/                 the engine
weights/net.npz      the trained network
tools/               dataset extraction, training, quantisation, network export, sprt
tests/               pytest, driving the binary over UCI
lichess/             lichess-bot config and launcher
docs/uci.md          the UCI surface
```
