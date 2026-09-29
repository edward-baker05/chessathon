# Chess engine

A UCI chess engine in C++: bitboard move generation, iterative-deepening PVS with a shared
transposition table (Lazy SMP), and an NNUE evaluation whose network is embedded in the binary.

```
make -C cpp                  # builds cpp/build/engine
cpp/build/engine             # speaks UCI on stdin/stdout
```

## Commands

```
make engine                                        # build cpp/build/engine
make test                                          # pytest, against the built engine
make gate                                          # ruff, mypy, perft and pytest
make perft / bench                                 # move generator check, node rate
make fastchess-setup                               # build fastchess, fetch the opening book
make sprt [BASE=<ref>] [DEV=<ref|binary>]          # SPRT under fastchess, see tools/sprt.py
make train / quantise                              # train and ship a new network
make lichess-setup / lichess                       # play on lichess through lichess-bot
```

## Layout

```
cpp/src/             the engine
cpp/Makefile         the build; embeds weights/net.npz
weights/net.npz      the trained network
tools/               dataset extraction, training, quantisation, network export, sprt
tests/               pytest, driving the binary over UCI
lichess/             lichess-bot config and launcher
```
