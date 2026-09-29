# Working in this repo

A chess engine in C++: a UCI binary with an NNUE evaluation whose network is embedded in it.
The old Python engine it was ported from lives on the `python-old` branch.

## Layout

- `cpp/src/` the engine: `bitboard`, `position`, `movegen`, `nnue`, `search`, `tt`, and `main`
  for UCI. `cpp/README.md` describes the UCI surface and how the engine tracks a game.
- `weights/net.npz` the network. `tools/export_cpp.py` flattens it to a blob the build links
  into the binary, so the engine carries its network with it.
- `tools/` net training (`extract`, `dataset`, `train`, `quantise`, `random_net`) and
  `sprt.py`, the strength tester.
- `tests/` pytest, driving the built binary over UCI, plus the dataset and sprt tools.
- `lichess/` the lichess-bot config and launcher.

## Working notes

- The process starts once per game. It starts in milliseconds, so nothing needs warming.
- `cpp/src/zobrist.inc` and the move, score and stack encodings in `types.h` are fixed: a
  change to them changes every hash and every search.
- Never name a Python file after a module you import (`chess.py`, `types.py`, `random.py`).
- Only ship a network you trained yourself.

## Verify

```
make -C cpp          # build cpp/build/engine; `perft` and `bench` targets too
make test            # builds, then pytest
make gate            # ruff, mypy, perft and pytest
```

Judge strength changes with `make sprt` (fastchess, installed once by `make fastchess-setup`):
it builds the working tree and HEAD and plays them until an SPRT decides, which hundreds of
games cannot do for the 5 to 15 Elo most changes are worth. Builds are frozen into
`sprt/<run>/` first, so the tree is free to edit while it runs. `tools/sprt.py --help` lists
the options.

## Style

C++20, warning clean under `-Wall -Wextra -Wshadow`. Python 3.12, type-annotated, ruff and
mypy strict clean. Match the comment density and naming of the surrounding code.
