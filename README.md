# Chess engine

A chess engine in Python: a numba-jitted bitboard search with an NNUE evaluation. The whole
engine is `agent.py`, which exposes one function:

```python
def get_move(fen: str, time_left_ms: int) -> str:  # UCI, e.g. "e2e4" or "e7e8q"
```

The process starts once per game and stays alive between moves, so module state (the
transposition table, position history) lasts for one game.

```
make setup
make play
```

## Commands

```
make play                                          # one game, 120 s + 0.5 s
make play FEN="<fen>"                              # start from a given position
make arena                                         # 20 fast games against a baseline
make test                                          # pytest
make gate                                          # ruff, mypy, and two games that must finish
make bench                                         # import time and search speed
make ab OPPONENT=<dir>                             # fixed-opening A/B match, in tests/match.py
make fastchess-setup                               # build fastchess, fetch the opening book
make sprt [BASE=<ref>] [DEV=<ref|dir|binary>]      # SPRT under fastchess, see tools/sprt.py
make replay PGN=<file>                             # time allocation over a played game
make train / quantise                            # train and ship a new network
make lichess-setup / lichess                       # play on lichess through lichess-bot
uv run python -m harness.arena --opponent ../my-old-version --games 200
```

Anything the agent prints goes to stderr, so `print` debugging works.

## Layout

```
agent.py             get_move, game tracking, move validation
search.py            iterative-deepening alpha-beta, time management
position.py          position encoding and make/unmake
movegen.py           pseudo-legal move generation
bitboard.py          attack tables and Zobrist keys
tt.py                transposition table
nnue.py, evaluate.py the network runtime and evaluation
weights/net.npz      the trained network
tools/               dataset extraction, training, quantisation, replay
baselines/           random, greedy, minimax and numba opponents
harness/             the referee, clock and subprocess protocol used for local games
tests/               unit tests, benchmarks and the A/B match runner
lichess/             lichess-bot bridge
docs/IDEAS.md        where the strength comes from
audit/, logs/        historical measurements and game logs
```

`harness/rules.py` holds the default time control and timeouts for local games. All of them can
be overridden on the command line.
