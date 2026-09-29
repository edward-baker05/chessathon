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

- lichess: `make lichess` builds the engine, checks perft, and runs lichess-bot, which starts
  `cpp/build/engine` once per game. It ponders on the opponent's time; `ponder` in
  `lichess/config.yml` turns that off.
- The harness: `cpp/agent.py` relays `get_move` to the binary, so `cpp` is an agent directory
  like any other: `uv run python -m harness.play --white cpp --black .`
- fastchess and `tools/sprt.py`: the binary carries its network, so
  `make sprt DEV=cpp/build/engine` plays it against HEAD directly.

## UCI

Every command and `go` parameter in the specification is accepted, in any order, and
anything not understood is skipped, as the specification asks: `joho debug on` turns
debugging on. `isready` is answered mid-search. `tests/test_cpp_uci.py` sends the awkward
forms as well as the usual ones.

| Input | What it does |
| --- | --- |
| `go wtime btime winc binc` | the Python engine's time allocation, from our side's clock |
| `go movetime N` | searches for N ms, less 10 ms to answer |
| `go nodes N`, `go depth N` | stops there |
| `go infinite`, a bare `go` | searches until `stop`; `infinite` also holds `bestmove` until then |
| `go ponder` ... `ponderhit` | searches on the opponent's time; the clock starts at `ponderhit` |
| `go searchmoves m1 m2 ...` | only those root moves; a list with none legal restricts nothing |
| `go movestogo`, `go mate` | accepted, not used yet |
| `setoption name Hash value N` | transposition table size in MB |
| `setoption name Clear Hash` | empties the table |
| `setoption name Threads value N` | searches with N threads (Lazy SMP); 1 by default |
| `Ponder` | declared and accepted; pondering follows `go ponder` |
| `debug on` | explains ignored input and new games in `info string` lines |
| `register` | accepted; nothing needs registering |

`bestmove` names a ponder move whenever the principal variation has one, and is
`bestmove (none)` when the side to move is mated or stalemated. A `position`, `go` or
`setoption` that arrives mid-search waits for a search that will end on its own and stops
one that will not.

Beyond UCI it answers `perft N`, `eval`, `d` and `bench [nodes]`, and runs its command-line
arguments as one command, so `cpp/build/engine bench` works.

## Game tracking

A `position ... moves ...` command is the game record, and is the repetition history. A game
record that passes through our last reply is the game being played, and anything else starts
a new one and clears the tables; so does `ucinewgame`. A bare `position fen`, which is all
the Python harness sends, continues the game when it is one legal move on from our last
reply, as in `agent.py`. A ponder search that is stopped rather than hit was never our move,
and does not count as a reply.

## Where it differs from the Python engine

The search and evaluation do not differ: that is what `tests/test_cpp.py` checks. Around
them:

- `Threads` above 1 runs Lazy SMP: helper threads search the same root with their own
  history tables and share the transposition table, and the main thread keeps the clock. A
  helper that completes a deeper iteration with a better score supplies the move. Such a
  search is not reproducible, and `go nodes` counts every thread's nodes.

- `go movetime` searches for that long, as UCI means it. The zygote hands a movetime to the
  Python agent as a whole clock, which spends a small share of it.
- Pondering, `searchmoves`, `stop` and `go infinite` exist only here.
- Beyond 2048 plies the repetition history keeps the most recent positions rather than the
  earliest.
