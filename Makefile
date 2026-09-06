SHELL := /bin/bash

.PHONY: setup play arena zip gate test bench ab replay random-net data train quantise

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/numba $(if $(FEN),--fen "$(FEN)") --pgn game.pgn

arena:
	uv run python -m harness.arena --opponent baselines/numba --games 20

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

test:
	uv run pytest -q

bench:
	uv run python tests/bench.py

# Time allocation over a played game, at the real control. Cheap evidence about the clock
# before spending arena hours on an A/B that measures strength.
replay:
	uv run python tools/replay.py $(if $(PGN),"$(PGN)",logs/*.pgn) --side $(if $(SIDE),$(SIDE),Edward)

ab:
	uv run python tests/match.py --opponent $(OPPONENT) --games $(if $(GAMES),$(GAMES),200) --nodes $(if $(NODES),$(NODES),200000)

# A randomly initialised network in the shipped format. Plays badly by construction; it
# exists so the runtime can be tested before any training has happened.
random-net:
	uv run python tools/random_net.py

# Download the Lichess evaluation file and pack it into training data. CPU heavy, one off.
data:
	uv run python tools/extract.py $(if $(LIMIT),--limit $(LIMIT))

train:
	uv run python tools/train.py $(if $(EPOCHS),--epochs $(EPOCHS))

quantise:
	uv run python tools/quantise.py
