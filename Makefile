SHELL := /bin/bash

.PHONY: setup play arena zip gate test bench ab replay random-net data train quantise

# agent.py freezes its increment from the environment at import and defaults to the rated
# 500ms. The harness plays its fast games at 100ms, and nothing forwards that, so an agent
# in a fast local game budgets for five times the increment it is given and overspends its
# clock every move. Forwarding it is the launcher's job: the harness mirrors the platform
# and is not to be edited. `play` uses the rated control, so it takes the rated increment.
RATED_INCREMENT_MS := 500
FAST_INCREMENT_MS := 100

setup:
	uv sync

play:
	CHESSATHON_INCREMENT_MS=$(RATED_INCREMENT_MS) \
	uv run python -m harness.play --white . --black baselines/numba $(if $(FEN),--fen "$(FEN)") --pgn game.pgn

arena:
	CHESSATHON_INCREMENT_MS=$(FAST_INCREMENT_MS) \
	uv run python -m harness.arena --opponent baselines/numba --games 20

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	CHESSATHON_INCREMENT_MS=$(FAST_INCREMENT_MS) \
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

test:
	uv run pytest -q

# Nodes to depth, then throughput. The first is deterministic and is the one to A/B; the
# second moves with whatever else the machine is doing. Neither measures strength.
bench:
	uv run python tests/bench.py $(if $(DEPTH),--depth $(DEPTH)) $(if $(NODES),--nodes $(NODES))

# Time allocation over a played game, at the real control. Cheap evidence about the clock
# before spending arena hours on an A/B that measures strength.
replay:
	uv run python tools/replay.py $(if $(PGN),"$(PGN)",logs/*.pgn) --side $(if $(SIDE),$(SIDE),Edward)

# The only thing here that measures strength. Openings are paired and the pair is the unit,
# so the interval it reports is honest about two games sharing one opening.
ab:
	uv run python tests/match.py --opponent $(OPPONENT) --games $(if $(GAMES),$(GAMES),200) \
		--nodes $(if $(NODES),$(NODES),200000) $(if $(WORKERS),--workers $(WORKERS))

# A randomly initialised network in the shipped format. Plays badly by construction; it
# exists so the runtime can be tested before any training has happened.
random-net:
	uv run python tools/random_net.py

# Extract training positions from the evaluation dump into the packed binary the trainer
# memory maps. SOURCE is the .jsonl.zst; MIN_DEPTH is the engine depth floor for a label.
data:
	uv run python tools/extract.py $(if $(SOURCE),--source "$(SOURCE)") \
		$(if $(MIN_DEPTH),--min-depth $(MIN_DEPTH)) $(if $(OUT),--out "$(OUT)")

train:
	uv run python tools/train.py $(if $(EPOCHS),--epochs $(EPOCHS))

quantise:
	uv run python tools/quantise.py
