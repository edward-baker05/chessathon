SHELL := /bin/bash

.PHONY: setup play arena gate test bench ab replay random-net data train quantise lichess-setup lichess

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/numba $(if $(FEN),--fen "$(FEN)") --pgn game.pgn

arena:
	uv run python -m harness.arena --opponent baselines/numba --games 20

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

train:
	uv run python tools/train.py $(if $(EPOCHS),--epochs $(EPOCHS))

quantise:
	uv run python tools/quantise.py

# Play on lichess through lichess-bot. See lichess/zygote.py for why the agent runs behind a
# fork server. ARGS go to lichess-bot, e.g. `make lichess ARGS=-u` upgrades the account.
lichess-setup:
	if [ -d lichess/lichess-bot ]; then git -C lichess/lichess-bot pull --ff-only; \
	else git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git lichess/lichess-bot; fi
	uv venv --allow-existing --python .venv/bin/python lichess/lichess-bot/venv
	uv pip install --python lichess/lichess-bot/venv -r lichess/lichess-bot/requirements.txt

lichess:
	lichess/run.sh $(ARGS)
