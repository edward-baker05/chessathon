SHELL := /bin/bash

.PHONY: setup engine test gate perft bench sprt fastchess-setup random-net data train quantise lichess-setup lichess

setup:
	uv sync

# The C++ engine, cpp/build/engine. See cpp/README.md.
engine:
	$(MAKE) -C cpp

test: engine
	uv run pytest -q

gate: engine
	uv run ruff check .
	uv run mypy
	$(MAKE) -C cpp perft
	uv run pytest -q

perft:
	$(MAKE) -C cpp perft

bench:
	$(MAKE) -C cpp bench

# Strength tests. The working tree (or DEV) against HEAD (or BASE) under fastchess, until an
# SPRT decides; see tools/sprt.py for what a build can be. ARGS go to tools/sprt.py.
sprt:
	uv run python tools/sprt.py --base $(if $(BASE),$(BASE),HEAD) $(if $(DEV),--dev $(DEV)) $(ARGS)

FASTCHESS_TAG := v1.8.2-alpha

fastchess-setup:
	if [ ! -d third_party/fastchess ]; then git clone --depth 1 --branch $(FASTCHESS_TAG) \
		https://github.com/Disservin/fastchess.git third_party/fastchess; fi
	$(MAKE) -C third_party/fastchess -j
	mkdir -p third_party/books
	if [ ! -f third_party/books/UHO_Lichess_4852_v1.epd ]; then cd third_party/books && \
		curl -sSfLO https://raw.githubusercontent.com/official-stockfish/books/master/UHO_Lichess_4852_v1.epd.zip && \
		unzip -q UHO_Lichess_4852_v1.epd.zip && rm UHO_Lichess_4852_v1.epd.zip; fi

# A randomly initialised network in the shipped format. Plays badly by construction; it
# exists so the engine can be tested before any training has happened.
random-net:
	uv run python tools/random_net.py

train:
	uv run python tools/train.py $(if $(EPOCHS),--epochs $(EPOCHS))

quantise:
	uv run python tools/quantise.py

# Play on lichess through lichess-bot, which starts cpp/build/engine once per game. ARGS go
# to lichess-bot, e.g. `make lichess ARGS=-u` upgrades the account.
lichess-setup:
	if [ -d lichess/lichess-bot ]; then git -C lichess/lichess-bot pull --ff-only; \
	else git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git lichess/lichess-bot; fi
	uv venv --allow-existing --python .venv/bin/python lichess/lichess-bot/venv
	uv pip install --python lichess/lichess-bot/venv -r lichess/lichess-bot/requirements.txt

lichess:
	lichess/run.sh $(ARGS)
