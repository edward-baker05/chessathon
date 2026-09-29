# The engine. `make` builds build/engine, a UCI binary with weights/net.npz embedded.
#
#   make                 build for this machine
#   make ARCH=x86-64-v3  build for any AVX2 machine, e.g. to copy elsewhere
#   make perft           the move generator against known node counts
#   make bench           node rate over a fixed set of positions
#   make test / gate     pytest against the built engine; gate adds ruff and mypy

SHELL := /bin/bash

CXX ?= g++
ARCH ?= native
BUILD := build
NET := weights/net.npz
BLOB := $(abspath $(BUILD))/net.bin
# What exports the network; tools/sprt.py names the interpreter it is already running under.
PYTHON ?= uv run python

CXXFLAGS := -std=c++20 -O3 -march=$(ARCH) -flto=auto -DNDEBUG -Wall -Wextra -Wshadow
LDFLAGS := -pthread

SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(SOURCES:src/%.cpp=$(BUILD)/%.o)

.PHONY: all perft bench clean test gate sprt fastchess-setup random-net train quantise lichess-setup lichess

all: $(BUILD)/engine

$(BUILD)/engine: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# The network is assembled into nnue.o, so a new net rebuilds that object and relinks.
$(BUILD)/nnue.o: $(BLOB)
$(BUILD)/nnue.o: CXXFLAGS += -DNNUE_BLOB='"$(BLOB)"'

$(BLOB): $(NET) tools/export_net.py | $(BUILD)
	$(PYTHON) tools/export_net.py $@

$(BUILD):
	mkdir -p $@

# Expected counts are tests/test_cpp.py's.
perft: $(BUILD)/engine
	@check() { got=$$(printf 'position fen %s\nperft %s\nquit\n' "$$1" "$$2" | \
		$(BUILD)/engine | awk '/^nodes/ {print $$2}'); \
		if [ "$$got" = "$$3" ]; then echo "ok   perft($$2) = $$3  $$1"; \
		else echo "FAIL perft($$2) = $$got, expected $$3  $$1"; exit 1; fi; }; \
	check "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 5 4865609 && \
	check "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" 4 4085603 && \
	check "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" 5 674624 && \
	check "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1" 4 422333 && \
	check "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8" 4 2103487 && \
	check "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10" 4 3894594

bench: $(BUILD)/engine
	$(BUILD)/engine bench

clean:
	rm -rf $(BUILD)

test: all
	uv run pytest -q

gate: all
	uv run ruff check .
	uv run mypy
	$(MAKE) perft
	uv run pytest -q

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

# Play on lichess through lichess-bot, which starts build/engine once per game. ARGS go
# to lichess-bot, e.g. `make lichess ARGS=-u` upgrades the account.
lichess-setup:
	if [ -d lichess/lichess-bot ]; then git -C lichess/lichess-bot pull --ff-only; \
	else git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git lichess/lichess-bot; fi
	uv venv --allow-existing --python .venv/bin/python lichess/lichess-bot/venv
	uv pip install --python lichess/lichess-bot/venv -r lichess/lichess-bot/requirements.txt

lichess:
	lichess/run.sh $(ARGS)

-include $(OBJECTS:.o=.d)
