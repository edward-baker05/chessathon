# The C++ engine. `make` builds build/engine, a UCI binary with weights/net.npz embedded.
#
#   make                 build for this machine
#   make ARCH=x86-64-v3  build for any AVX2 machine, e.g. to copy elsewhere
#   make perft           the move generator against known node counts
#   make bench           node rate over tests/bench.py's positions

CXX ?= g++
ARCH ?= native
BUILD := build
ROOT := $(abspath ..)
NET := $(ROOT)/weights/net.npz
BLOB := $(abspath $(BUILD))/net.bin

CXXFLAGS := -std=c++20 -O3 -march=$(ARCH) -flto=auto -DNDEBUG -Wall -Wextra -Wshadow
LDFLAGS := -pthread

SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(SOURCES:src/%.cpp=$(BUILD)/%.o)

.PHONY: all perft bench clean

all: $(BUILD)/engine

$(BUILD)/engine: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# The network is assembled into nnue.o, so a new net rebuilds that object and relinks.
$(BUILD)/nnue.o: $(BLOB)
$(BUILD)/nnue.o: CXXFLAGS += -DNNUE_BLOB='"$(BLOB)"'

$(BLOB): $(NET) $(ROOT)/tools/export_cpp.py | $(BUILD)
	cd $(ROOT) && uv run python tools/export_cpp.py net $(BLOB)

$(BUILD):
	mkdir -p $@

# Expected counts are tests/test_perft.py's.
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

-include $(OBJECTS:.o=.d)
