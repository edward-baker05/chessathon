// Quantised NNUE evaluation: `768 -> L1x2 -> 1` with squared clipped ReLU and piece-count
// output buckets, evaluated in integer arithmetic.
//
// The network is embedded in the binary at build time (see the Makefile), so the engine
// is one file that carries its own weights.

#pragma once

#include <string>

#include "position.h"

namespace nnue {

// The shape is fixed at compile time so every loop bound is a constant the compiler can
// vectorise. load() refuses a network of any other shape rather than misreading it.
#ifndef NNUE_L1
#define NNUE_L1 512
#endif
#ifndef NNUE_BUCKETS
#define NNUE_BUCKETS 8
#endif
constexpr int L1 = NNUE_L1;
constexpr int OUTPUT_BUCKETS = NNUE_BUCKETS;
constexpr int NUM_FEATURES = 768;

// An evaluation is clamped to this. The search reserves scores near MATE for real mates.
constexpr int EVAL_LIMIT = 10000;

struct alignas(64) Accumulator {
    int16_t values[2][L1];  // [perspective][neuron]
};

// Reads the embedded network. Returns false, with the reason in `error`, if it is malformed.
bool load(std::string& error);

// Rebuild both halves from the board. Called once per search, at the root.
void refresh(Accumulator& acc, const Position& pos);

// Write `dst` from `src` for one move. `before` is the position before the move: everything
// the update needs is derivable from it.
void apply(const Accumulator& src, Accumulator& dst, const Position& before, Move move);

// Centipawns, from the side to move's view.
int evaluate(const Accumulator& acc, const Position& pos);

}  // namespace nnue
