// Transposition table.
//
// Four entries per bucket, each a key and a packed data word, so a bucket is exactly one
// 64-byte cache line. The data word, bit by bit, with nothing spare:
//   0..15   score, biased
//   16..31  move, in the sixteen bit form of pack_move
//   32..39  depth
//   40..41  bound
//   42..47  age
//   48..63  static evaluation, biased
//
// The table is never cleared between moves of one game: last move's entries describe the
// same game. The UCI layer clears it only when a position does not chain onto the game.

#pragma once

#include <cstddef>
#include <memory>

#include "types.h"

constexpr int MATE = 30000;
constexpr int MATE_IN_MAX = MATE - 256;

constexpr int BOUND_NONE = 0;
constexpr int BOUND_UPPER = 1;
constexpr int BOUND_LOWER = 2;
constexpr int BOUND_EXACT = 3;

// Ages are compared cyclically, so everything that reads or advances one masks with this.
constexpr int AGE_MASK = 63;

constexpr size_t DEFAULT_HASH_MB = 128;

struct TTProbe {
    bool hit;
    int score;
    Move move;
    int depth;
    int bound;
    int static_eval;
};

class TranspositionTable {
public:
    explicit TranspositionTable(size_t megabytes = DEFAULT_HASH_MB) { resize(megabytes); }

    // Rounds down to a power of two buckets, and clears.
    void resize(size_t megabytes);
    void clear();

    // Stores an entry, rebasing mate scores to be relative to this node. A mate found at
    // ply N means "mate in K from here"; storing it unadjusted would make it mean "mate in
    // K from the root", which is the classic source of mate lines off by a few moves.
    void store(Key key, int ply, int score, Move move, int depth, int bound, int static_eval,
               int age);
    TTProbe probe(Key key, int ply) const;

private:
    struct alignas(64) Bucket {
        uint64_t slots[8];  // key, data, key, data, ...
    };
    struct FreeDeleter {
        void operator()(Bucket* p) const;
    };

    std::unique_ptr<Bucket[], FreeDeleter> buckets_;
    size_t mask_ = 0;
};
