#include "tt.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace {

constexpr int ENTRIES_PER_BUCKET = 4;
// Score is stored biased so it survives as an unsigned 16-bit field.
constexpr int64_t SCORE_BIAS = 32768;

constexpr int SCORE_SHIFT = 0;
constexpr int MOVE_SHIFT = 16;
constexpr int DEPTH_SHIFT = 32;
constexpr int BOUND_SHIFT = 40;
constexpr int AGE_SHIFT = 42;
constexpr int STATIC_SHIFT = 48;

constexpr uint64_t MOVE_MASK = 0xFFFF;
constexpr uint64_t DEPTH_MASK = 0xFF;
constexpr uint64_t BOUND_MASK = 3;
constexpr uint64_t STATIC_MASK = 0xFFFF;

// A generator move is seventeen bits and the word has room for sixteen. The seventeenth is
// redundant: a promotion piece is one of four, so it is stored as `promo - 1` in two bits
// and restored on the way out; every other move has no promotion piece at all.
inline int64_t pack_move(Move move) {
    int64_t wide = move;
    int64_t flag = (wide >> 15) & 3;
    int64_t promo = flag == FLAG_PROMO ? ((wide >> 12) & 7) - 1 : 0;
    return (wide & 0xFFF) | (promo << 12) | (flag << 14);
}

inline Move unpack_move(int64_t packed) {
    int64_t flag = (packed >> 14) & 3;
    int64_t promo = flag == FLAG_PROMO ? ((packed >> 12) & 3) + 1 : 0;
    return static_cast<Move>((packed & 0xFFF) | (promo << 12) | (flag << 15));
}

inline uint64_t pack_entry(int score, Move move, int depth, int bound, int static_eval, int age) {
    return (static_cast<uint64_t>(score + SCORE_BIAS) << SCORE_SHIFT) |
           ((static_cast<uint64_t>(pack_move(move)) & MOVE_MASK) << MOVE_SHIFT) |
           ((static_cast<uint64_t>(depth) & DEPTH_MASK) << DEPTH_SHIFT) |
           ((static_cast<uint64_t>(bound) & BOUND_MASK) << BOUND_SHIFT) |
           ((static_cast<uint64_t>(age) & AGE_MASK) << AGE_SHIFT) |
           (static_cast<uint64_t>(static_eval + SCORE_BIAS) << STATIC_SHIFT);
}

inline int field(uint64_t data, int shift, uint64_t mask) {
    return static_cast<int>((data >> shift) & mask);
}

}  // namespace

void TranspositionTable::FreeDeleter::operator()(Bucket* p) const { std::free(p); }

void TranspositionTable::resize(size_t megabytes) {
    size_t count = 1;
    while (count * 2 * sizeof(Bucket) <= megabytes * 1024 * 1024) count *= 2;
    buckets_.reset();
    void* memory = std::aligned_alloc(alignof(Bucket), count * sizeof(Bucket));
    if (!memory) throw std::bad_alloc();
    buckets_.reset(static_cast<Bucket*>(memory));
    mask_ = count - 1;
    clear();
}

void TranspositionTable::clear() { std::memset(buckets_.get(), 0, (mask_ + 1) * sizeof(Bucket)); }

void TranspositionTable::store(Key key, int ply, int score, Move move, int depth, int bound,
                               int static_eval, int age) {
    int adjusted = score;
    if (score >= MATE_IN_MAX)
        adjusted = score + ply;
    else if (score <= -MATE_IN_MAX)
        adjusted = score - ply;

    uint64_t* slots = buckets_[key & mask_].slots;
    int slot = 0;
    int worst = 1 << 30;
    Move kept_move = move;
    for (int i = 0; i < ENTRIES_PER_BUCKET; ++i) {
        Key stored_key = slots[i * 2];
        if (stored_key == 0) {
            slot = i;
            break;
        }
        uint64_t stored = slots[i * 2 + 1];
        int stored_depth = field(stored, DEPTH_SHIFT, DEPTH_MASK);
        int stored_age = field(stored, AGE_SHIFT, AGE_MASK);
        if (stored_key == key) {
            slot = i;
            // Never lose a known good move to a search that did not find one.
            if (kept_move == 0) kept_move = unpack_move(field(stored, MOVE_SHIFT, MOVE_MASK));
            // Depth-preferred: within one search, shallower work never displaces deeper
            // work for the same position, whatever its bound.
            if (depth < stored_depth && stored_age == (age & AGE_MASK)) return;
            break;
        }
        // Prefer to evict shallow work, and work from an older search more readily still.
        int value = stored_depth - 2 * ((age - stored_age) & AGE_MASK);
        if (value < worst) {
            worst = value;
            slot = i;
        }
    }
    slots[slot * 2] = key;
    slots[slot * 2 + 1] = pack_entry(adjusted, kept_move, depth, bound, static_eval, age);
}

TTProbe TranspositionTable::probe(Key key, int ply) const {
    const uint64_t* slots = buckets_[key & mask_].slots;
    for (int i = 0; i < ENTRIES_PER_BUCKET; ++i) {
        if (slots[i * 2] != key) continue;
        uint64_t stored = slots[i * 2 + 1];
        if (stored == 0) continue;
        int score = field(stored, SCORE_SHIFT, 0xFFFF) - static_cast<int>(SCORE_BIAS);
        if (score >= MATE_IN_MAX)
            score -= ply;
        else if (score <= -MATE_IN_MAX)
            score += ply;
        return {true,
                score,
                unpack_move(field(stored, MOVE_SHIFT, MOVE_MASK)),
                field(stored, DEPTH_SHIFT, DEPTH_MASK),
                field(stored, BOUND_SHIFT, BOUND_MASK),
                field(stored, STATIC_SHIFT, STATIC_MASK) - static_cast<int>(SCORE_BIAS)};
    }
    return {false, 0, 0, 0, BOUND_NONE, 0};
}
