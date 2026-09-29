#include "nnue.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifndef NNUE_BLOB
#error "NNUE_BLOB must name the network blob to embed; build through cpp/Makefile"
#endif

// The blob tools/export_cpp.py writes from weights/net.npz, linked straight into .rodata.
asm(".section .rodata\n"
    ".balign 64\n"
    ".globl nnue_blob_start\n"
    "nnue_blob_start:\n"
    ".incbin \"" NNUE_BLOB "\"\n"
    ".globl nnue_blob_end\n"
    "nnue_blob_end:\n"
    ".previous\n");
extern "C" const unsigned char nnue_blob_start[];
extern "C" const unsigned char nnue_blob_end[];

namespace nnue {

namespace {

constexpr uint32_t BLOB_VERSION = 1;

// One extra all-zero row past the real features. Pointing an unused slot at it lets every
// move share one "subtract two, add two" update instead of branching per move flag.
constexpr int ZERO_FEATURE = NUM_FEATURES;

struct Network {
    alignas(64) int16_t ft_weight[NUM_FEATURES + 1][L1];
    alignas(64) int16_t ft_bias[L1];
    alignas(64) int16_t out_weight[OUTPUT_BUCKETS][2 * L1];
    int32_t out_bias[OUTPUT_BUCKETS];
    int qa, qb, scale;
    // Buckets are chosen by piece count, 2 to 32, mapped onto 0 to OUTPUT_BUCKETS - 1.
    int bucket_divisor;
    // Whether every activation times output weight fits int16, which is what makes the
    // madd form of the output layer exact. True of the shipped net; checked, not assumed.
    bool madd_exact;
};

Network net;

// Index of one piece-on-square feature, as seen from one side. From black's side the board
// is flipped vertically and the colours are swapped, so both halves share weights.
inline int feature(int perspective, int colour, int piece, int square) {
    return ((colour ^ perspective) * 6 + piece) * 64 + (square ^ (perspective * 56));
}

inline void move_one(int16_t* dst, const int16_t* src, const int16_t* sub, const int16_t* add) {
    for (int i = 0; i < L1; ++i) dst[i] = static_cast<int16_t>(src[i] - sub[i] + add[i]);
}

inline void move_two(int16_t* dst, const int16_t* src, const int16_t* sub0, const int16_t* sub1,
                     const int16_t* add0, const int16_t* add1) {
    for (int i = 0; i < L1; ++i)
        dst[i] = static_cast<int16_t>(src[i] - sub0[i] - sub1[i] + add0[i] + add1[i]);
}

// Squared clipped ReLU against one output row, side to move first. The sum wraps at 32 bits
// exactly as nnue.py's int32 does; tools/quantise.py proves the shipped weights never wrap.
int32_t dot(const int16_t* us, const int16_t* them, const int16_t* weights) {
#if defined(__AVX2__)
    if (net.madd_exact) {
        // mullo(v, w) is exact because v * w fits int16, and madd(v * w, v) then sums
        // v * v * w in pairs into int32 lanes: the same integer, a quarter of the work.
        const __m256i zero = _mm256_setzero_si256();
        const __m256i ceiling = _mm256_set1_epi16(static_cast<int16_t>(net.qa));
        __m256i sum = zero;
        for (int half = 0; half < 2; ++half) {
            const int16_t* acc = half ? them : us;
            const int16_t* row = weights + half * L1;
            for (int i = 0; i < L1; i += 16) {
                __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
                v = _mm256_min_epi16(_mm256_max_epi16(v, zero), ceiling);
                __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i));
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(v, w), v));
            }
        }
        __m128i folded = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        folded = _mm_add_epi32(folded, _mm_shuffle_epi32(folded, 0x4E));
        folded = _mm_add_epi32(folded, _mm_shuffle_epi32(folded, 0xB1));
        return _mm_cvtsi128_si32(folded);
    }
#endif
    // Unsigned, so a wrap is defined behaviour rather than undefined; the bits are the same.
    uint32_t total = 0;
    for (int i = 0; i < L1; ++i) {
        int32_t v = std::clamp<int16_t>(us[i], 0, static_cast<int16_t>(net.qa));
        total += static_cast<uint32_t>(v * v * weights[i]);
    }
    for (int i = 0; i < L1; ++i) {
        int32_t v = std::clamp<int16_t>(them[i], 0, static_cast<int16_t>(net.qa));
        total += static_cast<uint32_t>(v * v * weights[L1 + i]);
    }
    return static_cast<int32_t>(total);
}

template <typename T>
const unsigned char* read(const unsigned char* at, T* out, size_t count) {
    std::memcpy(out, at, sizeof(T) * count);
    return at + sizeof(T) * count;
}

}  // namespace

bool load(std::string& error) {
    const unsigned char* at = nnue_blob_start;
    size_t size = static_cast<size_t>(nnue_blob_end - nnue_blob_start);
    if (size < 28 || std::memcmp(at, "NNUE", 4) != 0) {
        error = "the embedded network is not an NNUE blob";
        return false;
    }
    uint32_t header[3];
    int32_t scales[3];
    at = read(at + 4, header, 3);
    at = read(at, scales, 3);
    if (header[0] != BLOB_VERSION) {
        error = "the embedded network is blob version " + std::to_string(header[0]);
        return false;
    }
    if (header[1] != static_cast<uint32_t>(L1) || header[2] != static_cast<uint32_t>(OUTPUT_BUCKETS)) {
        error = "the embedded network is L1 " + std::to_string(header[1]) + " with " +
                std::to_string(header[2]) + " buckets; rebuild with NNUE_L1 and NNUE_BUCKETS set";
        return false;
    }
    size_t expected = 28 + sizeof(int16_t) * (NUM_FEATURES * L1 + L1 + OUTPUT_BUCKETS * 2 * L1) +
                      sizeof(int32_t) * OUTPUT_BUCKETS;
    if (size != expected) {
        error = "the embedded network is " + std::to_string(size) + " bytes, expected " +
                std::to_string(expected);
        return false;
    }
    at = read(at, &net.ft_weight[0][0], static_cast<size_t>(NUM_FEATURES) * L1);
    std::memset(net.ft_weight[ZERO_FEATURE], 0, sizeof net.ft_weight[ZERO_FEATURE]);
    at = read(at, net.ft_bias, L1);
    at = read(at, &net.out_weight[0][0], static_cast<size_t>(OUTPUT_BUCKETS) * 2 * L1);
    read(at, net.out_bias, OUTPUT_BUCKETS);
    net.qa = scales[0];
    net.qb = scales[1];
    net.scale = scales[2];
    net.bucket_divisor = (32 - 2) / OUTPUT_BUCKETS + 1;

    int largest = 0;
    for (const auto& row : net.out_weight)
        for (int16_t w : row) largest = std::max(largest, std::abs(static_cast<int>(w)));
    net.madd_exact = net.qa > 0 && net.qa <= 32767 && largest * net.qa <= 32767;
    return true;
}

void refresh(Accumulator& acc, const Position& pos) {
    for (int p = 0; p < 2; ++p) std::memcpy(acc.values[p], net.ft_bias, sizeof net.ft_bias);
    for (int square = 0; square < 64; ++square) {
        int piece = pos.mail[square];
        if (piece < 0) continue;
        int colour = (pos.colours[BLACK] & square_bit(square)) ? BLACK : WHITE;
        for (int p = 0; p < 2; ++p) {
            const int16_t* row = net.ft_weight[feature(p, colour, piece, square)];
            for (int i = 0; i < L1; ++i)
                acc.values[p][i] = static_cast<int16_t>(acc.values[p][i] + row[i]);
        }
    }
}

void apply(const Accumulator& src, Accumulator& dst, const Position& before, Move move) {
    // At most two features leave and two arrive. A capture removes the mover's old square
    // and the victim; castling removes the king's and the rook's old squares and adds both
    // new ones; a promotion adds a different piece from the one that left. Unused slots
    // point at the all-zero row so the arithmetic stays uniform.
    int from = move_from(move);
    int to = move_to(move);
    int flag = move_flag(move);
    int us = before.stm;
    int them = us ^ 1;
    int moved = before.mail[from];
    int arriving = flag == FLAG_PROMO ? move_promo(move) : moved;

    bool simple = flag == FLAG_NORMAL && before.mail[to] < 0;
    for (int p = 0; p < 2; ++p) {
        const int16_t* sub0 = net.ft_weight[feature(p, us, moved, from)];
        const int16_t* add0 = net.ft_weight[feature(p, us, arriving, to)];
        if (simple) {
            move_one(dst.values[p], src.values[p], sub0, add0);
            continue;
        }
        int sub1 = ZERO_FEATURE;
        int add1 = ZERO_FEATURE;
        if (flag == FLAG_EP) {
            // The captured pawn sits beside the target square, not behind it.
            sub1 = feature(p, them, PAWN, us == BLACK ? to + 8 : to - 8);
        } else if (flag == FLAG_CASTLE) {
            // The king's move is already in slot zero, so this is the rook.
            int rook_from, rook_to;
            if (to == 6) {
                rook_from = 7, rook_to = 5;
            } else if (to == 2) {
                rook_from = 0, rook_to = 3;
            } else if (to == 62) {
                rook_from = 63, rook_to = 61;
            } else {
                rook_from = 56, rook_to = 59;
            }
            sub1 = feature(p, us, ROOK, rook_from);
            add1 = feature(p, us, ROOK, rook_to);
        } else if (before.mail[to] >= 0) {
            sub1 = feature(p, them, before.mail[to], to);
        }
        move_two(dst.values[p], src.values[p], sub0, net.ft_weight[sub1], add0,
                 net.ft_weight[add1]);
    }
}

int evaluate(const Accumulator& acc, const Position& pos) {
    int stm = pos.stm;
    int bucket = (popcount(pos.occupied()) - 2) / net.bucket_divisor;
    int64_t total = dot(acc.values[stm], acc.values[stm ^ 1], net.out_weight[bucket]);
    // SCReLU squares the input scale, hence the extra divide by QA.
    int64_t scaled = floor_div(
        (floor_div(total, net.qa) + net.out_bias[bucket]) * net.scale,
        static_cast<int64_t>(net.qa) * net.qb);
    return static_cast<int>(std::clamp<int64_t>(scaled, -EVAL_LIMIT, EVAL_LIMIT));
}

}  // namespace nnue
