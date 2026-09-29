#include "bitboard.h"

#include <algorithm>
#include <vector>

Bitboard KNIGHT_ATT[64];
Bitboard KING_ATT[64];
Bitboard PAWN_ATT[2][64];
Magic ROOK_MAGICS[64];
Magic BISHOP_MAGICS[64];
int CASTLE_MASK[64];

namespace {

// Rook tables hold 102400 entries across all squares, bishop tables 5248.
Bitboard ROOK_TABLE[102400];
Bitboard BISHOP_TABLE[5248];

// Squares reachable along one direction, including the first blocker.
Bitboard ray_attacks(int square, int dr, int df, Bitboard occ) {
    Bitboard att = 0;
    int r = square / 8 + dr;
    int f = square % 8 + df;
    while (r >= 0 && r < 8 && f >= 0 && f < 8) {
        Bitboard bit = square_bit(r * 8 + f);
        att |= bit;
        if (occ & bit) break;
        r += dr;
        f += df;
    }
    return att;
}

// Relevant occupancy along one direction: everything but the final square. A piece on the
// edge cannot block anything beyond itself, so excluding it halves the table size.
Bitboard ray_mask(int square, int dr, int df) {
    Bitboard m = 0;
    int r = square / 8 + dr;
    int f = square % 8 + df;
    while (r >= 0 && r <= 7 && f >= 0 && f <= 7) {
        int nr = r + dr;
        int nf = f + df;
        if (nr < 0 || nr > 7 || nf < 0 || nf > 7) break;
        m |= square_bit(r * 8 + f);
        r = nr;
        f = nf;
    }
    return m;
}

constexpr int ROOK_DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
constexpr int BISHOP_DIRS[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

Bitboard slider_attacks(int square, Bitboard occ, bool rook) {
    const auto& dirs = rook ? ROOK_DIRS : BISHOP_DIRS;
    Bitboard att = 0;
    for (const auto& d : dirs) att |= ray_attacks(square, d[0], d[1], occ);
    return att;
}

Bitboard slider_mask(int square, bool rook) {
    const auto& dirs = rook ? ROOK_DIRS : BISHOP_DIRS;
    Bitboard m = 0;
    for (const auto& d : dirs) m |= ray_mask(square, d[0], d[1]);
    return m;
}

// Seeded PRNG, so the magics found are the same on every machine and every run.
uint64_t splitmix(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Fancy magic bitboards, with the magic constants found here rather than copied in. For
// each square: enumerate every occupancy subset of the relevant mask, compute the true
// attack set for each, then look for a multiplier that maps all of them into a table
// without a contradictory collision.
void build_magics(Magic* magics, Bitboard* table, bool rook, uint64_t seed) {
    uint64_t rng = seed;
    size_t offset = 0;
    std::vector<Bitboard> occs, atts, used;
    std::vector<char> seen;
    for (int square = 0; square < 64; ++square) {
        Magic& m = magics[square];
        m.mask = slider_mask(square, rook);
        int bits = popcount(m.mask);
        m.shift = 64 - bits;
        size_t base = offset;
        m.table = table + base;
        size_t n = size_t{1} << bits;
        offset += n;

        occs.assign(n, 0);
        atts.assign(n, 0);
        used.assign(n, 0);
        seen.assign(n, 0);
        // Carry-rippler walk over every subset of the mask.
        Bitboard sub = 0;
        for (size_t i = 0; i < n; ++i) {
            occs[i] = sub;
            atts[i] = slider_attacks(square, sub, rook);
            sub = (sub - m.mask) & m.mask;
        }

        while (true) {
            uint64_t candidate = splitmix(rng) & splitmix(rng) & splitmix(rng);
            // A magic needs enough high bits set to spread the index across the table.
            if (popcount((m.mask * candidate) & 0xFF00000000000000ull) < 6) continue;
            std::fill(seen.begin(), seen.end(), 0);
            bool ok = true;
            for (size_t i = 0; i < n; ++i) {
                size_t index = (occs[i] * candidate) >> m.shift;
                if (!seen[index]) {
                    seen[index] = 1;
                    used[index] = atts[i];
                } else if (used[index] != atts[i]) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                m.magic = candidate;
                for (size_t i = 0; i < n; ++i) table[base + i] = used[i];
                break;
            }
        }
    }
}

void build_leapers() {
    constexpr int KNIGHT_STEPS[8][2] = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1},
                                        {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};
    for (int square = 0; square < 64; ++square) {
        int r = square / 8;
        int f = square % 8;
        KNIGHT_ATT[square] = KING_ATT[square] = 0;
        PAWN_ATT[WHITE][square] = PAWN_ATT[BLACK][square] = 0;
        for (const auto& step : KNIGHT_STEPS) {
            int nr = r + step[0];
            int nf = f + step[1];
            if (nr >= 0 && nr <= 7 && nf >= 0 && nf <= 7) KNIGHT_ATT[square] |= square_bit(nr * 8 + nf);
        }
        for (int dr = -1; dr <= 1; ++dr) {
            for (int df = -1; df <= 1; ++df) {
                if (dr == 0 && df == 0) continue;
                int nr = r + dr;
                int nf = f + df;
                if (nr >= 0 && nr <= 7 && nf >= 0 && nf <= 7) KING_ATT[square] |= square_bit(nr * 8 + nf);
            }
        }
        for (int df : {-1, 1}) {
            int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            if (r < 7) PAWN_ATT[WHITE][square] |= square_bit((r + 1) * 8 + nf);
            if (r > 0) PAWN_ATT[BLACK][square] |= square_bit((r - 1) * 8 + nf);
        }
    }
}

}  // namespace

void init_bitboards() {
    build_leapers();
    build_magics(ROOK_MAGICS, ROOK_TABLE, true, 0x1234567);
    build_magics(BISHOP_MAGICS, BISHOP_TABLE, false, 0x89ABCDEF);

    for (int& mask : CASTLE_MASK) mask = ALL_CASTLING;
    CASTLE_MASK[0] = ALL_CASTLING ^ WHITE_QUEENSIDE;                     // a1
    CASTLE_MASK[4] = ALL_CASTLING ^ (WHITE_KINGSIDE | WHITE_QUEENSIDE);  // e1
    CASTLE_MASK[7] = ALL_CASTLING ^ WHITE_KINGSIDE;                      // h1
    CASTLE_MASK[56] = ALL_CASTLING ^ BLACK_QUEENSIDE;                    // a8
    CASTLE_MASK[60] = ALL_CASTLING ^ (BLACK_KINGSIDE | BLACK_QUEENSIDE); // e8
    CASTLE_MASK[63] = ALL_CASTLING ^ BLACK_KINGSIDE;                     // h8
}
