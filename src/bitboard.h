// Bit primitives, attack tables and Zobrist keys. No position logic lives here.

#pragma once

#include "types.h"

#include "zobrist.inc"

inline int popcount(Bitboard b) { return __builtin_popcountll(b); }

// Index of the lowest set bit. Undefined for zero, so callers must guard.
inline int lsb(Bitboard b) { return __builtin_ctzll(b); }

inline int pop_lsb(Bitboard& b) {
    int square = lsb(b);
    b &= b - 1;
    return square;
}

struct Magic {
    Bitboard mask;
    Bitboard magic;
    const Bitboard* table;
    int shift;
};

extern Bitboard KNIGHT_ATT[64];
extern Bitboard KING_ATT[64];
// PAWN_ATT[colour][sq] is where a pawn of that colour on sq attacks.
extern Bitboard PAWN_ATT[2][64];
extern Magic ROOK_MAGICS[64];
extern Magic BISHOP_MAGICS[64];
// Castling rights survive a move unless its origin or destination is a king or rook home
// square. Masking on the destination is what handles a rook being captured where it stands.
extern int CASTLE_MASK[64];

inline Bitboard rook_attacks(int square, Bitboard occ) {
    const Magic& m = ROOK_MAGICS[square];
    return m.table[((occ & m.mask) * m.magic) >> m.shift];
}

inline Bitboard bishop_attacks(int square, Bitboard occ) {
    const Magic& m = BISHOP_MAGICS[square];
    return m.table[((occ & m.mask) * m.magic) >> m.shift];
}

inline Bitboard queen_attacks(int square, Bitboard occ) {
    return rook_attacks(square, occ) | bishop_attacks(square, occ);
}

// Builds every table. Call once at startup, before anything else.
void init_bitboards();
