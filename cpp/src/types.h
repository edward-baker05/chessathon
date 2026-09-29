// Shared vocabulary: bitboards, pieces, the move codec and the stack sizes.
//
// The encodings are the Python engine's, bit for bit. A move is the same int32 in both,
// which is what lets a fixed-node search here be checked against search.py node for node.

#pragma once

#include <cstdint>

using Bitboard = uint64_t;
using Key = uint64_t;

// from | to << 6 | promo << 12 | flag << 15. Zero is "no move": a1a1 is never generated.
using Move = int32_t;

enum Piece : int { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
constexpr int NO_PIECE = -1;

enum Colour : int { WHITE, BLACK };

// Castling rights bits.
constexpr int WHITE_KINGSIDE = 1;
constexpr int WHITE_QUEENSIDE = 2;
constexpr int BLACK_KINGSIDE = 4;
constexpr int BLACK_QUEENSIDE = 8;
constexpr int ALL_CASTLING = 15;

constexpr int FLAG_NORMAL = 0;
constexpr int FLAG_EP = 1;
constexpr int FLAG_CASTLE = 2;
constexpr int FLAG_PROMO = 3;

constexpr int NO_SQUARE = -1;

constexpr int STACK_PLIES = 256;
constexpr int MAX_MOVES = 256;

constexpr Move make_move(int from, int to, int promo = 0, int flag = FLAG_NORMAL) {
    return from | (to << 6) | (promo << 12) | (flag << 15);
}
constexpr int move_from(Move m) { return m & 63; }
constexpr int move_to(Move m) { return (m >> 6) & 63; }
constexpr int move_promo(Move m) { return (m >> 12) & 7; }
constexpr int move_flag(Move m) { return (m >> 15) & 3; }

constexpr Bitboard square_bit(int square) { return Bitboard{1} << square; }

// Python's `//`, which floors. C++ division truncates toward zero, and the two disagree on
// every negative odd quotient: a history halving or an evaluation scaled that way is off by
// one, which is enough to make a fixed-node search diverge from the Python engine's.
constexpr int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
