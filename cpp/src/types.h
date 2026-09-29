// Shared vocabulary: bitboards, pieces, the move codec and the stack sizes.
//
// A move is an int32; the fields are in move_from and the functions beside it.

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

// Division that floors, as Python's `//` does. C++ division truncates toward zero, and the
// two disagree on every negative odd quotient; the search and evaluation use this one, and
// changing it changes every search.
constexpr int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
