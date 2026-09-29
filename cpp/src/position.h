// Position state, FEN, copy-make and attack detection.
//
// A Position is the Python engine's state row and mailbox in one struct. It lives in a
// preallocated per-ply stack so the search never allocates, and moves are made by copying
// into the next ply rather than unmaking. The mailbox carries only the piece type; colour
// is read from the colour bitboards.

#pragma once

#include <string>

#include "bitboard.h"
#include "types.h"

// Values SEE uses. Deliberately separate from the evaluation: SEE is about whether an
// exchange wins material, not about how the position should be judged.
inline constexpr int SEE_VALUE[6] = {100, 320, 330, 500, 900, 20000};

inline constexpr const char* START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct Position {
    Bitboard pieces[6];
    Bitboard colours[2];
    Key key;
    int stm;
    int castle;
    // The en passant target square as the FEN or the last double push gave it, or
    // NO_SQUARE. The key only includes it when a capture there is actually possible.
    int ep;
    int halfmove;
    int fullmove;
    int8_t mail[64];

    Bitboard occupied() const { return colours[WHITE] | colours[BLACK]; }
};

// Parses a FEN, or returns false and leaves `pos` unspecified. Castling rights without the
// king and rook on their home squares are dropped, as python-chess does.
bool from_fen(Position& pos, const std::string& fen);
std::string to_fen(const Position& pos);

// Recompute the Zobrist key from scratch. The reference for incremental updates.
Key full_key(const Position& pos);

// True when an enemy pawn could actually make the en passant capture. Hashing the square
// unconditionally makes otherwise identical positions hash differently.
bool ep_is_capturable(const Position& pos, int ep_square);

inline int king_square(const Position& pos, int colour) {
    return lsb(pos.pieces[KING] & pos.colours[colour]);
}

bool attacked(const Position& pos, int square, int by);

inline bool in_check(const Position& pos) {
    return attacked(pos, king_square(pos, pos.stm), pos.stm ^ 1);
}

// Did the side that just moved leave its own king attacked?
inline bool legal_after(const Position& after, int mover) {
    return !attacked(after, king_square(after, mover), mover ^ 1);
}

void make(const Position& src, Position& dst, Move move);
void make_null(const Position& src, Position& dst);

Bitboard attackers_to(const Position& pos, int square, Bitboard occ);

// Static exchange evaluation: the material outcome of the capture sequence on `to`.
int see(const Position& pos, Move move);

// Null move is unsafe without this: a side with only pawns can be in zugzwang.
inline bool has_non_pawn_material(const Position& pos, int colour) {
    return (pos.colours[colour] &
            (pos.pieces[KNIGHT] | pos.pieces[BISHOP] | pos.pieces[ROOK] | pos.pieces[QUEEN])) != 0;
}

// King versus king, or king and one minor versus king.
inline bool insufficient_material(const Position& pos) {
    if (pos.pieces[PAWN] | pos.pieces[ROOK] | pos.pieces[QUEEN]) return false;
    return popcount(pos.pieces[KNIGHT] | pos.pieces[BISHOP]) <= 1;
}

// The same position for game-continuity purposes: pieces, side to move, castling rights and
// an en passant square only where a capture is possible. The last is why this compares
// keys rather than raw fields: a double push nobody can capture sets `ep` but changes
// nothing about the position, and a FEN written afterwards omits the square.
inline bool same_position(const Position& a, const Position& b) {
    for (int i = 0; i < 6; ++i)
        if (a.pieces[i] != b.pieces[i]) return false;
    return a.colours[WHITE] == b.colours[WHITE] && a.colours[BLACK] == b.colours[BLACK] &&
           a.stm == b.stm && a.castle == b.castle && a.key == b.key;
}

std::string square_name(int square);
