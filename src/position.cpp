#include "position.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace {

constexpr const char* PIECE_CHARS = "pnbrqk";

int piece_from_char(char c) {
    const char* found = std::strchr(PIECE_CHARS, c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    return found && *found ? static_cast<int>(found - PIECE_CHARS) : NO_PIECE;
}

int parse_square(const std::string& text) {
    if (text.size() != 2 || text[0] < 'a' || text[0] > 'h' || text[1] < '1' || text[1] > '8')
        return NO_SQUARE;
    return (text[1] - '1') * 8 + (text[0] - 'a');
}

bool piece_on(const Position& pos, int colour, int piece, int square) {
    return (pos.pieces[piece] & pos.colours[colour] & square_bit(square)) != 0;
}

}  // namespace

std::string square_name(int square) {
    return {static_cast<char>('a' + square % 8), static_cast<char>('1' + square / 8)};
}

bool from_fen(Position& pos, const std::string& fen) {
    std::istringstream in(fen);
    std::string placement, side, castling = "-", ep = "-";
    int halfmove = 0, fullmove = 1;
    if (!(in >> placement >> side)) return false;
    in >> castling >> ep >> halfmove >> fullmove;

    std::memset(&pos, 0, sizeof pos);
    std::memset(pos.mail, NO_PIECE, sizeof pos.mail);
    int rank = 7, file = 0;
    for (char c : placement) {
        if (c == '/') {
            if (file != 8 || rank == 0) return false;
            --rank;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
        } else {
            int piece = piece_from_char(c);
            if (piece == NO_PIECE || file > 7) return false;
            int square = rank * 8 + file;
            int colour = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
            pos.pieces[piece] |= square_bit(square);
            pos.colours[colour] |= square_bit(square);
            pos.mail[square] = static_cast<int8_t>(piece);
            ++file;
        }
        if (file > 8) return false;
    }
    if (rank != 0 || file != 8) return false;
    if (popcount(pos.pieces[KING] & pos.colours[WHITE]) != 1 ||
        popcount(pos.pieces[KING] & pos.colours[BLACK]) != 1)
        return false;

    if (side != "w" && side != "b") return false;
    pos.stm = side == "w" ? WHITE : BLACK;

    for (char c : castling) {
        if (c == 'K' && piece_on(pos, WHITE, KING, 4) && piece_on(pos, WHITE, ROOK, 7))
            pos.castle |= WHITE_KINGSIDE;
        if (c == 'Q' && piece_on(pos, WHITE, KING, 4) && piece_on(pos, WHITE, ROOK, 0))
            pos.castle |= WHITE_QUEENSIDE;
        if (c == 'k' && piece_on(pos, BLACK, KING, 60) && piece_on(pos, BLACK, ROOK, 63))
            pos.castle |= BLACK_KINGSIDE;
        if (c == 'q' && piece_on(pos, BLACK, KING, 60) && piece_on(pos, BLACK, ROOK, 56))
            pos.castle |= BLACK_QUEENSIDE;
    }
    pos.ep = ep == "-" ? NO_SQUARE : parse_square(ep);
    pos.halfmove = halfmove;
    pos.fullmove = fullmove;
    pos.key = full_key(pos);
    return true;
}

std::string to_fen(const Position& pos) {
    std::string out;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int square = rank * 8 + file;
            int piece = pos.mail[square];
            if (piece < 0) {
                ++empty;
                continue;
            }
            if (empty) out += static_cast<char>('0' + empty);
            empty = 0;
            char symbol = PIECE_CHARS[piece];
            out += (pos.colours[WHITE] & square_bit(square)) ? static_cast<char>(symbol - 'a' + 'A')
                                                             : symbol;
        }
        if (empty) out += static_cast<char>('0' + empty);
        if (rank) out += '/';
    }
    out += pos.stm == WHITE ? " w " : " b ";
    std::string castling;
    if (pos.castle & WHITE_KINGSIDE) castling += 'K';
    if (pos.castle & WHITE_QUEENSIDE) castling += 'Q';
    if (pos.castle & BLACK_KINGSIDE) castling += 'k';
    if (pos.castle & BLACK_QUEENSIDE) castling += 'q';
    out += castling.empty() ? "-" : castling;
    out += ' ';
    out += pos.ep == NO_SQUARE ? "-" : square_name(pos.ep);
    out += ' ' + std::to_string(pos.halfmove) + ' ' + std::to_string(pos.fullmove);
    return out;
}

bool ep_is_capturable(const Position& pos, int ep_square) {
    // The squares a black pawn could capture from are the squares a white pawn on
    // ep_square would attack, so black to move reads PAWN_ATT[WHITE].
    Bitboard attackers = PAWN_ATT[pos.stm ^ 1][ep_square];
    return (attackers & pos.pieces[PAWN] & pos.colours[pos.stm]) != 0;
}

Key full_key(const Position& pos) {
    Key key = 0;
    for (int piece = 0; piece < 6; ++piece) {
        for (int colour = 0; colour < 2; ++colour) {
            Bitboard bits = pos.pieces[piece] & pos.colours[colour];
            while (bits) key ^= ZOBRIST_PIECE[colour][piece][pop_lsb(bits)];
        }
    }
    if (pos.stm == BLACK) key ^= ZOBRIST_STM;
    key ^= ZOBRIST_CASTLE[pos.castle];
    if (pos.ep != NO_SQUARE && ep_is_capturable(pos, pos.ep)) key ^= ZOBRIST_EP[pos.ep % 8];
    return key;
}

bool attacked(const Position& pos, int square, int by) {
    Bitboard occ = pos.occupied();
    Bitboard them = pos.colours[by];
    // PAWN_ATT[WHITE][sq] is where a white pawn on sq attacks, which is exactly the set of
    // squares a black pawn would have to stand on to attack sq. The index is inverted
    // relative to intuition, and getting it backwards is a rarely triggered bug.
    if (PAWN_ATT[by ^ 1][square] & pos.pieces[PAWN] & them) return true;
    if (KNIGHT_ATT[square] & pos.pieces[KNIGHT] & them) return true;
    if (KING_ATT[square] & pos.pieces[KING] & them) return true;
    if (bishop_attacks(square, occ) & (pos.pieces[BISHOP] | pos.pieces[QUEEN]) & them) return true;
    return (rook_attacks(square, occ) & (pos.pieces[ROOK] | pos.pieces[QUEEN]) & them) != 0;
}

void make(const Position& src, Position& dst, Move move) {
    dst = src;

    int from = move_from(move);
    int to = move_to(move);
    int promo = move_promo(move);
    int flag = move_flag(move);

    int us = src.stm;
    int them = us ^ 1;
    Bitboard from_bit = square_bit(from);
    Bitboard to_bit = square_bit(to);
    int moved = src.mail[from];

    dst.halfmove = src.halfmove + 1;

    // Incremental Zobrist. A full recompute costs a scan of all twelve bitboards per node.
    Key key = src.key ^ ZOBRIST_STM;
    if (src.ep != NO_SQUARE && ep_is_capturable(src, src.ep)) key ^= ZOBRIST_EP[src.ep % 8];

    if (flag == FLAG_EP) {
        // The captured pawn sits beside the target square, not behind it: a black pawn
        // capturing onto e3 removes the white pawn on e4.
        int capture_square = us == BLACK ? to + 8 : to - 8;
        Bitboard capture_bit = square_bit(capture_square);
        dst.pieces[PAWN] &= ~capture_bit;
        dst.colours[them] &= ~capture_bit;
        dst.mail[capture_square] = NO_PIECE;
        dst.pieces[PAWN] = (dst.pieces[PAWN] & ~from_bit) | to_bit;
        dst.colours[us] = (dst.colours[us] & ~from_bit) | to_bit;
        dst.mail[from] = NO_PIECE;
        dst.mail[to] = PAWN;
        dst.halfmove = 0;
        key ^= ZOBRIST_PIECE[them][PAWN][capture_square];
        key ^= ZOBRIST_PIECE[us][PAWN][from];
        key ^= ZOBRIST_PIECE[us][PAWN][to];
    } else {
        int captured = src.mail[to];
        if (captured >= 0) {
            dst.pieces[captured] &= ~to_bit;
            dst.colours[them] &= ~to_bit;
            dst.halfmove = 0;
            key ^= ZOBRIST_PIECE[them][captured][to];
        }
        if (moved == PAWN) dst.halfmove = 0;

        dst.pieces[moved] &= ~from_bit;
        dst.colours[us] = (dst.colours[us] & ~from_bit) | to_bit;
        dst.mail[from] = NO_PIECE;
        key ^= ZOBRIST_PIECE[us][moved][from];
        if (flag == FLAG_PROMO) {
            dst.pieces[promo] |= to_bit;
            dst.mail[to] = static_cast<int8_t>(promo);
            key ^= ZOBRIST_PIECE[us][promo][to];
        } else {
            dst.pieces[moved] |= to_bit;
            dst.mail[to] = static_cast<int8_t>(moved);
            key ^= ZOBRIST_PIECE[us][moved][to];
        }

        if (flag == FLAG_CASTLE) {
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
            Bitboard rook_from_bit = square_bit(rook_from);
            Bitboard rook_to_bit = square_bit(rook_to);
            dst.pieces[ROOK] = (dst.pieces[ROOK] & ~rook_from_bit) | rook_to_bit;
            dst.colours[us] = (dst.colours[us] & ~rook_from_bit) | rook_to_bit;
            dst.mail[rook_from] = NO_PIECE;
            dst.mail[rook_to] = ROOK;
            key ^= ZOBRIST_PIECE[us][ROOK][rook_from];
            key ^= ZOBRIST_PIECE[us][ROOK][rook_to];
        }
    }

    dst.ep = (moved == PAWN && (to - from == 16 || from - to == 16)) ? (from + to) / 2 : NO_SQUARE;

    int new_rights = src.castle & CASTLE_MASK[from] & CASTLE_MASK[to];
    dst.castle = new_rights;
    key ^= ZOBRIST_CASTLE[src.castle] ^ ZOBRIST_CASTLE[new_rights];

    dst.stm = them;
    if (us == BLACK) dst.fullmove = src.fullmove + 1;

    // The new en passant term needs the finished position: ep_is_capturable reads the
    // updated pawns and the flipped side to move.
    if (dst.ep != NO_SQUARE && ep_is_capturable(dst, dst.ep)) key ^= ZOBRIST_EP[dst.ep % 8];
    dst.key = key;
}

void make_null(const Position& src, Position& dst) {
    // The halfmove clock is reset rather than carried. A null move breaks the alternating
    // parity that repetition scanning relies on, and zeroing the clock bounds that scan to
    // nothing inside the null subtree. The cost is that a fifty-move draw deep inside a
    // null subtree can go unnoticed, which is conservative: it never invents a draw.
    dst = src;
    Key key = src.key ^ ZOBRIST_STM;
    if (src.ep != NO_SQUARE && ep_is_capturable(src, src.ep)) key ^= ZOBRIST_EP[src.ep % 8];
    dst.ep = NO_SQUARE;
    dst.halfmove = 0;
    dst.stm = src.stm ^ 1;
    dst.key = key;
}

Bitboard attackers_to(const Position& pos, int square, Bitboard occ) {
    // Taking `occ` as an argument is what makes x-ray detection work: SEE clears each
    // consumed piece from `occ` and calls this again, which reveals sliders behind it.
    return ((PAWN_ATT[WHITE][square] & pos.pieces[PAWN] & pos.colours[BLACK]) |
            (PAWN_ATT[BLACK][square] & pos.pieces[PAWN] & pos.colours[WHITE]) |
            (KNIGHT_ATT[square] & pos.pieces[KNIGHT]) | (KING_ATT[square] & pos.pieces[KING]) |
            (bishop_attacks(square, occ) & (pos.pieces[BISHOP] | pos.pieces[QUEEN])) |
            (rook_attacks(square, occ) & (pos.pieces[ROOK] | pos.pieces[QUEEN]))) &
           occ;
}

int see(const Position& pos, Move move) {
    // The swap algorithm. Build the list of gains assuming both sides always recapture
    // with their least valuable attacker, then walk it backwards applying the option not
    // to continue.
    int from = move_from(move);
    int to = move_to(move);
    int flag = move_flag(move);

    // Castling never captures, and en passant is not worth special-casing here.
    if (flag == FLAG_CASTLE) return 0;

    int captured = pos.mail[to];
    int gain[32] = {};
    gain[0] = captured >= 0 ? SEE_VALUE[captured] : 0;
    if (flag == FLAG_EP) gain[0] = SEE_VALUE[PAWN];

    Bitboard occ = pos.occupied() & ~square_bit(from);
    if (flag == FLAG_EP) occ &= ~square_bit(pos.stm == BLACK ? to + 8 : to - 8);

    int on_square = pos.mail[from];
    int side = pos.stm;
    Bitboard attacks = attackers_to(pos, to, occ);

    int depth = 0;
    while (true) {
        side ^= 1;
        Bitboard mine = attacks & pos.colours[side] & occ;
        if (!mine) break;

        // Recapture with the least valuable attacker available.
        int piece = NO_PIECE;
        for (int candidate = 0; candidate < 6; ++candidate) {
            if (mine & pos.pieces[candidate]) {
                piece = candidate;
                break;
            }
        }
        if (piece < 0) break;

        ++depth;
        if (depth >= 31) break;
        gain[depth] = SEE_VALUE[on_square] - gain[depth - 1];

        occ &= ~square_bit(lsb(mine & pos.pieces[piece]));
        on_square = piece;
        // Re-derive attackers so sliders behind the piece just consumed are included.
        attacks = attackers_to(pos, to, occ);
    }

    // Walk back: at each point the side to move could simply decline the exchange.
    while (depth > 0) {
        gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
        --depth;
    }
    return gain[0];
}
