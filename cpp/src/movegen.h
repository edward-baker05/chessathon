// Pseudo-legal move generation, filtered afterwards by making each move and testing whether
// the mover left its king attacked. The order moves come out in is the Python engine's,
// because move ordering breaks ties by that order.

#pragma once

#include <string>

#include "position.h"

// Every pseudo-legal move. Returns how many were written.
int generate(const Position& pos, Move* moves);

// Captures and queen promotions only, for quiescence search.
int generate_captures(const Position& pos, Move* moves);

// Strictly legal moves, for the UCI boundary. Not used inside the search.
int generate_legal(const Position& pos, Move* moves);

uint64_t perft(const Position& pos, int depth);

std::string move_to_uci(Move move);

// The legal move in `pos` a UCI string names, or 0.
Move parse_uci_move(const Position& pos, const std::string& text);
