// Search: iterative deepening, PVS, quiescence, move ordering and time control. With one
// thread a fixed-node search is deterministic; with more (see set_threads) the threads race
// through the shared table, and no two searches are alike.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "position.h"
#include "tt.h"

namespace search {

constexpr int MAX_DEPTH = 127;
constexpr int MAX_THREADS = 256;

struct Limits {
    // Our clock and increment. Without a clock the search runs until a node or depth
    // limit, or a stop.
    bool has_clock = false;
    int64_t time_left_ms = 0;
    int64_t increment_ms = 0;
    // Search exactly this long, when positive, whatever the clock says.
    int64_t movetime_ms = 0;
    int64_t node_limit = 0;
    int max_depth = MAX_DEPTH;
    // Only these root moves, when not empty. Moves not legal at the root are ignored.
    std::vector<Move> searchmoves;
};

// The last completed iteration of the last search, from the thread whose move it played,
// with every thread's nodes.
struct Report {
    Move best;
    int depth;
    int score;
    int seldepth;
    int64_t nodes;
    int64_t elapsed_ms;
};

// Allocates the search state for one thread. Call once, after init_bitboards and nnue::load.
void init();

// Search with this many threads (Lazy SMP), clamped to 1..MAX_THREADS. Threads added start
// with empty history tables; not to be called during a search.
void set_threads(int count);

TranspositionTable& table();

// Set from another thread to end the current search at the next clock check.
extern std::atomic<bool> stop_requested;

// While set, the search is on the opponent's time: no time limit applies. Set it before
// starting a `go ponder` search; ponderhit() clears it.
extern std::atomic<bool> pondering;

// The opponent played the expected move. The limits the search was started with apply from
// now, as if it had been started now.
void ponderhit();

// Positions already played in this game, oldest first, ending with the root.
void set_game_history(const std::vector<Key>& keys);

// Forgets everything learned about the last game except the transposition table.
void clear_tables();

// Best move for `root`, or 0 when it has none. `on_iteration` hears each completed depth.
Move think(const Position& root, const Limits& limits,
           const std::function<void(const Report&)>& on_iteration = {});

Report last_report();

// The last search's best move and the replies it expects, walked out of the table.
std::vector<Move> principal_variation(const Position& root);

// A report as a UCI `info` line.
std::string uci_info(const Position& root, const Report& report);

}  // namespace search
