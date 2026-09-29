// The UCI front end, and the game tracking agent.py does for the Python engine.
//
// The Python harness hands over a bare FEN with no game identity, so a position that is one
// legal move on from our last reply continues the game and anything else starts a new one,
// clearing what was learned. A `position ... moves ...` command carries the whole game and is
// used as the repetition history directly.
//
// Beyond UCI: `perft N`, `eval`, `d` and `bench [nodes]`. Arguments on the command line
// run as one command and exit, so `engine bench` works.

#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "movegen.h"
#include "nnue.h"
#include "search.h"

namespace {

std::mutex output_mutex;

void say(const std::string& line) {
    std::lock_guard lock(output_mutex);
    std::cout << line << '\n' << std::flush;
}

struct Game {
    Position root;
    // Every position of the game so far, ending with the root.
    std::vector<Key> history;
    // The position after our own last reply.
    std::optional<Position> last_reply;
};

Game game;
std::thread worker;

void wait_for_search() {
    if (worker.joinable()) worker.join();
}

void new_game() {
    search::table().clear();
    search::clear_tables();
    game.history.clear();
    game.last_reply.reset();
}

// Is `pos` one legal move on from `from`?
bool follows(const Position& from, const Position& pos) {
    Move moves[MAX_MOVES];
    int count = generate_legal(from, moves);
    Position after;
    for (int i = 0; i < count; ++i) {
        make(from, after, moves[i]);
        if (same_position(after, pos)) return true;
    }
    return false;
}

void handle_position(std::istringstream& in) {
    std::string token, fen;
    in >> token;
    if (token == "startpos") {
        fen = START_FEN;
        in >> token;
    } else if (token == "fen") {
        while (in >> token && token != "moves") fen += (fen.empty() ? "" : " ") + token;
    } else {
        say("info string unrecognised position command");
        return;
    }
    Position pos;
    if (!from_fen(pos, fen)) {
        say("info string invalid fen: " + fen);
        return;
    }
    std::vector<Key> keys{pos.key};
    bool listed = false;
    if (token == "moves") {
        while (in >> token) {
            Move move = parse_uci_move(pos, token);
            if (move == 0) {
                say("info string illegal move " + token + " in " + to_fen(pos));
                break;
            }
            Position next;
            make(pos, next, move);
            pos = next;
            keys.push_back(pos.key);
            listed = true;
        }
    }

    if (!game.last_reply || !follows(*game.last_reply, pos)) {
        new_game();
        say("info string new game");
    }
    if (listed)
        game.history = keys;
    else
        game.history.push_back(pos.key);
    game.root = pos;
}

void handle_go(std::istringstream& in) {
    search::Limits limits;
    bool infinite = false;
    std::string token;
    int64_t wtime = -1, btime = -1, winc = 0, binc = 0;
    while (in >> token) {
        if (token == "infinite") {
            infinite = true;
            continue;
        }
        int64_t value = 0;
        if (!(in >> value)) break;
        if (token == "wtime") wtime = value;
        else if (token == "btime") btime = value;
        else if (token == "winc") winc = value;
        else if (token == "binc") binc = value;
        else if (token == "movetime") limits.movetime_ms = value;
        else if (token == "nodes") limits.node_limit = value;
        else if (token == "depth") limits.max_depth = static_cast<int>(value);
    }
    bool white = game.root.stm == WHITE;
    if (!infinite) {
        limits.time_left_ms = white ? wtime : btime;
        limits.increment_ms = white ? winc : binc;
    }

    search::stop_requested = false;
    search::set_game_history(game.history);
    worker = std::thread([limits, infinite] {
        const Position root = game.root;
        Move best = search::think(root, limits, [&root](const search::Report& report) {
            say(search::uci_info(root, report));
        });
        // UCI forbids answering an infinite search before it is told to stop.
        while (infinite && !search::stop_requested) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        search::Report report = search::last_report();
        if (report.depth > 0) say(search::uci_info(root, report));
        if (best != 0) {
            Position reply;
            make(root, reply, best);
            game.history.push_back(reply.key);
            game.last_reply = reply;
        }
        say("bestmove " + move_to_uci(best));
    });
}

void handle_perft(std::istringstream& in) {
    int depth = 1;
    in >> depth;
    auto started = std::chrono::steady_clock::now();
    uint64_t nodes = perft(game.root, depth);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    say("nodes " + std::to_string(nodes) + " time " + std::to_string(static_cast<int64_t>(seconds * 1000)));
}

void handle_eval() {
    nnue::Accumulator acc;
    nnue::refresh(acc, game.root);
    char key[32];
    std::snprintf(key, sizeof key, "%016llx", static_cast<unsigned long long>(game.root.key));
    say("eval " + std::to_string(nnue::evaluate(acc, game.root)) + " key " + key);
}

// Node rate and depth over fixed positions. The positions and the node limit are
// tests/bench.py's, and the table is carried from one position to the next as it is there,
// so the two engines' moves and node counts can be compared line by line.
void handle_bench(std::istringstream& in) {
    const std::pair<const char*, const char*> positions[] = {
        {"startpos", START_FEN},
        {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"},
        {"midgame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"},
        {"endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"},
        {"tactical", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"},
    };
    int64_t node_limit = 400000;
    in >> node_limit;
    int64_t total_nodes = 0;
    double total_seconds = 0.0;
    char line[160];
    std::snprintf(line, sizeof line, "%-10s %10s %8s %8s %6s  %s", "position", "nodes", "time", "knps",
                  "depth", "move");
    say(line);
    for (const auto& [name, fen] : positions) {
        Position root;
        from_fen(root, fen);
        search::set_game_history({root.key});
        search::Limits limits;
        limits.time_left_ms = 600000;
        limits.increment_ms = 500;
        limits.node_limit = node_limit;
        auto started = std::chrono::steady_clock::now();
        Move best = search::think(root, limits);
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        search::Report report = search::last_report();
        total_nodes += report.nodes;
        total_seconds += seconds;
        std::snprintf(line, sizeof line, "%-10s %10lld %7.2fs %8.0f %6d  %s", name,
                      static_cast<long long>(report.nodes), seconds, report.nodes / seconds / 1000.0,
                      report.depth, move_to_uci(best).c_str());
        say(line);
    }
    say("total " + std::to_string(total_nodes) + " nodes, " +
        std::to_string(static_cast<int64_t>(total_nodes / total_seconds)) + " nps");
}

// Returns false on quit.
bool handle(const std::string& line) {
    std::istringstream in(line);
    std::string command;
    in >> command;
    if (command == "uci") {
        say("id name chessathon-cpp");
        say("id author edward-baker05");
        say("option name Hash type spin default " + std::to_string(DEFAULT_HASH_MB) + " min 1 max 65536");
        say("uciok");
    } else if (command == "isready") {
        say("readyok");
    } else if (command == "setoption") {
        std::string token, name, value;
        in >> token;  // "name"
        while (in >> token && token != "value") name += (name.empty() ? "" : " ") + token;
        in >> value;
        wait_for_search();
        if (name == "Hash" && !value.empty()) search::table().resize(std::stoull(value));
    } else if (command == "ucinewgame") {
        wait_for_search();
        new_game();
    } else if (command == "position") {
        wait_for_search();
        handle_position(in);
    } else if (command == "go") {
        wait_for_search();
        handle_go(in);
    } else if (command == "stop") {
        search::stop_requested = true;
        wait_for_search();
    } else if (command == "quit") {
        search::stop_requested = true;
        wait_for_search();
        return false;
    } else if (command == "perft") {
        wait_for_search();
        handle_perft(in);
    } else if (command == "eval") {
        wait_for_search();
        handle_eval();
    } else if (command == "d") {
        wait_for_search();
        char key[32];
        std::snprintf(key, sizeof key, "%016llx", static_cast<unsigned long long>(game.root.key));
        say(to_fen(game.root) + " key " + key);
    } else if (command == "bench") {
        wait_for_search();
        handle_bench(in);
    } else if (!command.empty()) {
        say("info string unknown command " + command);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    init_bitboards();
    std::string error;
    if (!nnue::load(error)) {
        std::cerr << "cannot load the network: " << error << '\n';
        return 1;
    }
    search::init();
    from_fen(game.root, START_FEN);
    game.history = {game.root.key};

    if (argc > 1) {
        std::string line;
        for (int i = 1; i < argc; ++i) line += (i > 1 ? " " : "") + std::string(argv[i]);
        handle(line);
        wait_for_search();
        return 0;
    }
    std::string line;
    while (std::getline(std::cin, line))
        if (!handle(line)) return 0;
    search::stop_requested = true;
    wait_for_search();
    return 0;
}
