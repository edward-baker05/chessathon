// The UCI front end, and the game tracking that lets a bare FEN continue a game.
//
// Every command in the UCI specification is accepted, with every parameter, in any order,
// and whatever is not understood is ignored rather than rejected: the specification asks an
// engine to skip unknown tokens and "try to parse the rest of the string in this line", so
// `joho debug on` still turns debugging on. Some input is accepted without changing
// anything yet; the notes on each handler say which.
//
// Game tracking. A `position ... moves ...` command carries the whole game and is the
// repetition history. A bare FEN carries no game identity, so
// there a position one legal move on from our last reply continues the game, and anything
// else starts a new one, clearing what was learned.
//
// Beyond UCI: `perft N`, `eval`, `d` and `bench [nodes]`. Arguments on the command line run
// as one command and exit, so `engine bench` works.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "movegen.h"
#include "nnue.h"
#include "search.h"

namespace {

using Tokens = std::vector<std::string>;

constexpr int64_t MAX_HASH_MB = 65536;

std::mutex output_mutex;

void say(const std::string& line) {
    std::lock_guard lock(output_mutex);
    std::cout << line << '\n' << std::flush;
}

// `debug on` asks for extra information as `info string` lines.
bool debugging = false;

void debug(const std::string& text) {
    if (debugging) say("info string " + text);
}

struct Game {
    Position root;
    // Every position of the game so far, ending with the root.
    std::vector<Key> history;
    // The position after our own last reply.
    std::optional<Position> last_reply;
};

Game game;

// The search in progress, if any, and whether it would ever end on its own.
std::thread worker;
bool infinite_search = false;
bool unlimited_search = false;

// Waits for the search to finish, stopping it first if it never would on its own. What
// the GUI sends after `go` is not meant to arrive until `bestmove` has, but waiting on an
// infinite search would never answer at all.
void finish_search() {
    if (!worker.joinable()) return;
    if (infinite_search || unlimited_search || search::pondering) search::stop_requested = true;
    worker.join();
}

void stop_search() {
    if (!worker.joinable()) return;
    search::stop_requested = true;
    worker.join();
}

Tokens split(const std::string& line) {
    Tokens tokens;
    std::istringstream in(line);
    for (std::string token; in >> token;) tokens.push_back(token);
    return tokens;
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string join(const Tokens& tokens, size_t from, size_t to) {
    std::string out;
    for (size_t i = from; i < to && i < tokens.size(); ++i) out += (out.empty() ? "" : " ") + tokens[i];
    return out;
}

std::optional<int64_t> to_int(const std::string& token) {
    int64_t value = 0;
    const char* begin = token.data() + (token.starts_with('+') ? 1 : 0);
    const char* end = token.data() + token.size();
    auto [ptr, error] = std::from_chars(begin, end, value);
    if (error != std::errc() || ptr != end) return std::nullopt;
    return value;
}

std::string hex(Key key) {
    char text[32];
    std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(key));
    return text;
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

void handle_uci() {
    say("id name chessathon");
    say("id author edward-baker05");
    say("option name Hash type spin default " + std::to_string(DEFAULT_HASH_MB) + " min 1 max " +
        std::to_string(MAX_HASH_MB));
    say("option name Clear Hash type button");
    say("option name Ponder type check default false");
    say("option name Threads type spin default 1 min 1 max " + std::to_string(search::MAX_THREADS));
    say("uciok");
}

void handle_debug(const Tokens& args) {
    // A bare `debug` is taken as `debug on`.
    debugging = args.empty() || args[0] != "off";
}

// `setoption name <id> [value <x>]`. Both the name and the value may contain spaces, and
// names are matched without regard to case, as most GUIs expect.
void handle_setoption(const Tokens& args) {
    auto name_at = std::find(args.begin(), args.end(), "name");
    if (name_at == args.end()) return;
    auto value_at = std::find(name_at, args.end(), "value");
    size_t name_index = static_cast<size_t>(name_at - args.begin()) + 1;
    size_t value_index = static_cast<size_t>(value_at - args.begin());
    std::string name = lower(join(args, name_index, value_index));
    std::string value = value_at == args.end() ? "" : join(args, value_index + 1, args.size());

    finish_search();
    if (name == "hash") {
        std::optional<int64_t> megabytes = to_int(value);
        if (!megabytes) return debug("Hash needs a number of megabytes, not '" + value + "'");
        try {
            search::table().resize(static_cast<size_t>(std::clamp<int64_t>(*megabytes, 1, MAX_HASH_MB)));
        } catch (const std::bad_alloc&) {
            say("info string cannot allocate " + value + " MB for the hash; keeping the current table");
        }
    } else if (name == "clear hash") {
        search::table().clear();
    } else if (name == "threads") {
        std::optional<int64_t> count = to_int(value);
        if (!count) return debug("Threads needs a number, not '" + value + "'");
        try {
            search::set_threads(static_cast<int>(std::clamp<int64_t>(*count, 1, search::MAX_THREADS)));
        } catch (const std::bad_alloc&) {
            say("info string cannot allocate " + value + " threads");
        }
    } else if (name == "ponder") {
        // Pondering happens when a GUI sends `go ponder`, whatever this says. It is accepted
        // so that GUIs which always set it can.
    } else {
        debug("no option named '" + name + "'");
    }
}

// `position [fen <fenstring> | startpos] [moves <move1> ... <movei>]`.
void handle_position(const Tokens& args) {
    if (args.empty()) return;
    std::string fen;
    size_t i = 1;
    if (args[0] == "startpos") {
        fen = START_FEN;
    } else if (args[0] == "fen") {
        for (; i < args.size() && args[i] != "moves"; ++i) fen += (fen.empty() ? "" : " ") + args[i];
    } else {
        return debug("position needs startpos or fen, not '" + args[0] + "'");
    }
    Position pos;
    if (!from_fen(pos, fen)) {
        say("info string invalid fen '" + fen + "'; keeping the previous position");
        return;
    }
    while (i < args.size() && args[i] != "moves") ++i;

    std::vector<Key> keys{pos.key};
    bool listed = false;
    for (++i; i < args.size(); ++i) {
        Move move = parse_uci_move(pos, lower(args[i]));
        if (move == 0) {
            say("info string illegal move '" + args[i] + "' in " + to_fen(pos) + "; ignoring the rest");
            break;
        }
        Position next;
        make(pos, next, move);
        pos = next;
        keys.push_back(pos.key);
        listed = true;
    }

    // A listed game that passes through our last reply is the game we are playing, however
    // it went on from there; that is what a GUI sends after pondering on a move that was
    // not played. A bare FEN has to be exactly one move on.
    bool continues =
        game.last_reply && ((listed && std::find(keys.begin(), keys.end(), game.last_reply->key) != keys.end()) ||
                            follows(*game.last_reply, pos));
    if (!continues) {
        new_game();
        debug("new game");
    }
    if (listed)
        game.history = keys;
    else
        game.history.push_back(pos.key);
    game.root = pos;
}

bool is_go_keyword(const std::string& token) {
    static const char* const KEYWORDS[] = {"searchmoves", "ponder", "wtime", "btime", "winc", "binc",
                                           "movestogo", "depth", "nodes", "mate", "movetime", "infinite"};
    return std::find(std::begin(KEYWORDS), std::end(KEYWORDS), token) != std::end(KEYWORDS);
}

// `go` with any of its parameters. `movestogo` and `mate` are accepted and do not change
// the search yet: the time allocation already caps one move at a twelfth of the clock, and
// the search stops at the first forced mate it proves.
void handle_go(const Tokens& args) {
    search::Limits limits;
    bool infinite = false;
    bool ponder = false;
    std::optional<int64_t> wtime, btime, winc, binc;
    bool depth_given = false;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& token = args[i];
        if (token == "infinite") {
            infinite = true;
        } else if (token == "ponder") {
            ponder = true;
        } else if (token == "searchmoves") {
            while (i + 1 < args.size() && !is_go_keyword(args[i + 1])) {
                Move move = parse_uci_move(game.root, lower(args[++i]));
                if (move != 0) limits.searchmoves.push_back(move);
            }
        } else if (is_go_keyword(token)) {
            std::optional<int64_t> value = i + 1 < args.size() ? to_int(args[i + 1]) : std::nullopt;
            if (!value) continue;
            ++i;
            if (token == "wtime") wtime = value;
            else if (token == "btime") btime = value;
            else if (token == "winc") winc = value;
            else if (token == "binc") binc = value;
            else if (token == "movetime") limits.movetime_ms = std::max<int64_t>(*value, 1);
            else if (token == "nodes") limits.node_limit = std::max<int64_t>(*value, 0);
            else if (token == "depth") {
                limits.max_depth = static_cast<int>(std::clamp<int64_t>(*value, 1, search::MAX_DEPTH));
                depth_given = true;
            }
        }
    }
    const std::optional<int64_t>& clock = game.root.stm == WHITE ? wtime : btime;
    const std::optional<int64_t>& increment = game.root.stm == WHITE ? winc : binc;
    if (clock && !infinite) {
        limits.has_clock = true;
        limits.time_left_ms = *clock;
        limits.increment_ms = increment.value_or(0);
    }

    infinite_search = infinite;
    unlimited_search = !limits.has_clock && limits.movetime_ms == 0 && limits.node_limit == 0 && !depth_given;
    search::stop_requested = false;
    search::pondering = ponder;
    search::set_game_history(game.history);
    worker = std::thread([limits, infinite, ponder] {
        const Position root = game.root;
        Move best = search::think(root, limits, [&root](const search::Report& report) {
            say(search::uci_info(root, report));
        });
        // UCI forbids answering an infinite or pondering search before it is told to stop,
        // or, when pondering, that the expected move was played.
        while ((infinite || search::pondering) && !search::stop_requested)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // A ponder search that was stopped rather than hit was never a move we played.
        bool played = !(ponder && search::pondering);
        search::pondering = false;

        search::Report report = search::last_report();
        if (report.depth > 0) say(search::uci_info(root, report));
        if (best == 0) {
            say("bestmove (none)");
            return;
        }
        if (played) {
            Position reply;
            make(root, reply, best);
            game.history.push_back(reply.key);
            game.last_reply = reply;
        }
        std::string line = "bestmove " + move_to_uci(best);
        std::vector<Move> pv = search::principal_variation(root);
        if (pv.size() >= 2 && pv[0] == best) line += " ponder " + move_to_uci(pv[1]);
        say(line);
    });
}

void handle_perft(const Tokens& args) {
    int depth = static_cast<int>(std::clamp<int64_t>(args.empty() ? 1 : to_int(args[0]).value_or(1), 0, 20));
    auto started = std::chrono::steady_clock::now();
    uint64_t nodes = perft(game.root, depth);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    say("nodes " + std::to_string(nodes) + " time " + std::to_string(static_cast<int64_t>(seconds * 1000)));
}

void handle_eval() {
    nnue::Accumulator acc;
    nnue::refresh(acc, game.root);
    say("eval " + std::to_string(nnue::evaluate(acc, game.root)) + " key " + hex(game.root.key));
}

// Node rate and depth over fixed positions. The positions and the node limit are
// fixed, and the table is carried from one position to the next, so the moves and node counts
// can be compared line by line between builds.
void handle_bench(const Tokens& args) {
    const std::pair<const char*, const char*> positions[] = {
        {"startpos", START_FEN},
        {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"},
        {"midgame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"},
        {"endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"},
        {"tactical", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"},
    };
    int64_t node_limit = args.empty() ? 400000 : to_int(args[0]).value_or(400000);
    int64_t total_nodes = 0;
    double total_seconds = 0.0;
    char line[160];
    std::snprintf(line, sizeof line, "%-10s %10s %8s %8s %6s  %s", "position", "nodes", "time", "knps",
                  "depth", "move");
    say(line);
    search::stop_requested = false;
    for (const auto& [name, fen] : positions) {
        Position root;
        from_fen(root, fen);
        search::set_game_history({root.key});
        search::Limits limits;
        limits.has_clock = true;
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

// Runs one input line. Returns false on quit.
bool handle(const std::string& line) {
    Tokens tokens = split(line);
    static const char* const COMMANDS[] = {"uci",  "debug", "isready", "setoption", "register", "ucinewgame",
                                           "position", "go", "stop", "ponderhit", "quit", "perft",
                                           "eval", "d", "bench"};
    auto at = std::find_if(tokens.begin(), tokens.end(), [](const std::string& token) {
        return std::find(std::begin(COMMANDS), std::end(COMMANDS), token) != std::end(COMMANDS);
    });
    if (at == tokens.end()) {
        if (!tokens.empty()) debug("unknown command '" + tokens[0] + "'");
        return true;
    }
    std::string command = *at;
    Tokens args(at + 1, tokens.end());

    if (command == "uci") {
        handle_uci();
    } else if (command == "debug") {
        handle_debug(args);
    } else if (command == "isready") {
        // Answered at once, even mid-search, as the specification requires.
        say("readyok");
    } else if (command == "setoption") {
        handle_setoption(args);
    } else if (command == "register") {
        // No registration is needed; `register later` and a name and code are all fine.
    } else if (command == "ucinewgame") {
        finish_search();
        new_game();
    } else if (command == "position") {
        finish_search();
        handle_position(args);
    } else if (command == "go") {
        finish_search();
        handle_go(args);
    } else if (command == "stop") {
        stop_search();
    } else if (command == "ponderhit") {
        if (search::pondering) search::ponderhit();
    } else if (command == "quit") {
        stop_search();
        return false;
    } else if (command == "perft") {
        finish_search();
        handle_perft(args);
    } else if (command == "eval") {
        finish_search();
        handle_eval();
    } else if (command == "d") {
        finish_search();
        say(to_fen(game.root) + " key " + hex(game.root.key));
    } else if (command == "bench") {
        finish_search();
        handle_bench(args);
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
        finish_search();
        return 0;
    }
    for (std::string line; std::getline(std::cin, line);)
        if (!handle(line)) return 0;
    // The GUI went away without saying quit.
    stop_search();
    return 0;
}
