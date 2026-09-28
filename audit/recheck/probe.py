"""Current-state reproducers; no production edits."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import chess
import numpy as np

import nnue
import position
import search
import tt


def prep(fen):
    b = chess.Board(fen)
    assert b.is_valid(), (fen, b.status())
    w = search.WORK
    tt.tt_clear(w.table)
    search.clear_tables(w)
    search.set_game_history([])
    search.set_pruning(True)
    search._prepare(b, 3600000, 0, 0, w)
    return b, w


out = {}
for name, fen in [
    ("four_piece_stalemate", "7k/7p/7p/7p/7P/8/8/K5R1 b - - 0 1"),
    ("bare_king_stalemate", "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"),
    ("quiet_mate", "rrrrr2k/8/5KQ1/8/8/8/8/8 w - - 0 1"),
]:
    b, w = prep(fen)
    static = int(nnue.forward(w.acc, 0, w.state[0]))
    row = {"fen": fen, "static": static, "stalemate": b.is_stalemate(), "mates": []}
    for m in list(b.legal_moves):
        b.push(m)
        if b.is_checkmate():
            row["mates"].append(m.uci())
        b.pop()
    row["q"] = int(search.qsearch(w, 0, np.int32(-32000), np.int32(32000)))
    for enabled in (True, False):
        for alpha in (0, static - 200, static + 1000):
            b, w = prep(fen)
            search.set_pruning(enabled)
            row[f"prune_{enabled}_alpha_{alpha}"] = int(
                search.negamax(w, 0, 1, np.int32(alpha), np.int32(alpha + 1), False)
            )
    out[name] = row
    print(name, row, flush=True)
# TT rule50 context: a real exact search at low halfmove then re-use at halfmove99.
fen = "7k/8/8/8/8/8/8/KR6 w - - 0 1"
b, w = prep(fen)
a = int(search.negamax(w, 0, 1, np.int32(-32000), np.int32(32000), True))
b.halfmove_clock = 99
position.encode(b, w.state[0], w.mail[0])
nnue.refresh(w.acc, 0, w.state[0], w.mail[0])
w.ints[search.I_ABORT] = 0
warm_wide = int(search.negamax(w, 0, 1, np.int32(-32000), np.int32(32000), False))
warm = int(search.negamax(w, 0, 1, np.int32(-100), np.int32(100), False))
tt.tt_clear(w.table)
cold_wide = int(search.negamax(w, 0, 1, np.int32(-32000), np.int32(32000), False))
tt.tt_clear(w.table)
cold = int(search.negamax(w, 0, 1, np.int32(-100), np.int32(100), False))
search.set_pruning(False)
tt.tt_clear(w.table)
full = int(search.negamax(w, 0, 1, np.int32(-32000), np.int32(32000), True))
out["rule50"] = {
    "fen": b.fen(),
    "low_half_exact": a,
    "warm": warm,
    "cold": cold,
    "warm_wide": warm_wide,
    "cold_wide": cold_wide,
    "full": full,
}
print("rule50", out["rule50"], flush=True)
Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=2) + "\n")
