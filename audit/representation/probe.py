"""Frozen board-layer correctness and compiled performance probe. No production edits."""

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
from numba import njit

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--engine", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(Path(args.engine).resolve()))
started = time.perf_counter()
import bitboard as bb  # noqa: E402
import movegen as mg  # noqa: E402
import nnue  # noqa: E402
import position as pos  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - started


def corpus() -> list[chess.Board]:
    rng = random.Random(731209)
    boards = []
    starts = [
        chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/P6p/8/3pP3/8/8/p6P/4K3 w - d6 0 1",
    ]
    for game in range(80):
        board = chess.Board(starts[game % len(starts)])
        for ply in range(160):
            if board.is_game_over():
                break
            if ply % 4 == 0:
                boards.append(board.copy(stack=False))
            board.push(rng.choice(list(board.legal_moves)))
    return boards


@njit(cache=False)
def exhaustive_sliders() -> int:
    count = 0
    for rook in (False, True):
        for sq in range(64):
            mask = bb.slider_mask(sq, rook)
            sub = np.uint64(0)
            while True:
                actual = bb.rook_attacks(sq, sub) if rook else bb.bishop_attacks(sq, sub)
                assert actual == bb.slider_attacks(sq, sub, rook)
                # Occupancy outside the relevant mask, including board edges, must agree too.
                full = sub | ~mask
                actual = bb.rook_attacks(sq, full) if rook else bb.bishop_attacks(sq, full)
                assert actual == bb.slider_attacks(sq, full, rook)
                count += 2
                sub = (sub - mask) & mask
                if sub == 0:
                    break
    return count


def verify(boards: list[chess.Board]) -> dict[str, Any]:
    states, mails = pos.new_stacks()
    moves = np.full(1024, -1234567, np.int32)
    acc = nnue.new_accumulator(3)
    edges = checked = ep_checks = 0
    flags: dict[int, int] = {}
    max_pseudo = 0
    for board in boards:
        pos.encode(board, states[0], mails[0])
        nnue.refresh(acc, 0, states[0], mails[0])
        assert pos.in_check(states[0]) == board.is_check()
        for side in (chess.WHITE, chess.BLACK):
            for square in range(64):
                assert pos.attacked(states[0], square, int(not side)) == board.is_attacked_by(
                    side, square
                ), (board.fen(), side, square)
            pin_mask = sum(
                1 << sq for sq in board.pieces(chess.PAWN, side) if board.is_pinned(side, sq)
            )
            for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                pin_mask |= sum(
                    1 << sq for sq in board.pieces(piece, side) if board.is_pinned(side, sq)
                )
            assert pos.pinned_pieces(states[0], int(not side)) == pin_mask
        if board.ep_square is not None:
            assert pos.ep_is_capturable(states[0], board.ep_square) == board.has_legal_en_passant()
            ep_checks += 1
        count = mg.generate(states[0], moves, 0)
        max_pseudo = max(count, max_pseudo)
        assert count < mg.MAX_MOVES
        assert np.all(moves[mg.MAX_MOVES :] == -1234567)
        legal = set()
        for packed in moves[:count].copy():
            pos.make(states[0], mails[0], states[1], mails[1], packed)
            if not pos.legal_after(states[1], int(not board.turn)):
                continue
            uci = mg.move_to_uci(int(packed))
            legal.add(uci)
            nnue.apply(acc, 0, states[0], mails[0], packed)
            board.push_uci(uci)
            pos.encode(board, states[2], mails[2])
            np.testing.assert_array_equal(states[1], states[2])
            np.testing.assert_array_equal(mails[1], mails[2])
            nnue.refresh(acc, 2, states[2], mails[2])
            np.testing.assert_array_equal(acc[1], acc[2])
            board.pop()
            flag = int(packed) >> 15
            flags[flag] = flags.get(flag, 0) + 1
            edges += 1
        assert legal == {m.uci() for m in board.legal_moves}
        checked += int(board.is_check())
        # Captures deliberately include queen promotions only in current qsearch.
        count = mg.generate_captures(states[0], moves, 0)
        captures = set()
        for packed in moves[:count].copy():
            pos.make(states[0], mails[0], states[1], mails[1], packed)
            if pos.legal_after(states[1], int(not board.turn)):
                captures.add(mg.move_to_uci(int(packed)))
        expected = {
            m.uci()
            for m in board.legal_moves
            if m.promotion == chess.QUEEN or (board.is_capture(m) and not m.promotion)
        }
        assert captures == expected
        assert mg.has_legal_move(states[0], mails[0], states[1], mails[1], moves) == bool(legal)
        pos.make_null(states[0], mails[0], states[1], mails[1])
        np.testing.assert_array_equal(states[0, :8], states[1, :8])
        np.testing.assert_array_equal(mails[0], mails[1])
        assert states[1, bb.KEY] == pos.full_key(states[1])
    for a in range(64):
        assert int(bb.KNIGHT_ATT[a]) == chess.BB_KNIGHT_ATTACKS[a]
        assert int(bb.KING_ATT[a]) == chess.BB_KING_ATTACKS[a]
        for color in (0, 1):
            assert int(bb.PAWN_ATT[color, a]) == chess.BB_PAWN_ATTACKS[not color][a]
        for b in range(64):
            assert int(bb.BETWEEN[a, b]) == chess.between(a, b)
            assert int(bb.LINE[a, b]) == chess.ray(a, b)
    return {
        "boards": len(boards),
        "legal_edges": edges,
        "checked_positions": checked,
        "ep_positions": ep_checks,
        "flags": flags,
        "max_pseudo_moves": max_pseudo,
        "slider_cases": exhaustive_sliders(),
    }


@njit(cache=False)
def stages(
    states: Any,
    mails: Any,
    edges: Any,
    out_state: Any,
    out_mail: Any,
    moves: Any,
    acc: Any,
    mode: int,
    loops: int,
) -> int:
    checksum = 0
    for _ in range(loops):
        for i in range(edges.shape[0]):
            index = edges[i, 0]
            move = np.int32(edges[i, 1])
            state = states[index]
            mail = mails[index]
            if mode == 0:
                checksum += mg.generate(state, moves, 0)
            elif mode == 1:
                checksum += mg.generate_captures(state, moves, 0)
            elif mode == 2:
                pos.make(state, mail, out_state, out_mail, move)
                checksum += np.int64(out_state[bb.KEY] & np.uint64(65535))
            elif mode == 3:
                pos.make(state, mail, out_state, out_mail, move)
                checksum += int(pos.legal_after(out_state, np.int64(state[bb.STM])))
            elif mode == 4:
                checksum += pos.see(state, mail, move)
            elif mode == 5:
                checksum += int(mg.has_legal_move(state, mail, out_state, out_mail, moves))
            elif mode == 6:
                nnue.apply(acc, 0, state, mail, move)
                checksum += nnue.forward(acc, 1, state)
            elif mode == 7:
                checksum += int(pos.in_check(state))
            elif mode == 8:
                pos.make_null(state, mail, out_state, out_mail)
                checksum += np.int64(out_state[bb.KEY] & np.uint64(65535))
    return checksum


def packed_move(board: chess.Board, uci: str) -> np.int32:
    m = chess.Move.from_uci(uci)
    flag = 3 if m.promotion else 2 if board.is_castling(m) else 1 if board.is_en_passant(m) else 0
    return np.int32(mg.pack(m.from_square, m.to_square, (m.promotion or 1) - 1, flag))


def see_cases() -> list[dict[str, Any]]:
    examples = [
        ("4k3/8/8/8/8/4n3/8/K2bR3 w - - 0 1", "e1d1", -170),
        ("8/7k/8/8/8/R7/1p6/b6K w - - 0 1", "a3a1", -970),
    ]
    result = []
    states, mails = pos.new_stacks()
    for fen, uci, expected in examples:
        board = chess.Board(fen)
        assert board.is_valid() and chess.Move.from_uci(uci) in board.legal_moves
        pos.encode(board, states[0], mails[0])
        actual = pos.see(states[0], mails[0], packed_move(board, uci))
        board.push_uci(uci)
        replies = [
            m.uci() for m in board.legal_moves if m.to_square == chess.parse_square(uci[2:4])
        ]
        result.append(
            {
                "fen": fen,
                "move": uci,
                "actual": actual,
                "expected": expected,
                "legal_recaptures": replies,
            }
        )
    return result


def main() -> None:
    boards = corpus()
    result: dict[str, Any] = {
        "engine": args.engine,
        "import_seconds": IMPORT_SECONDS,
        "corpus_boards": len(boards),
    }
    if args.verify:
        result["verification"] = verify(boards)
        print(result["verification"], flush=True)
    result["see_cases"] = see_cases()
    states = np.zeros((len(boards), bb.NFIELDS), np.uint64)
    mails = np.zeros((len(boards), 64), np.int8)
    edges = []
    moves = np.zeros(1024, np.int32)
    categories: dict[str, int] = {
        "pseudo": 0,
        "legal": 0,
        "in_check_pseudo": 0,
        "in_check_legal": 0,
        "noncheck_pseudo": 0,
        "noncheck_legal": 0,
    }
    out_state = np.zeros(bb.NFIELDS, np.uint64)
    out_mail = np.zeros(64, np.int8)
    for index, board in enumerate(boards):
        pos.encode(board, states[index], mails[index])
        n = mg.generate(states[index], moves, 0)
        for m in moves[:n].copy():
            edges.append((index, int(m)))
            pos.make(states[index], mails[index], out_state, out_mail, m)
            legal = int(pos.legal_after(out_state, int(not board.turn)))
            categories["pseudo"] += 1
            categories["legal"] += legal
            prefix = "in_check" if board.is_check() else "noncheck"
            categories[prefix + "_pseudo"] += 1
            categories[prefix + "_legal"] += legal
    result["move_counts"] = categories
    edges_array = np.array(edges, np.int64)
    result["corpus_sha256"] = hashlib.sha256(states.tobytes() + edges_array.tobytes()).hexdigest()
    acc = nnue.new_accumulator(3)
    nnue.refresh(acc, 0, states[0], mails[0])
    result["micro_ns"] = {}
    names = [
        "generate",
        "captures",
        "make",
        "make_and_legal",
        "see",
        "has_legal",
        "nnue_update_forward",
        "in_check",
        "null",
    ]
    for mode, name in enumerate(names):
        stages(states, mails, edges_array, out_state, out_mail, moves, acc, mode, 1)
        timings = []
        for _ in range(7):
            start = time.perf_counter_ns()
            check = stages(states, mails, edges_array, out_state, out_mail, moves, acc, mode, 8)
            timings.append((time.perf_counter_ns() - start) / (len(edges) * 8))
        result["micro_ns"][name] = {
            "median": statistics.median(timings),
            "all": timings,
            "checksum": check,
        }
        print(name, round(statistics.median(timings), 2), flush=True)
    llvm = pos.make.inspect_llvm(pos.make.signatures[0])
    (OUT / f"{args.name}-make.ll").write_text(llvm)
    asm = bb.lsb.inspect_asm(bb.lsb.signatures[0])
    (OUT / f"{args.name}-lsb.asm").write_text(asm)
    result["codegen"] = {
        "make_has_vector_i8": "<32 x i8>" in llvm or "<16 x i8>" in llvm,
        "popcount_intrinsic": "llvm.ctpop" in bb.popcount.inspect_llvm(bb.popcount.signatures[0]),
        "lsb_tzcnt_or_bsf": "tzcnt" in asm or "bsfq" in asm,
        "table_bytes": {
            name: getattr(bb, name).nbytes
            for name in ["RTABLE", "BTABLE", "BETWEEN", "LINE", "Z_PIECE"]
        },
    }
    (OUT / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
