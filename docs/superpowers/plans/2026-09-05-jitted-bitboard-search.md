# Jitted Bitboard Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `python-chess` alpha-beta in `search.py` with a numba-jitted bitboard engine reaching roughly 40x the node rate, and a modern search on top of it.

**Architecture:** Own bitboard move generation, make and search, entirely inside `njit` functions over numpy arrays. `python-chess` survives only at the FEN and UCI boundary in `agent.py` and as the legality oracle in tests. Large tables (attack tables, transposition table, history) are module-level numpy arrays read directly by jitted code; per-node values are passed as arguments.

**Tech Stack:** Python 3.12, numba 0.67 (`njit`, `objmode`), numpy 2.5, python-chess 1.11 (boundary and tests only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-jitted-bitboard-search-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Never name a root file after an importable module.** The zip is first on `sys.path`. `chess.py`, `types.py`, `random.py` would shadow the real module and the failure will look unrelated.
- **Root `*.py` only ships.** `harness/package.py` globs root `*.py` plus `weights`. A subpackage directory silently does not ship. All engine modules go flat at the repo root.
- **`cache=False` on every `njit` decorator, with no exceptions and no opt-in.** numba bakes the contents of global numpy arrays into cached binaries with no warning, and this design keeps its tables in globals. Verified: a cached function returned a stale value after its global table changed.
- **Do not edit `harness/`.** It mirrors the platform protocol and clock. `tests/match.py` imports from it instead.
- **No new runtime dependencies.** The platform preinstalls torch, numpy, python-chess, onnxruntime and numba, and installs nothing else. A `requirements.txt` is ignored. pytest is a dev dependency only and never imported by shipped code.
- **No native binaries in the zip.** Ship Python source only.
- **No commit watermark.** Do not add `Co-Authored-By` or `Claude-Session` lines to commits.
- **No em dashes** in any prose, comment or docstring.
- **Style:** Python 3.12, type-annotated, `ruff` and `mypy --strict` clean at `line-length = 100`. `mypy --strict` already passes on `njit`-decorated code in this repo, so no decorator override is needed.
- **Import budget is 90 s** and every jitted function must be warmed once at import with the exact argument types it will really see, because numba compiles per signature.
- **Time control is 120 s + 0.5 s per move on wall time**, one AMD EPYC 9V74 core at 2.60 GHz, 2 GB RAM. A flag loses the game.
- **Move encoding is a single `int32`:** `from | to << 6 | promo << 12 | flag << 15`, flags `0` normal, `1` en passant, `2` castle, `3` promotion, promo `1..4` meaning knight, bishop, rook, queen.
- **En passant is stored as square + 1, with 0 meaning none.** The state vector is unsigned, so a `-1` sentinel compares wrongly. This bug was hit during the design probe.
- **Score convention:** centipawns, positive means the side to move is better. `MATE = 30000`, `MATE_IN_MAX = MATE - 256`, `INF = 32000`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `bitboard.py` | Field indices, bit primitives, leaper tables, magic sliders, Zobrist keys. No position logic. |
| `position.py` | State layout, `encode`, `attacked`, `make`, SEE, material and draw predicates. |
| `movegen.py` | Pseudo-legal generation, move field accessors, UCI codec. |
| `tt.py` | Transposition table storage, probe and store. |
| `search.py` | Iterative deepening, PVS, quiescence, ordering, pruning, time control. |
| `evaluate.py` | Owned by the user. This plan only changes its signature and ports the existing material term. |
| `agent.py` | FEN to state, game history across moves, search call, UCI out, legality safety net. |
| `tests/*.py` | pytest suites. Not shipped. |
| `snapshots/<tag>/` | Frozen engine copies used as A/B opponents. Not shipped. |

---

### Task 1: Build configuration and test scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Create: `tests/conftest.py`
- Create: `tests/test_scaffolding.py`
- Create: `.gitignore` entry for `snapshots/`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/conftest.py` exposing `random_positions(count: int, seed: int, max_plies: int) -> Iterator[chess.Board]`, used by every later test task.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scaffolding.py`:

```python
"""Proves the test harness can import root modules and generate positions."""

from tests.conftest import random_positions


def test_random_positions_are_varied_and_legal() -> None:
    boards = list(random_positions(count=50, seed=1, max_plies=30))
    assert len(boards) == 50
    fens = {board.fen() for board in boards}
    assert len(fens) > 40, "positions should be varied, not the same opening repeated"
    for board in boards:
        assert board.is_valid()


def test_random_positions_are_deterministic() -> None:
    first = [board.fen() for board in random_positions(count=20, seed=7, max_plies=20)]
    second = [board.fen() for board in random_positions(count=20, seed=7, max_plies=20)]
    assert first == second


def test_root_modules_are_importable_from_tests() -> None:
    """pythonpath must reach the repo root, or every later test fails to collect."""
    import agent

    assert callable(agent.get_move)
```

Do not assert anything about `evaluate.evaluate` here. Task 9 changes its signature, and a
test pinned to the current one would start failing two tasks later for no useful reason.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scaffolding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.conftest'` or a collection error, because pytest config and `conftest.py` do not exist yet.

- [ ] **Step 3: Write the implementation**

Create `tests/conftest.py`:

```python
"""Shared fixtures. Kept dependency-free so every test module can import it."""

import random
from collections.abc import Iterator

import chess


def random_positions(count: int, seed: int, max_plies: int = 40) -> Iterator[chess.Board]:
    """Yield varied legal positions by walking random legal moves from the start.

    Deterministic for a given seed so a failing test can always be reproduced.
    Games that end early restart from the initial position.
    """
    rng = random.Random(seed)
    produced = 0
    while produced < count:
        board = chess.Board()
        for _ in range(rng.randint(1, max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        yield board.copy()
        produced += 1
```

Add to `pyproject.toml` under `[dependency-groups]`, replacing the existing `dev` line:

```toml
[dependency-groups]
dev = ["mypy>=1.18", "ruff>=0.14", "pytest>=8.3"]
```

Add a new section to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Change the mypy section of `pyproject.toml` to cover the engine modules that later tasks create. Include them now so `make gate` starts failing the moment one of them is untyped:

```toml
[tool.mypy]
strict = true
files = ["agent.py", "evaluate.py", "search.py", "harness", "tests"]
```

Only files that already exist may appear here. mypy fails with "Can't read file" on a
listed path that is missing, so **each later task appends its own new module to this list as
part of its commit**: Task 2 adds `bitboard.py`, Task 3 adds `position.py`, Task 4 adds
`movegen.py`, Task 8 adds `tt.py`.

`mypy --strict` already passes on `njit`-decorated code in this repo, so no
`disallow_untyped_decorators` override is needed.

Add targets to `Makefile`, and add `test` to the `.PHONY` line:

```makefile
.PHONY: setup play arena zip gate test bench ab

test:
	uv run pytest -q

bench:
	uv run python tests/bench.py

ab:
	uv run python tests/match.py --opponent $(OPPONENT) --games $(if $(GAMES),$(GAMES),200) --nodes $(if $(NODES),$(NODES),200000)
```

Create `.gitignore` if absent, or append:

```
snapshots/
game.pgn
submission.zip
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv sync && uv run pytest tests/test_scaffolding.py -v`
Expected: 3 passed.

Run: `make gate`
Expected: ruff clean, mypy clean, two games finish.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml Makefile uv.lock .gitignore tests/
git commit -m "test: add pytest scaffolding and position generator"
```

---

### Task 2: `bitboard.py` primitives, tables and magics

**Files:**
- Create: `bitboard.py`
- Create: `tests/test_bitboard.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - Field index constants `PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5, WOCC=6, BOCC=7, STM=8, CASTLE=9, EP=10, HALF=11, KEY=12`, and `NFIELDS=16`.
  - `ONE`, `ZERO`: `np.uint64` constants.
  - `popcount(b: uint64) -> int64`, `lsb(b: uint64) -> int64`, both `njit`.
  - `rook_attacks(sq: int64, occ: uint64) -> uint64`, `bishop_attacks`, `queen_attacks`, all `njit`.
  - Global arrays `KNIGHT_ATT: uint64[64]`, `KING_ATT: uint64[64]`, `PAWN_ATT: uint64[2, 64]` where index 0 is white, `CASTLE_MASK: int64[64]`.
  - Zobrist globals `Z_PIECE: uint64[2, 6, 64]`, `Z_STM: uint64[1]`, `Z_CASTLE: uint64[16]`, `Z_EP: uint64[8]`.

**Notes for the implementer:**

`PAWN_ATT[0, sq]` is the set of squares a *white* pawn on `sq` attacks. To ask whether square `sq` is attacked by a *black* pawn, intersect `PAWN_ATT[0, sq]` with the black pawns. The index is inverted relative to intuition and getting it backwards produces a subtle, rarely triggered bug.

`Z_STM` is a one-element array rather than a scalar because numba freezes global scalars as compile-time constants but reads global array elements dynamically. Keep every mutable or late-initialised value in an array.

Generate the de Bruijn index table from the constant rather than transcribing it. A transcribed table is impossible to eyeball and wrong in a way that only shows up on some squares.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bitboard.py`:

```python
"""Bit primitives and attack tables, checked against python-chess."""

import chess
import numpy as np
import pytest

import bitboard as bb
from tests.conftest import random_positions


@pytest.mark.parametrize("square", range(64))
def test_lsb_finds_lowest_set_bit(square: int) -> None:
    assert bb.lsb(np.uint64(1) << np.uint64(square)) == square


def test_lsb_ignores_higher_bits() -> None:
    value = (np.uint64(1) << np.uint64(9)) | (np.uint64(1) << np.uint64(40))
    assert bb.lsb(value) == 9


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 1), (0xFF, 8), (0xFFFFFFFFFFFFFFFF, 64)])
def test_popcount(value: int, expected: int) -> None:
    assert bb.popcount(np.uint64(value)) == expected


@pytest.mark.parametrize("square", range(64))
def test_knight_and_king_tables_match_python_chess(square: int) -> None:
    assert int(bb.KNIGHT_ATT[square]) == chess.BB_KNIGHT_ATTACKS[square]
    assert int(bb.KING_ATT[square]) == chess.BB_KING_ATTACKS[square]


@pytest.mark.parametrize("square", range(64))
def test_pawn_attack_table_orientation(square: int) -> None:
    # PAWN_ATT[0] is what a WHITE pawn on `square` attacks.
    assert int(bb.PAWN_ATT[0, square]) == int(chess.BB_PAWN_ATTACKS[chess.WHITE][square])
    assert int(bb.PAWN_ATT[1, square]) == int(chess.BB_PAWN_ATTACKS[chess.BLACK][square])


def test_slider_attacks_match_python_chess_on_real_positions() -> None:
    for board in random_positions(count=60, seed=11):
        occ = np.uint64(board.occupied)
        for square in range(64):
            expected_rook = int(chess.BB_RANK_ATTACKS[square][board.occupied & chess.BB_RANK_MASKS[square]]
                                | chess.BB_FILE_ATTACKS[square][board.occupied & chess.BB_FILE_MASKS[square]])
            expected_bishop = int(chess.BB_DIAG_ATTACKS[square][board.occupied & chess.BB_DIAG_MASKS[square]])
            assert int(bb.rook_attacks(square, occ)) == expected_rook, f"rook on {square}"
            assert int(bb.bishop_attacks(square, occ)) == expected_bishop, f"bishop on {square}"
            assert int(bb.queen_attacks(square, occ)) == expected_rook | expected_bishop


def test_zobrist_keys_are_distinct_and_nonzero() -> None:
    keys = set()
    for colour in range(2):
        for piece in range(6):
            for square in range(64):
                key = int(bb.Z_PIECE[colour, piece, square])
                assert key != 0
                keys.add(key)
    assert len(keys) == 2 * 6 * 64, "Zobrist keys must not collide"


def test_castle_mask_clears_the_right_rights() -> None:
    # bits: 1 white kingside, 2 white queenside, 4 black kingside, 8 black queenside
    assert bb.CASTLE_MASK[chess.E1] == 15 ^ 3
    assert bb.CASTLE_MASK[chess.A1] == 15 ^ 2
    assert bb.CASTLE_MASK[chess.H1] == 15 ^ 1
    assert bb.CASTLE_MASK[chess.E8] == 15 ^ 12
    assert bb.CASTLE_MASK[chess.A8] == 15 ^ 8
    assert bb.CASTLE_MASK[chess.H8] == 15 ^ 4
    assert bb.CASTLE_MASK[chess.D4] == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bitboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bitboard'`.

- [ ] **Step 3: Write the implementation**

Create `bitboard.py` with this structure. The magic search is the only non-obvious part; the probe at `docs/superpowers/specs/2026-09-05-jitted-bitboard-search-design.md` validated this exact approach.

```python
"""Bit primitives, attack tables and Zobrist keys. No position logic lives here."""

import numpy as np
from numba import boolean, int64, njit, uint64

U = np.uint64
ONE = U(1)
ZERO = U(0)

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
WOCC, BOCC, STM, CASTLE, EP, HALF, KEY = 6, 7, 8, 9, 10, 11, 12
NFIELDS = 16

DEBRUIJN = U(0x03F79D71B4CB0A89)
```

Then, in order:

1. `DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)`, filled in a plain Python loop by `DEBRUIJN_INDEX[int(((ONE << U(i)) * DEBRUIJN) >> U(58))] = i` for `i in range(64)`. Deriving the table from the constant makes a wrong transcription impossible.
2. `popcount` and `lsb`, both decorated `@njit(int64(uint64), cache=False)`. `lsb` is `DEBRUIJN_INDEX[int(((b & (~b + ONE)) * DEBRUIJN) >> U(58))]`, where `b & (~b + ONE)` isolates the lowest set bit.
3. `ray_attacks(sq, dr, df, occ)`, `ray_mask(sq, dr, df)`, `slider_attacks(sq, occ, is_rook)`, `slider_mask(sq, is_rook)` as `njit` helpers. `ray_mask` walks outward but stops before the last square on the board, because edge squares do not affect a slider's blocked ray.
4. `splitmix(state: uint64[:]) -> uint64` as a seeded PRNG so magics are deterministic.
5. `build_magics(is_rook, seed)` as an `njit` function returning `(magic, mask, shift, offset, table)`. For each square: build the mask, enumerate its `2**bits` occupancy subsets with the carry-rippler `sub = (sub - mask) & mask`, compute the true attacks for each, then try sparse random candidates `splitmix() & splitmix() & splitmix()` until one maps every subset to a slot holding the same attack set.
6. Build the tables at import: `KNIGHT_ATT, KING_ATT, PAWN_ATT = build_leapers()`, then `RMAGIC, RMASK, RSHIFT, ROFF, RTABLE = build_magics(True, 0x1234567)` and the bishop equivalents with a different seed.
7. `rook_attacks`, `bishop_attacks` as `@njit(uint64(int64, uint64), cache=False)` doing `TABLE[OFF[sq] + int(((occ & MASK[sq]) * MAGIC[sq]) >> U(SHIFT[sq]))]`. `queen_attacks` is their union.
8. `CASTLE_MASK = np.full(64, 15, dtype=np.int64)` then clear the six relevant squares as asserted in the test.
9. Zobrist arrays filled from `np.random.default_rng(0x5EED)` via `rng.integers(1, 2**64, ..., dtype=np.uint64)`, so keys are deterministic and never zero.
10. A warm-up block at the bottom calling `popcount`, `lsb`, `rook_attacks`, `bishop_attacks` and `queen_attacks` once each with `uint64`/`int64` arguments, so compilation lands in the import budget.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bitboard.py -v`
Expected: all pass.

Run: `uv run python -c "import time; t=time.perf_counter(); import bitboard; print(f'{time.perf_counter()-t:.2f}s')"`
Expected: under 5 s.

- [ ] **Step 5: Commit**

```bash
git add bitboard.py tests/test_bitboard.py
git commit -m "feat(bitboard): add bit primitives, attack tables and self-generated magics"
```

---

### Task 3: `position.py` state layout, encoding and attack detection

**Files:**
- Create: `position.py`
- Create: `tests/test_position.py`

**Interfaces:**
- Consumes: everything `bitboard.py` produces.
- Produces:
  - `STACK_PLIES = 256`
  - `new_stacks() -> tuple[np.ndarray, np.ndarray]` returning `(state: uint64[STACK_PLIES, NFIELDS], mailbox: int8[STACK_PLIES, 64])`.
  - `encode(board: chess.Board, state: np.ndarray, mailbox: np.ndarray) -> None` writing into one row of each. Plain Python, called once per move.
  - `decode_fen(state: np.ndarray, mailbox: np.ndarray) -> str` for tests and debugging. Plain Python.
  - `attacked(state: uint64[:], sq: int64, by_black: int64) -> boolean`, `njit`.
  - `king_square(state: uint64[:], black: int64) -> int64`, `njit`.
  - `in_check(state: uint64[:]) -> boolean`, `njit`.

**Notes:** `mailbox[sq]` holds the piece type `0..5` for an occupied square and `-1` for empty. Colour comes from the occupancy bitboards, not the mailbox, so the mailbox stays a single small array.

- [ ] **Step 1: Write the failing test**

Create `tests/test_position.py`:

```python
"""Position encoding and attack detection, checked against python-chess."""

import chess
import numpy as np

import bitboard as bb
import position
from tests.conftest import random_positions


def encoded(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    return state[0], mailbox[0]


def test_encode_matches_python_chess_bitboards() -> None:
    for board in random_positions(count=100, seed=21):
        state, mailbox = encoded(board)
        assert int(state[bb.PAWN]) == board.pawns
        assert int(state[bb.KNIGHT]) == board.knights
        assert int(state[bb.BISHOP]) == board.bishops
        assert int(state[bb.ROOK]) == board.rooks
        assert int(state[bb.QUEEN]) == board.queens
        assert int(state[bb.KING]) == board.kings
        assert int(state[bb.WOCC]) == board.occupied_co[chess.WHITE]
        assert int(state[bb.BOCC]) == board.occupied_co[chess.BLACK]
        assert int(state[bb.STM]) == (0 if board.turn else 1)
        assert int(state[bb.HALF]) == board.halfmove_clock


def test_encode_stores_en_passant_as_square_plus_one() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    state, _ = encoded(board)
    assert int(state[bb.EP]) == chess.E3 + 1, "0 must be reserved to mean 'no en passant'"

    quiet = chess.Board()
    state, _ = encoded(quiet)
    assert int(state[bb.EP]) == 0


def test_encode_stores_castling_rights_bits() -> None:
    state, _ = encoded(chess.Board())
    assert int(state[bb.CASTLE]) == 15
    state, _ = encoded(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"))
    assert int(state[bb.CASTLE]) == 1 | 8


def test_mailbox_agrees_with_the_board() -> None:
    for board in random_positions(count=60, seed=22):
        _, mailbox = encoded(board)
        for square in range(64):
            piece = board.piece_at(square)
            assert mailbox[square] == (-1 if piece is None else piece.piece_type - 1)


def test_attacked_matches_python_chess() -> None:
    for board in random_positions(count=60, seed=23):
        state, _ = encoded(board)
        for square in range(64):
            for by_black, colour in ((0, chess.WHITE), (1, chess.BLACK)):
                assert bool(position.attacked(state, square, by_black)) == board.is_attacked_by(
                    colour, square
                ), f"square {square} by {'black' if by_black else 'white'} in {board.fen()}"


def test_in_check_matches_python_chess() -> None:
    for board in random_positions(count=200, seed=24):
        state, _ = encoded(board)
        assert bool(position.in_check(state)) == board.is_check()


def test_decode_fen_round_trips() -> None:
    for board in random_positions(count=60, seed=25):
        state, mailbox = encoded(board)
        assert position.decode_fen(state, mailbox) == board.fen()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_position.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'position'`.

- [ ] **Step 3: Write the implementation**

Create `position.py`. `attacked` is the performance-critical part and reads:

```python
@njit(boolean(uint64[:], int64, int64), cache=False)
def attacked(state, sq, by_black):
    occ = state[WOCC] | state[BOCC]
    them = state[BOCC] if by_black else state[WOCC]
    # PAWN_ATT[0, sq] is where a white pawn on sq attacks, which is exactly the set of
    # squares a black pawn would have to stand on to attack sq.
    if PAWN_ATT[0 if by_black else 1, sq] & state[PAWN] & them:
        return True
    if KNIGHT_ATT[sq] & state[KNIGHT] & them:
        return True
    if KING_ATT[sq] & state[KING] & them:
        return True
    if bishop_attacks(sq, occ) & (state[BISHOP] | state[QUEEN]) & them:
        return True
    if rook_attacks(sq, occ) & (state[ROOK] | state[QUEEN]) & them:
        return True
    return False
```

`encode` writes each field, sets `mailbox` from `board.piece_map()` with `-1` elsewhere, packs castling rights as `1` white kingside, `2` white queenside, `4` black kingside, `8` black queenside, stores `ep_square + 1` or `0`, and computes `state[KEY]` from scratch using the Zobrist tables. Compute the key here in one place so Task 6 has something to check its incremental updates against.

Add a warm-up block calling `attacked`, `king_square` and `in_check` once at import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_position.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add position.py tests/test_position.py
git commit -m "feat(position): add state layout, FEN encoding and attack detection"
```

---

### Task 4: `movegen.py` pseudo-legal generation and the move codec

**Files:**
- Create: `movegen.py`
- Create: `tests/test_movegen.py`

**Interfaces:**
- Consumes: `bitboard.py`, `position.attacked`.
- Produces:
  - `MAX_MOVES = 256`
  - `generate(state: uint64[:], moves: int32[:], base: int64) -> int64`, `njit`, returning the index one past the last move written.
  - `generate_captures(state: uint64[:], moves: int32[:], base: int64) -> int64`, `njit`, captures and queen promotions only.
  - `move_from(m: int32) -> int64`, `move_to`, `move_promo`, `move_flag`, all `njit`.
  - `pack(frm: int, to: int, promo: int = 0, flag: int = 0) -> int` and `move_to_uci(m: int) -> str`, plain Python for tests and `agent.py`.

**Notes:** castling generation checks that the king's origin and the square it passes through are unattacked, and that the intervening squares are empty. It deliberately does *not* check the destination, because the legality filter in Task 5 catches that. Castling destination squares are g1=6, c1=2, g8=62, c8=58.

- [ ] **Step 1: Write the failing test**

Create `tests/test_movegen.py`:

```python
"""Pseudo-legal generation, checked against python-chess."""

import chess
import numpy as np

import movegen
import position
from tests.conftest import random_positions


def generated_ucis(board: chess.Board, captures_only: bool = False) -> set[str]:
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    generator = movegen.generate_captures if captures_only else movegen.generate
    count = generator(state[0], moves, 0)
    return {movegen.move_to_uci(int(moves[i])) for i in range(count)}


def test_move_codec_round_trips() -> None:
    packed = movegen.pack(chess.E7, chess.E8, promo=4, flag=3)
    assert movegen.move_from(np.int32(packed)) == chess.E7
    assert movegen.move_to(np.int32(packed)) == chess.E8
    assert movegen.move_promo(np.int32(packed)) == 4
    assert movegen.move_flag(np.int32(packed)) == 3
    assert movegen.move_to_uci(packed) == "e7e8q"


def test_pseudo_legal_is_a_superset_of_legal() -> None:
    for board in random_positions(count=200, seed=31):
        legal = {move.uci() for move in board.legal_moves}
        assert legal <= generated_ucis(board), f"missing legal moves in {board.fen()}"


def test_pseudo_legal_matches_python_chess_exactly() -> None:
    # python-chess's pseudo_legal_moves excludes castling through check, and so do we.
    for board in random_positions(count=200, seed=32):
        expected = {move.uci() for move in board.pseudo_legal_moves}
        got = generated_ucis(board)
        castles = {m.uci() for m in board.pseudo_legal_moves if board.is_castling(m)}
        assert got - castles == expected - castles, f"mismatch in {board.fen()}"


def test_generate_captures_yields_only_captures_and_queen_promotions() -> None:
    for board in random_positions(count=120, seed=33):
        for uci in generated_ucis(board, captures_only=True):
            move = chess.Move.from_uci(uci)
            assert board.is_capture(move) or move.promotion == chess.QUEEN, uci


def test_generate_captures_finds_every_legal_capture() -> None:
    for board in random_positions(count=120, seed=34):
        expected = {m.uci() for m in board.legal_moves if board.is_capture(m)}
        assert expected <= generated_ucis(board, captures_only=True), board.fen()


def test_en_passant_capture_is_generated() -> None:
    board = chess.Board("8/8/8/8/4pP2/8/8/K6k b - f3 0 1")
    assert "e4f3" in generated_ucis(board)


def test_all_four_promotions_are_generated() -> None:
    board = chess.Board("8/4P3/8/8/8/8/8/K6k w - - 0 1")
    assert {"e7e8q", "e7e8r", "e7e8b", "e7e8n"} <= generated_ucis(board)


def test_castling_is_generated_when_available() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert {"e1g1", "e1c1"} <= generated_ucis(board)


def test_castling_through_an_attacked_square_is_not_generated() -> None:
    # Bishop on a6 attacks f1 along a6-b5-c4-d3-e2-f1. It does NOT attack e1, so the king
    # is not in check: kingside castling is illegal purely because f1 is crossed.
    board = chess.Board("4k3/8/b7/8/8/8/8/R3K2R w KQ - 0 1")
    generated = generated_ucis(board)
    assert "e1g1" not in generated, "f1 is attacked, so kingside castling is illegal"
    assert "e1c1" in generated, "queenside is unaffected and must still be generated"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_movegen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'movegen'`.

- [ ] **Step 3: Write the implementation**

Create `movegen.py`. Generate in this order, appending `int32` values into `moves` starting at `base`:

1. **Pawns**, branching once on side to move rather than per pawn. Single push when the target is empty, then double push only from the home rank when both squares are empty. Promotions expand to four moves with `flag = 3`. Captures come from `PAWN_ATT[side, frm] & them`. En passant is generated when `state[EP] != 0` and `PAWN_ATT[side, frm]` covers `state[EP] - 1`, with `flag = 1`.
2. **Knights**: `KNIGHT_ATT[frm] & ~us`.
3. **Bishops and queens**: `bishop_attacks(frm, occ) & ~us`.
4. **Rooks and queens**: `rook_attacks(frm, occ) & ~us`.
5. **King**: `KING_ATT[ksq] & ~us`.
6. **Castling**, with `flag = 2`, gated on the rights bit, the intervening squares being empty, and `attacked` returning false for the king's square and the square it crosses.

`generate_captures` is the same code restricted to `them` as the target set, plus queen promotions only.

`move_to_uci` maps a promotion index `1..4` to `"nbrq"[promo - 1]`.

Add a warm-up block at the bottom calling `generate` and `generate_captures` once on the start position.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_movegen.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add movegen.py tests/test_movegen.py
git commit -m "feat(movegen): add pseudo-legal generation and the move codec"
```

---

### Task 5: `make`, legality filtering and the perft gate

**Files:**
- Modify: `position.py`
- Create: `tests/test_perft.py`
- Create: `tests/test_make.py`

**Interfaces:**
- Consumes: Tasks 2 to 4.
- Produces:
  - `make(src_state: uint64[:], src_mail: int8[:], dst_state: uint64[:], dst_mail: int8[:], move: int32) -> None`, `njit`, copy-make.
  - `legal_after(dst_state: uint64[:], mover_black: int64) -> boolean`, `njit`, true when the side that just moved is not leaving its king in check.
  - `perft(state: uint64[:, :], mail: int8[:, :], moves: int32[:], ply: int64, depth: int64) -> int64`, `njit`, self-recursive with an explicit signature.

**Notes for the implementer, in order of how likely each is to bite:**

- **The en passant captured pawn is on `to + 8` when black is capturing and `to - 8` when white is.** Getting this backwards was the single bug behind all four perft mismatches in the design probe, and it presents as a wrong count only in positions with a horizontal pin, so it survives casual testing.
- Set `dst_state[EP]` to `(frm + to) // 2 + 1` after a double pawn push and to `0` otherwise. Write it on every path.
- Castling rights update as `dst[CASTLE] = src[CASTLE] & CASTLE_MASK[frm] & CASTLE_MASK[to]`. Masking on `to` is what handles a rook being captured on its home square.
- The halfmove clock resets on a pawn move or a capture and increments otherwise.
- Keep `mailbox` in step with the bitboards on every path, including the rook's move during castling and the removed pawn during en passant.
- Leave `dst_state[KEY]` alone for now. Task 6 adds incremental hashing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_make.py`:

```python
"""make() must produce exactly the state python-chess would."""

import chess
import numpy as np

import bitboard as bb
import movegen
import position
from tests.conftest import random_positions


def test_make_matches_python_chess_on_every_legal_move() -> None:
    state, mailbox = position.new_stacks()
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    for board in random_positions(count=150, seed=41):
        position.encode(board, state[0], mailbox[0])
        count = movegen.generate(state[0], moves, 0)
        for i in range(count):
            uci = movegen.move_to_uci(int(moves[i]))
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            position.make(state[0], mailbox[0], state[1], mailbox[1], moves[i])
            board.push(move)
            expected_state, expected_mail = position.new_stacks()
            position.encode(board, expected_state[0], expected_mail[0])
            board.pop()
            for field in (bb.PAWN, bb.KNIGHT, bb.BISHOP, bb.ROOK, bb.QUEEN, bb.KING,
                          bb.WOCC, bb.BOCC, bb.STM, bb.CASTLE, bb.EP, bb.HALF):
                assert state[1][field] == expected_state[0][field], (
                    f"field {field} wrong after {uci} in {board.fen()}"
                )
            assert list(mailbox[1]) == list(expected_mail[0]), f"mailbox wrong after {uci}"


def test_en_passant_removes_the_pawn_beside_the_target_not_behind_it() -> None:
    # Black plays e4xf3 e.p.; the captured white pawn stands on f4, which is to + 8.
    board = chess.Board("8/8/8/8/4pP2/8/8/K6k b - f3 0 1")
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    count = movegen.generate(state[0], moves, 0)
    move = next(moves[i] for i in range(count) if movegen.move_to_uci(int(moves[i])) == "e4f3")
    position.make(state[0], mailbox[0], state[1], mailbox[1], move)
    assert state[1][bb.PAWN] & (np.uint64(1) << np.uint64(chess.F4)) == 0, "f4 pawn must be gone"
    assert mailbox[1][chess.F4] == -1
    assert mailbox[1][chess.F3] == chess.PAWN - 1


def test_en_passant_that_exposes_the_king_is_rejected_by_the_legality_filter() -> None:
    # The classic horizontal pin: after e2e4, f4xe3 e.p. would expose the black king on h4.
    board = chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    board.push_uci("e2e4")
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    count = movegen.generate(state[0], moves, 0)
    move = next(moves[i] for i in range(count) if movegen.move_to_uci(int(moves[i])) == "f4e3")
    position.make(state[0], mailbox[0], state[1], mailbox[1], move)
    assert not position.legal_after(state[1], 1)
    assert chess.Move.from_uci("f4e3") not in board.legal_moves
```

Create `tests/test_perft.py`:

```python
"""Perft is the gate. One illegal move loses a game outright."""

import chess
import numpy as np
import pytest

import movegen
import position

CASES = [
    ("startpos", chess.STARTING_FEN, [20, 400, 8902, 197281, 4865609]),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [48, 2039, 97862, 4085603]),
    ("ep-pin", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238, 674624]),
    ("promotion", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
     [6, 264, 9467, 422333]),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [44, 1486, 62379, 2103487]),
    ("position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [46, 2079, 89890, 3894594]),
]


@pytest.mark.parametrize("name,fen,expected", CASES, ids=[c[0] for c in CASES])
def test_perft(name: str, fen: str, expected: list[int]) -> None:
    state, mailbox = position.new_stacks()
    moves = np.zeros(64 * movegen.MAX_MOVES, dtype=np.int32)
    for depth, want in enumerate(expected, start=1):
        position.encode(chess.Board(fen), state[0], mailbox[0])
        got = position.perft(state, mailbox, moves, 0, depth)
        assert got == want, f"{name} perft({depth}) = {got}, expected {want}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_perft.py tests/test_make.py -v`
Expected: FAIL with `AttributeError: module 'position' has no attribute 'make'`.

- [ ] **Step 3: Write the implementation**

Add `make`, `legal_after` and `perft` to `position.py`. `perft` needs an explicit signature for numba to accept the self-recursion; the probe confirmed this works:

```python
@njit(int64(uint64[:, :], int8[:, :], int32[:], int64, int64), cache=False)
def perft(state, mail, moves, ply, depth):
    if depth == 0:
        return 1
    base = ply * MAX_MOVES
    count = generate(state[ply], moves, base)
    mover_black = int64(state[ply][STM])
    total = 0
    for i in range(base, count):
        make(state[ply], mail[ply], state[ply + 1], mail[ply + 1], moves[i])
        if legal_after(state[ply + 1], mover_black):
            total += perft(state, mail, moves, ply + 1, depth - 1)
    return total
```

Warm `make`, `legal_after` and `perft` at import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_perft.py tests/test_make.py -v`
Expected: all pass. If any perft count is wrong, do not guess. Write a divide routine that compares your move list against `board.legal_moves` at each node and descends into the first child whose count disagrees; that isolates the exact position and move in seconds.

- [ ] **Step 5: Commit**

```bash
git add position.py tests/test_perft.py tests/test_make.py
git commit -m "feat(position): add copy-make, legality filtering and a perft gate"
```

---

### Task 6: Incremental Zobrist hashing

**Files:**
- Modify: `position.py`
- Create: `tests/test_zobrist.py`

**Interfaces:**
- Consumes: Task 5.
- Produces: `make` now maintains `dst_state[KEY]`. `full_key(state: uint64[:]) -> uint64`, `njit`, recomputes from scratch.

**Notes:** hash side to move, all four castling-rights bits as a single indexed key `Z_CASTLE[rights]`, and the en passant *file* only when an en passant capture is actually available. Hashing the ep square unconditionally makes otherwise identical positions hash differently and quietly destroys transposition table hit rates.

- [ ] **Step 1: Write the failing test**

Create `tests/test_zobrist.py`:

```python
"""The incrementally updated key must equal the from-scratch key, always."""

import chess
import numpy as np

import bitboard as bb
import movegen
import position
from tests.conftest import random_positions


def test_incremental_key_matches_full_recompute() -> None:
    state, mailbox = position.new_stacks()
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    for board in random_positions(count=200, seed=51):
        position.encode(board, state[0], mailbox[0])
        count = movegen.generate(state[0], moves, 0)
        for i in range(count):
            position.make(state[0], mailbox[0], state[1], mailbox[1], moves[i])
            assert state[1][bb.KEY] == position.full_key(state[1]), (
                f"key diverged after {movegen.move_to_uci(int(moves[i]))} in {board.fen()}"
            )


def test_transpositions_reach_the_same_key() -> None:
    left = chess.Board()
    for uci in ("g1f3", "g8f6", "d2d4", "d7d5"):
        left.push_uci(uci)
    right = chess.Board()
    for uci in ("d2d4", "d7d5", "g1f3", "g8f6"):
        right.push_uci(uci)
    state, mailbox = position.new_stacks()
    position.encode(left, state[0], mailbox[0])
    position.encode(right, state[1], mailbox[1])
    assert state[0][bb.KEY] == state[1][bb.KEY]


def test_en_passant_only_hashes_when_a_capture_is_available() -> None:
    # A double push with no enemy pawn able to take must hash the same as the quiet position.
    with_push = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    with_push.push_uci("e2e4")
    same_without_ep = chess.Board("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1")
    state, mailbox = position.new_stacks()
    position.encode(with_push, state[0], mailbox[0])
    position.encode(same_without_ep, state[1], mailbox[1])
    assert state[0][bb.KEY] == state[1][bb.KEY]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_zobrist.py -v`
Expected: FAIL, either `AttributeError: module 'position' has no attribute 'full_key'` or a key mismatch.

- [ ] **Step 3: Write the implementation**

Add `full_key` to `position.py` and extend `make` to xor out the moving piece at `frm`, xor in the piece at `to` (using the promotion type when promoting), xor out any captured piece, xor the rook's two squares when castling, xor `Z_CASTLE[old_rights] ^ Z_CASTLE[new_rights]`, xor `Z_STM[0]`, and xor `Z_EP[file]` for the old and new en passant states when and only when an enemy pawn could actually capture. Update `encode` to compute the key through the same helper so the two can never drift.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_zobrist.py tests/test_perft.py -v`
Expected: all pass. Re-running perft here confirms hashing did not disturb `make`.

- [ ] **Step 5: Commit**

```bash
git add position.py tests/test_zobrist.py
git commit -m "feat(position): maintain the Zobrist key incrementally in make"
```

---

### Task 7: Static exchange evaluation and position predicates

**Files:**
- Modify: `position.py`
- Create: `tests/test_see.py`

**Interfaces:**
- Consumes: Task 6.
- Produces:
  - `see(state: uint64[:], mail: int8[:], move: int32) -> int32`, `njit`, the centipawn outcome of the capture sequence on the destination square assuming both sides play optimally.
  - `has_non_pawn_material(state: uint64[:], black: int64) -> boolean`, `njit`.
  - `insufficient_material(state: uint64[:]) -> boolean`, `njit`.
  - `SEE_VALUE: int32[6]`, the values SEE uses: pawn 100, knight 320, bishop 330, rook 500, queen 900, king 20000.

**Notes:** implement the swap algorithm. Maintain an occupancy bitboard you clear as pieces are consumed, and after each capture re-derive the attackers of the target square so that x-ray attackers behind the captured piece are added. A SEE that ignores x-rays misjudges rook batteries and will prune winning captures.

- [ ] **Step 1: Write the failing test**

Create `tests/test_see.py`:

```python
"""SEE, checked against hand-computed exchanges."""

import chess
import numpy as np
import pytest

import movegen
import position


def see_of(fen: str, uci: str) -> int:
    board = chess.Board(fen)
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    moves = np.zeros(movegen.MAX_MOVES, dtype=np.int32)
    count = movegen.generate(state[0], moves, 0)
    move = next(moves[i] for i in range(count) if movegen.move_to_uci(int(moves[i])) == uci)
    return int(position.see(state[0], mailbox[0], move))


def test_free_pawn_capture_is_worth_a_pawn() -> None:
    # d5 is undefended, so the bishop just wins it.
    assert see_of("4k3/8/8/3p4/4B3/8/8/4K3 w - - 0 1", "e4d5") == 100


def test_defended_pawn_capture_by_a_bishop_loses_material() -> None:
    # Pawn on c6 defends d5. Bxd5 cxd5 is 100 - 330.
    assert see_of("4k3/8/2p5/3p4/4B3/8/8/4K3 w - - 0 1", "e4d5") == 100 - 330


def test_rook_takes_defended_rook_is_an_even_trade() -> None:
    # Rxe7+ Kxe7 is 500 - 500.
    assert see_of("4k3/4r3/8/8/8/8/4R3/4RK2 w - - 0 1", "e2e7") == 0


def test_x_ray_attacker_behind_the_capturer_is_counted() -> None:
    """The doubled rook on e1 is not an attacker of e4 until the e2 rook is consumed.

    python-chess's own `attackers()` does not list it, which is exactly why a SEE that
    only enumerates direct attackers gets this wrong and prunes a winning capture.
    Rxe4 Rxe4 Rxe4 leaves white a pawn up: 100 - 500 + 500.
    """
    assert see_of("4k3/4r3/8/8/4p3/8/4R3/4RK2 w - - 0 1", "e2e4") == 100


def test_same_capture_without_the_battery_loses_a_rook() -> None:
    # Only one white rook, so Rxe4 Rxe4 is 100 - 500. Compare with the test above:
    # the two positions differ only by the rook on e1.
    assert see_of("4k3/4r3/8/8/4p3/8/4R3/5K2 w - - 0 1", "e2e4") == 100 - 500


def test_has_non_pawn_material() -> None:
    state, mailbox = position.new_stacks()
    position.encode(chess.Board("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1"), state[0], mailbox[0])
    assert not position.has_non_pawn_material(state[0], 0)
    assert not position.has_non_pawn_material(state[0], 1)
    position.encode(chess.Board("4k3/4p3/8/8/8/8/4P3/3QK3 w - - 0 1"), state[0], mailbox[0])
    assert position.has_non_pawn_material(state[0], 0)
    assert not position.has_non_pawn_material(state[0], 1)


@pytest.mark.parametrize("fen,expected", [
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", True),
    ("4k3/8/8/8/8/8/8/3BK3 w - - 0 1", True),
    ("4k3/8/8/8/8/8/8/3NK3 w - - 0 1", True),
    ("4k3/8/8/8/8/8/8/3RK3 w - - 0 1", False),
    ("4k3/4p3/8/8/8/8/8/4K3 w - - 0 1", False),
])
def test_insufficient_material(fen: str, expected: bool) -> None:
    state, mailbox = position.new_stacks()
    position.encode(chess.Board(fen), state[0], mailbox[0])
    assert bool(position.insufficient_material(state[0])) is expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_see.py -v`
Expected: FAIL with `AttributeError: module 'position' has no attribute 'see'`.

- [ ] **Step 3: Write the implementation**

Add the three functions and `SEE_VALUE` to `position.py`, warming each at import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_see.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add position.py tests/test_see.py
git commit -m "feat(position): add static exchange evaluation and material predicates"
```

---

### Task 8: `tt.py` transposition table

**Files:**
- Create: `tt.py`
- Create: `tests/test_tt.py`

**Interfaces:**
- Consumes: nothing beyond numpy and numba.
- Produces:
  - `BOUND_NONE = 0`, `BOUND_UPPER = 1`, `BOUND_LOWER = 2`, `BOUND_EXACT = 3`.
  - `MATE = 30000`, `MATE_IN_MAX = MATE - 256`. These live here rather than in `search.py` because the rebasing logic in `tt_store` and `tt_probe` needs them and `search.py` imports `tt`, not the other way round. `search.py` re-exports them.
  - `TT: uint64[BUCKETS, 8]`, four entries per 64-byte bucket, sized to 128 MB.
  - `tt_probe(key: uint64, ply: int64) -> tuple[boolean, int32, int32, int64, int64, int32]` returning `(hit, score, move, depth, bound, static_eval)`, `njit`.
  - `tt_store(key: uint64, ply: int64, score: int32, move: int32, depth: int64, bound: int64, static_eval: int32, age: int64) -> None`, `njit`.
  - `tt_clear() -> None`, `njit`.

**Notes:** an entry packs into one `uint64` as `(score + 32768) & 0xFFFF`, `move << 16`, `depth << 32`, `(bound | age << 2) << 40`, `(static_eval + 32768) << 48`. **Mate scores must be stored relative to the node, not the root.** On store, add `ply` to a score above `MATE_IN_MAX` and subtract `ply` from one below `-MATE_IN_MAX`; on probe, do the reverse. Skipping this produces mate lines that are off by a few moves and occasionally loops the engine.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tt.py`:

```python
"""Transposition table storage, replacement and mate-score rebasing."""

import numpy as np

import tt


def setup_function() -> None:
    tt.tt_clear()


def test_probe_misses_on_an_empty_table() -> None:
    hit, _, _, _, _, _ = tt.tt_probe(np.uint64(0x1234), 0)
    assert not hit


def test_store_then_probe_round_trips() -> None:
    key = np.uint64(0xDEADBEEFCAFEF00D)
    tt.tt_store(key, 0, np.int32(-137), np.int32(1234), 7, tt.BOUND_EXACT, np.int32(42), 1)
    hit, score, move, depth, bound, static = tt.tt_probe(key, 0)
    assert hit
    assert score == -137
    assert move == 1234
    assert depth == 7
    assert bound == tt.BOUND_EXACT
    assert static == 42


def test_different_keys_do_not_collide_within_a_bucket() -> None:
    for i in range(4):
        tt.tt_store(np.uint64(0x1000 + i), 0, np.int32(i), np.int32(i), 5, tt.BOUND_EXACT,
                    np.int32(0), 1)
    for i in range(4):
        hit, score, _, _, _, _ = tt.tt_probe(np.uint64(0x1000 + i), 0)
        assert hit and score == i


def test_mate_scores_are_stored_relative_to_the_node() -> None:
    key = np.uint64(0xABCD)
    mate_at_root = np.int32(tt.MATE - 10)
    # Stored at ply 4, so the entry holds a mate that is 4 plies nearer from there.
    tt.tt_store(key, 4, mate_at_root, np.int32(0), 9, tt.BOUND_EXACT, np.int32(0), 1)
    hit, score, _, _, _, _ = tt.tt_probe(key, 4)
    assert hit and score == mate_at_root

    hit, score, _, _, _, _ = tt.tt_probe(key, 0)
    assert hit and score == mate_at_root + 4, "a mate found deeper is nearer when probed shallower"


def test_deeper_entry_is_not_replaced_by_a_shallower_one_of_the_same_age() -> None:
    key = np.uint64(0x5555)
    tt.tt_store(key, 0, np.int32(100), np.int32(11), 12, tt.BOUND_EXACT, np.int32(0), 1)
    tt.tt_store(key, 0, np.int32(200), np.int32(22), 3, tt.BOUND_EXACT, np.int32(0), 1)
    hit, score, move, depth, _, _ = tt.tt_probe(key, 0)
    assert hit and depth == 12 and score == 100 and move == 11


def test_clear_empties_the_table() -> None:
    tt.tt_store(np.uint64(0x99), 0, np.int32(1), np.int32(1), 1, tt.BOUND_EXACT, np.int32(0), 1)
    tt.tt_clear()
    hit, _, _, _, _, _ = tt.tt_probe(np.uint64(0x99), 0)
    assert not hit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tt'`.

- [ ] **Step 3: Write the implementation**

Create `tt.py`. Bucket index is `key & (BUCKETS - 1)` with `BUCKETS` a power of two. Store the full key in the even slots and the packed data in the odd slots of each bucket row. Replacement picks, in order: a slot whose key matches, an empty slot, then the slot minimising `depth - 2 * age_gap`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tt.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tt.py tests/test_tt.py
git commit -m "feat(tt): add a bucketed transposition table with mate-score rebasing"
```

---

### Task 9: Port `evaluate.py` to the jitted contract

**Files:**
- Modify: `evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `bitboard.py`, `position.py`.
- Produces: `evaluate(state: uint64[:], mailbox: int8[:]) -> int32`, `njit`, centipawns, positive means the side to move is better.

**Notes:** this task changes the signature and ports the existing material term. It does not improve the evaluation, which is the user's work. Delete `evaluate_pure_python` and the old `chess.Board` entry point along with `material_score` and the local `popcount`, since `bitboard.popcount` replaces it. The checkmate special case moves out of the evaluation entirely: the search detects having no legal moves and scores it, so an evaluation that inspects for checkmate would be both slow and wrong at a node the search already handled.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
"""The evaluation contract. Content is the user's to change; the contract is not."""

import chess

import evaluate
import position
from tests.conftest import random_positions


def evaluate_board(board: chess.Board) -> int:
    state, mailbox = position.new_stacks()
    position.encode(board, state[0], mailbox[0])
    return int(evaluate.evaluate(state[0], mailbox[0]))


def test_start_position_is_balanced() -> None:
    assert evaluate_board(chess.Board()) == 0


def test_score_is_relative_to_the_side_to_move() -> None:
    white_up_a_queen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    assert evaluate_board(chess.Board(white_up_a_queen)) == 900
    black_to_move = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"
    assert evaluate_board(chess.Board(black_to_move)) == -900


def test_evaluation_is_symmetric_under_colour_flip() -> None:
    for board in random_positions(count=80, seed=61):
        mirrored = board.mirror()
        assert evaluate_board(board) == evaluate_board(mirrored), board.fen()


def test_returns_a_plain_int32() -> None:
    state, mailbox = position.new_stacks()
    position.encode(chess.Board(), state[0], mailbox[0])
    assert isinstance(int(evaluate.evaluate(state[0], mailbox[0])), int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL, because `evaluate.evaluate` still takes a `chess.Board`.

- [ ] **Step 3: Write the implementation**

Rewrite `evaluate.py`:

```python
"""Evaluation. Centipawns, positive means the side to move is better.

The search calls this at leaf nodes from jitted code, so it must stay njit-compatible.
When this becomes an NNUE, the accumulator lives in the state vector's spare fields and
this function reads it. torch and onnxruntime must not appear here: their per-call
overhead of 20 to 50 us is larger than the entire budget for a node.
"""

import numpy as np
from numba import int32, int8, njit, uint64

from bitboard import BISHOP, BOCC, KNIGHT, PAWN, QUEEN, ROOK, STM, WOCC, popcount

PIECE_VALUE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)


@njit(int32(uint64[:], int8[:]), cache=False)
def evaluate(state, mailbox):
    white = state[WOCC]
    black = state[BOCC]
    score = 0
    for piece in range(5):
        score += PIECE_VALUE[piece] * (
            popcount(state[piece] & white) - popcount(state[piece] & black)
        )
    return int32(-score if state[STM] else score)
```

Add a warm-up call at the bottom of the module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: all pass.

Run: `uv run mypy` and `uv run ruff check .`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "refactor(evaluate): move to the jitted state-vector contract"
```

---

### Task 10: `search.py` core and `agent.py` integration

This is the largest task. It replaces `search.py` entirely and makes the engine play.

**Files:**
- Rewrite: `search.py`
- Rewrite: `agent.py`
- Create: `tests/test_search.py`
- Create: `tests/bench.py`

**Interfaces:**
- Consumes: Tasks 2 to 9.
- Produces:
  - Module-level globals in `search.py`, all numpy arrays because numba freezes global scalars but reads array elements dynamically:
    - `STATE: uint64[STACK_PLIES, NFIELDS]`, `MAIL: int8[STACK_PLIES, 64]`, `MOVES: int32[STACK_PLIES * MAX_MOVES]`
    - `KILLERS: int32[STACK_PLIES, 2]`, `HISTORY: int32[2, 64, 64]`, `COUNTER: int32[2, 64, 64]`
    - `HISTORY_KEYS: uint64[1024]`, `HISTORY_LEN: int64[1]` for pre-root repetition
    - `FLAGS: int64[16]` with `F_NO_PRUNING = 0`, `F_NODE_LIMIT = 1`
    - `NODES: int64[1]`, `DEADLINE: float64[2]` holding hard then soft, `ABORT: uint8[1]`
    - `STATS: int64[8]` holding depth reached and seldepth for reporting
  - `negamax(ply: int64, depth: int64, alpha: int32, beta: int32, is_pv: boolean) -> int32`, `njit`, self-recursive with an explicit signature.
  - `qsearch(ply: int64, alpha: int32, beta: int32) -> int32`, `njit`.
  - `search_root(max_depth: int64) -> int32`, `njit`, returning the best move.
  - `think(board: chess.Board, time_left_ms: int, increment_ms: int = 500, node_limit: int = 0, max_depth: int = 127) -> str` in plain Python, returning UCI.
  - `set_game_history(keys: list[int]) -> None` in plain Python.
  - `search_value(board: chess.Board, depth: int) -> int` in plain Python, returning the score rather than the move. Used by the oracle tests.
  - `set_pruning(enabled: bool) -> None` in plain Python, writing `FLAGS[F_NO_PRUNING]`. Used by the oracle tests.
  - `clear_tables() -> None` in plain Python, zeroing `HISTORY`, `COUNTER` and `KILLERS`.
  - `budget_ms(time_left_ms: int, increment_ms: int) -> tuple[float, float]` in plain Python, returning soft then hard.
  - Re-exports `MATE`, `MATE_IN_MAX` from `tt`, and defines `INF = 32000`.
  - A terminal position (checkmate or stalemate at the root) must return a score from `search_value` rather than raising. `think` is never called on one, because the referee ends the game first.

**Notes for the implementer:**

- **`check_time` uses `objmode`,** which is the only way to read a clock from jitted code. numba does not support `time.time()` in nopython mode; this was verified. Call it every 2048 nodes:

```python
@njit(cache=False)
def check_time():
    if NODES[0] & 2047 != 0:
        return
    if FLAGS[F_NODE_LIMIT] != 0 and NODES[0] >= FLAGS[F_NODE_LIMIT]:
        ABORT[0] = 1
        return
    with objmode(now="f8"):
        now = time.time()
    if now >= DEADLINE[0]:
        ABORT[0] = 1
```

- **Never act on an aborted search's result.** `search_root` keeps the best move from the last *completed* iteration and only adopts a partial iteration's move if that move was already proven better than the previous iteration's best.
- **Depth 1 must always complete** before `ABORT` is honoured, so a legal move always exists.
- **Mate scores are ply-adjusted:** a checkmate at `ply` scores `-MATE + ply`, so shallower mates are preferred.
- **Repetition:** a position whose key appears anywhere in the current search path or in `HISTORY_KEYS` scores `0`. One repetition is enough inside the search; requiring three loses far more than it gains.
- **The transposition table is not cleared between moves.** It is cleared once per game, in `agent.py`, when the incoming FEN does not chain onto the remembered history.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search.py`:

```python
"""Search invariants. These are what tell a broken heuristic from a weak one."""

import chess

import search


def best_move(fen: str, depth: int = 6, nodes: int = 0) -> str:
    return search.think(chess.Board(fen), time_left_ms=60_000, node_limit=nodes, max_depth=depth)


def test_returns_a_legal_move_from_many_positions() -> None:
    from tests.conftest import random_positions

    for board in random_positions(count=40, seed=71):
        uci = search.think(board, time_left_ms=2_000, max_depth=4)
        assert chess.Move.from_uci(uci) in board.legal_moves, board.fen()


MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"           # Ra8 is mate
MATE_IN_ONE_CAPTURE = "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"  # Rxd8 is mate


def test_finds_mate_in_one() -> None:
    assert best_move(MATE_IN_ONE, depth=3) == "a1a8"


def test_finds_mate_in_one_by_capture() -> None:
    assert best_move(MATE_IN_ONE_CAPTURE, depth=3) == "d1d8"


def test_mate_score_is_ply_adjusted() -> None:
    """The bookkeeping that goes wrong silently. A mate in one scores MATE - 1."""
    assert search.search_value(chess.Board(MATE_IN_ONE), depth=3) == search.MATE - 1


def test_being_mated_scores_negative_mate() -> None:
    mated = chess.Board(MATE_IN_ONE)
    mated.push_uci("a1a8")
    assert search.search_value(mated, depth=3) == -search.MATE


def test_takes_a_free_queen() -> None:
    assert best_move("4k3/8/8/3q4/4B3/8/8/4K3 w - - 0 1", depth=4) == "e4d5"


def test_does_not_hang_a_queen_to_a_pawn() -> None:
    # The pawn on c6 attacks b5 and d5 but not d4, so the queen is safe where it stands
    # and d4d5 would simply hang it.
    uci = best_move("4k3/8/2p5/8/3Q4/8/8/4K3 w - - 0 1", depth=6)
    assert uci != "d4d5", "moved the queen onto a square the c6 pawn attacks"


def test_pruning_disabled_search_equals_plain_alpha_beta() -> None:
    """The oracle. Every heuristic reshapes the tree but must not change the value."""
    from tests.conftest import random_positions

    search.set_pruning(False)
    try:
        for board in random_positions(count=8, seed=72, max_plies=12):
            got = search.search_value(board, depth=4)
            want = reference_alpha_beta(board, 4, -search.INF, search.INF)
            assert got == want, f"{got} != {want} in {board.fen()}"
    finally:
        search.set_pruning(True)


def reference_alpha_beta(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """Deliberately naive. Slow, obviously correct, and the only thing we trust."""
    import evaluate
    import position

    if board.is_checkmate():
        return -search.MATE + (4 - depth)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth == 0:
        state, mailbox = position.new_stacks()
        position.encode(board, state[0], mailbox[0])
        return int(evaluate.evaluate(state[0], mailbox[0]))
    best = -search.INF
    for move in board.legal_moves:
        board.push(move)
        value = -reference_alpha_beta(board, depth - 1, -beta, -alpha)
        board.pop()
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


def test_node_limit_is_respected() -> None:
    search.think(chess.Board(), time_left_ms=60_000, node_limit=50_000, max_depth=127)
    assert search.NODES[0] <= 55_000, "node limit should stop the search promptly"


def test_node_limited_search_is_deterministic() -> None:
    first = best_move(chess.STARTING_FEN, depth=127, nodes=100_000)
    second = best_move(chess.STARTING_FEN, depth=127, nodes=100_000)
    assert first == second


def test_respects_a_tight_clock() -> None:
    import time

    started = time.perf_counter()
    search.think(chess.Board(), time_left_ms=1_000, max_depth=127)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.9, f"took {elapsed:.2f}s of a 1.0s clock, which risks a flag"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL with `AttributeError: module 'search' has no attribute 'think'`.

- [ ] **Step 3: Write the implementation**

Rewrite `search.py` with, in order: constants, global arrays, `check_time`, `is_repetition`, move ordering scoring, `qsearch`, `negamax`, `search_root`, `set_pruning`, `search_value`, `think`, and a warm-up block.

`negamax` in this task implements only: mate-distance pruning, TT probe and cutoff, draw detection, quiescence at `depth <= 0`, PVS with a null-window re-search, staged ordering by TT move then MVV-LVA captures then killers then history, and TT store. **The whole pruning schedule arrives in Tasks 12 and 13**, gated on `FLAGS[F_NO_PRUNING] == 0` from the moment it is added, so the oracle test keeps working.

`search_root` runs iterative deepening with aspiration windows from depth 5 at plus or minus 18 centipawns, widening asymmetrically through 72 and 288 to infinity.

`think` computes the budget, encodes the board, resets `NODES` and `ABORT`, sets `DEADLINE`, calls `search_root`, and returns UCI:

```python
RESERVE_MS = 300.0

def budget_ms(time_left_ms: int, increment_ms: int) -> tuple[float, float]:
    usable = max(float(time_left_ms) - RESERVE_MS, 10.0)
    soft = usable / 22.0 + 0.75 * increment_ms
    hard = min(usable / 4.0, soft * 5.0)
    return min(soft, hard), hard
```

Rewrite `agent.py`:

```python
"""The submission entrypoint. The platform imports this file and calls get_move."""

import io
import sys

import chess

import search

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)

_history: list[int] = []
_expected: dict[int, None] = {}


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    _track(board)
    uci = search.think(board, time_left_ms, increment_ms=500)
    move = _validated(board, uci)
    _remember_reply(board, move)
    return move.uci()


def _validated(board: chess.Board, uci: str) -> chess.Move:
    """Never lose a game to a malformed move. This should never fire."""
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        move = chess.Move.null()
    if move in board.legal_moves:
        return move
    print(f"engine produced an unplayable move {uci!r}; falling back")
    return next(iter(board.legal_moves))
```

`_track` appends the position's key, resetting `_history` and calling `tt.tt_clear()` when the incoming position does not chain onto the remembered one. `_remember_reply` pushes our own move and appends the resulting key, so the history covers positions the platform never shows us. Both call `search.set_game_history(_history)`.

Create `tests/bench.py` as a runnable script printing nodes, depth and nps for a fixed set of positions, and asserting that importing the engine stays under 30 seconds.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -v`
Expected: all pass.

Run: `make gate`
Expected: ruff clean, mypy clean, two games finish cleanly.

Run: `make arena` and `uv run python tests/bench.py`
Expected: a large score against `greedy`, and a recorded nps figure to compare against later.

- [ ] **Step 5: Commit**

```bash
git add search.py agent.py tests/test_search.py tests/bench.py
git commit -m "feat(search): replace the negamax with a jitted PVS, TT and quiescence search"
```

---

### Task 11: A/B match harness and snapshots

**Files:**
- Create: `tests/match.py`
- Create: `tests/openings.py`
- Modify: `Makefile` if the `ab` target needs adjusting

**Interfaces:**
- Consumes: `harness.referee.play_match`, `harness.sandbox.local`.
- Produces: `tests/match.py` as a CLI, and `snapshot(tag: str) -> Path` copying the shipped root `*.py` files into `snapshots/<tag>/`.

**Notes:** `harness/arena.py` never passes `start_fen` to `play_match`, so it replays one opening and cannot measure a deterministic engine. This harness passes a varied opening set instead and alternates colours within each opening. Do not edit `harness/`.

Report a score with a confidence interval, because a bare percentage over 200 games invites reading noise as progress. Use the normal approximation on the win/draw/loss trinomial, and print the Elo difference with its interval.

- [ ] **Step 1: Write the failing test**

Create `tests/test_match.py`:

```python
"""The A/B harness itself, tested without playing real games."""

import chess

from tests.match import elo_with_interval
from tests.openings import OPENINGS


def test_openings_are_varied_legal_and_balanced() -> None:
    assert len(OPENINGS) >= 20
    assert len(set(OPENINGS)) == len(OPENINGS)
    for fen in OPENINGS:
        board = chess.Board(fen)
        assert board.is_valid()
        assert not board.is_game_over()


def test_even_score_is_zero_elo() -> None:
    elo, low, high = elo_with_interval(wins=50, draws=0, losses=50)
    assert elo == 0
    assert low < 0 < high


def test_winning_score_is_positive_elo() -> None:
    elo, low, high = elo_with_interval(wins=70, draws=0, losses=30)
    assert elo > 100
    assert low > 0


def test_interval_narrows_with_more_games() -> None:
    _, low_small, high_small = elo_with_interval(wins=14, draws=0, losses=6)
    _, low_big, high_big = elo_with_interval(wins=700, draws=0, losses=300)
    assert (high_big - low_big) < (high_small - low_small)


def test_draws_count_as_half() -> None:
    elo, _, _ = elo_with_interval(wins=0, draws=100, losses=0)
    assert elo == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_match.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.match'`.

- [ ] **Step 3: Write the implementation**

Create `tests/openings.py` holding a fixed list of at least 20 FENs. Produce them once with the snippet below, paste the printed list into the module as a literal, and never regenerate it, because A/B runs are only comparable when every version played the same openings.

```python
# Run once, paste the output into tests/openings.py as OPENINGS = [...]
import chess

LINES = [
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5", "d2d4 d7d5 c2c4 e7e6", "d2d4 g8f6 c2c4 e7e6",
    "d2d4 g8f6 c2c4 g7g6", "c2c4 e7e5 b1c3 g8f6", "g1f3 d7d5 g2g3 g8f6",
    "e2e4 d7d5 e4d5 g8f6", "d2d4 d7d5 g1f3 g8f6", "e2e4 g8f6 e4e5 f6d5",
    "d2d4 f7f5 g2g3 g8f6", "e2e4 d7d6 d2d4 g8f6", "c2c4 c7c5 g1f3 g8f6",
    "d2d4 e7e6 c2c4 f8b4", "e2e4 b8c6 d2d4 d7d5", "g1f3 g8f6 c2c4 c7c5",
    "d2d4 d7d6 e2e4 g8f6", "e2e4 e7e5 f1c4 g8f6", "d2d4 c7c5 d4d5 e7e6",
    "e2e4 e7e5 b1c3 g8f6",
]
seen: list[str] = []
for line in LINES:
    board = chess.Board()
    for uci in line.split():
        board.push_uci(uci)
    fen = board.fen()
    if fen not in seen:          # two of these lines transpose into the same position
        seen.append(fen)
for fen in seen:
    print(f'    "{fen}",')
print(f"# {len(seen)} openings")
```

These 22 lines were verified legal and produce 21 distinct positions, which satisfies the
`len(set(OPENINGS)) == len(OPENINGS)` assertion in the test only because of the dedup above.
Do not drop it.

Create `tests/match.py` with `elo_with_interval(wins, draws, losses) -> tuple[float, float, float]` using `elo = -400 * log10(1 / score - 1)` with the score's standard error propagated, and a `main()` accepting `--agent`, `--opponent`, `--games`, `--nodes`, `--base-ms`, `--increment-ms`. When `--nodes` is set it exports `CHESSATHON_NODE_LIMIT` into the agent process environment so both sides search a fixed node count and the result is immune to machine load.

That environment variable must be read in `search.think` and mapped onto `node_limit`. Add that line in this task.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_match.py -v`
Expected: all pass.

Run: `uv run python tests/match.py --snapshot baseline` then `make ab OPPONENT=snapshots/baseline GAMES=40`
Expected: a score near 50% against a copy of itself, confirming the harness is unbiased.

- [ ] **Step 5: Commit**

```bash
git add tests/match.py tests/openings.py tests/test_match.py search.py Makefile
git commit -m "test: add a varied-opening A/B match harness with Elo intervals"
```

---

### Task 12: Whole-node pruning

**Files:**
- Modify: `search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes: Task 11.
- Produces: reverse futility, null move and razoring inside `negamax`, all gated on `FLAGS[F_NO_PRUNING] == 0`.

**Notes:** null move must not run when the side to move has no non-pawn material, or the engine walks into zugzwang. `position.has_non_pawn_material` exists for exactly this. Add a verification search at depth 10 and above, re-searching without null move, so deep zugzwang is caught.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_search.py`:

```python
def test_null_move_does_not_break_zugzwang_positions() -> None:
    # White to move must not believe a null move is safe here.
    uci = best_move("8/8/8/8/8/6k1/6p1/6K1 w - - 0 1", depth=10)
    assert chess.Move.from_uci(uci) in chess.Board("8/8/8/8/8/6k1/6p1/6K1 w - - 0 1").legal_moves


def test_pruning_still_agrees_with_plain_alpha_beta_when_disabled() -> None:
    """Re-run the oracle now that whole-node pruning exists."""
    from tests.conftest import random_positions

    search.set_pruning(False)
    try:
        for board in random_positions(count=8, seed=81, max_plies=12):
            assert search.search_value(board, depth=4) == reference_alpha_beta(
                board, 4, -search.INF, search.INF
            )
    finally:
        search.set_pruning(True)


def test_pruning_finds_the_same_mates_as_full_width() -> None:
    assert best_move("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", depth=5) == "a1a8"
```

- [ ] **Step 2: Establish the guard before changing anything**

Run: `uv run pytest tests/test_search.py -v`
Expected: everything passes, including the oracle. This task's deliverable is measured in Elo rather than in a newly passing test, so the oracle passing *before* the change is what makes it meaningful *after*. Record the bench figure now:

Run: `uv run python tests/bench.py` and `uv run python tests/match.py --snapshot pre-nullmove`

- [ ] **Step 3: Write the implementation**

Add to `negamax`, before the move loop and only when not in check, not a PV node, and `FLAGS[F_NO_PRUNING] == 0`:

- Reverse futility: `depth <= 8` and `static_eval - 75 * depth >= beta` returns `static_eval`.
- Null move: `depth >= 3`, `static_eval >= beta`, `has_non_pawn_material(state, stm)`. Reduction `R = 3 + depth // 4 + min((static_eval - beta) // 200, 3)`. Make a null move by copying the state, flipping `STM`, clearing `EP`, and xoring `Z_STM[0]` into the key. If the null search returns at least beta, return beta, running a verification search without null move first when `depth >= 10`.
- Razoring: `depth <= 3` and `static_eval + 200 * depth < alpha`, drop to `qsearch` and return its value if it is still at or below alpha.

- [ ] **Step 4: Run tests and measure**

Run: `uv run pytest tests/test_search.py -v`
Expected: all pass, including the oracle.

Run: `make ab OPPONENT=snapshots/pre-nullmove GAMES=400 NODES=200000` against the snapshot taken in Step 2.
Expected: a clearly positive Elo interval. If the interval spans zero, the technique is not working and should be debugged, not kept.

Record the measured Elo in the commit message.

- [ ] **Step 5: Commit**

```bash
git add search.py tests/test_search.py
git commit -m "feat(search): add reverse futility, null move and razoring"
```

---

### Task 13: Move-loop pruning and reductions

**Files:**
- Modify: `search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes: Task 12.
- Produces: futility pruning, late move pruning, SEE pruning, late move reductions, internal iterative reduction and check extensions, all gated on `FLAGS[F_NO_PRUNING] == 0`. Adds `LMR_TABLE: int64[64, 64]` built at import.

**Notes:** the LMR re-search is where implementations go wrong. When a reduced search returns a value above alpha, the move must be re-searched at full depth before its score is trusted. Skipping the re-search produces an engine that looks fine in tests and quietly plays bad moves.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_search.py`:

```python
def test_lmr_table_is_monotonic() -> None:
    for depth in range(3, 40):
        for index in range(3, 40):
            assert search.LMR_TABLE[depth][index] >= search.LMR_TABLE[depth][index - 1], (
                "later moves should be reduced at least as much as earlier ones"
            )
        assert search.LMR_TABLE[depth][10] >= search.LMR_TABLE[depth - 1][10]


def test_reductions_never_reduce_below_one_ply() -> None:
    for depth in range(1, 64):
        for index in range(64):
            assert depth - search.LMR_TABLE[depth][index] >= 1


def test_oracle_still_holds_with_the_full_pruning_schedule() -> None:
    from tests.conftest import random_positions

    search.set_pruning(False)
    try:
        for board in random_positions(count=8, seed=91, max_plies=12):
            assert search.search_value(board, depth=4) == reference_alpha_beta(
                board, 4, -search.INF, search.INF
            )
    finally:
        search.set_pruning(True)


def test_deep_search_still_finds_forced_mates() -> None:
    assert best_move(MATE_IN_ONE, depth=8) == "a1a8"
    assert best_move(MATE_IN_ONE_CAPTURE, depth=8) == "d1d8"
    assert search.search_value(chess.Board(MATE_IN_ONE), depth=8) == search.MATE - 1


def test_winning_capture_is_not_pruned_away() -> None:
    """SEE pruning and LMP must never discard a capture that simply wins material."""
    assert best_move("4k3/8/8/3q4/4B3/8/8/4K3 w - - 0 1", depth=9) == "e4d5"


def test_deeper_search_is_never_worse_at_finding_a_free_queen() -> None:
    for depth in range(4, 12):
        assert best_move("4k3/8/8/3q4/4B3/8/8/4K3 w - - 0 1", depth=depth) == "e4d5", (
            f"lost the free queen at depth {depth}, so a reduction is not being re-searched"
        )
```

- [ ] **Step 2: Run tests to verify the oracle guards the change**

Run: `uv run pytest tests/test_search.py::test_oracle_still_holds_with_the_full_pruning_schedule -v`
Expected: PASS before the change, and it must still pass after.

- [ ] **Step 3: Write the implementation**

Build `LMR_TABLE` at import with `int(0.75 + log(depth) * log(index) / 2.25)`, clamped so `depth - reduction >= 1`.

Inside the move loop, gated on `FLAGS[F_NO_PRUNING] == 0` and skipped for PV nodes, captures, promotions and moves giving check as appropriate:

- Futility: `depth <= 6`, quiet move, `static_eval + 100 + 90 * depth <= alpha`, skip remaining quiets.
- Late move pruning: `depth <= 8`, quiet move index above `3 + depth * depth`, halved when not improving.
- SEE pruning: `depth <= 8`, skip quiets with `see < -50 * depth` and captures with `see < -100 * depth`.
- Late move reductions: `depth >= 3` and move index at least 3. Take `LMR_TABLE[depth][index]`, subtract one on a PV node, subtract one when in check or giving check, subtract one for a killer or counter-move, add one when not improving, and adjust by `history // 4000`. Search at the reduced depth with a null window; **if that returns above alpha, re-search at full depth.**
- Internal iterative reduction: `depth >= 4` and no TT move, reduce depth by one.
- Check extension: extend by one ply when in check, with a cap on total extensions along a line.

- [ ] **Step 4: Run tests and measure**

Run: `uv run pytest -q`
Expected: everything passes.

Run: `uv run python tests/match.py --snapshot pre-lmr` before, then `make ab OPPONENT=snapshots/pre-lmr GAMES=400 NODES=200000`.
Expected: a clearly positive Elo interval. LMR is normally the single largest gain in this list.

Run: `uv run python tests/bench.py`
Expected: a much larger depth at the same node count than Task 10 recorded.

- [ ] **Step 5: Commit**

```bash
git add search.py tests/test_search.py
git commit -m "feat(search): add futility, LMP, SEE pruning, LMR, IIR and check extensions"
```

---

### Task 14: Singular extensions

**Files:**
- Modify: `search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes: Task 13.
- Produces: `negamax` gains an `excluded: int32` parameter. Its signature becomes `negamax(ply: int64, depth: int64, alpha: int32, beta: int32, is_pv: boolean, excluded: int32) -> int32`. Every existing call site passes `0`.

**Notes:** a node searching with `excluded != 0` must not probe or store the transposition table under the plain key, or the exclusion leaks into unrelated searches. Either skip the TT entirely at such nodes or mix the excluded move into the key.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_search.py`:

```python
def test_excluded_move_is_not_played() -> None:
    board = chess.Board(MATE_IN_ONE)
    assert search.best_move_excluding(board, depth=4, excluded_uci="a1a8") != "a1a8"


def test_singular_extension_still_finds_mate() -> None:
    assert best_move(MATE_IN_ONE, depth=10) == "a1a8"


def test_oracle_holds_with_singular_extensions() -> None:
    from tests.conftest import random_positions

    search.set_pruning(False)
    try:
        for board in random_positions(count=8, seed=101, max_plies=12):
            assert search.search_value(board, depth=4) == reference_alpha_beta(
                board, 4, -search.INF, search.INF
            )
    finally:
        search.set_pruning(True)


def test_singular_search_does_not_corrupt_the_transposition_table() -> None:
    """A second search from the same position must agree with the first."""
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    first = best_move(fen, depth=8)
    second = best_move(fen, depth=8)
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py -k singular -v`
Expected: FAIL with `AttributeError: module 'search' has no attribute 'best_move_excluding'`.

- [ ] **Step 3: Write the implementation**

Add the `excluded` parameter and, inside `negamax` before the move loop, when `depth >= 8`, `excluded == 0`, a TT move exists, `tt_depth >= depth - 3`, the bound is lower or exact, and the TT score is not a mate score:

```
margin = 2 * depth
singular_beta = tt_score - margin
value = negamax(ply, (depth - 1) // 2, singular_beta - 1, singular_beta, False, tt_move)
if value < singular_beta:
    extension = 1          # the TT move is singular, search it deeper
elif singular_beta >= beta:
    return singular_beta   # multi-cut
```

Skip the TT move itself in the move loop when `move == excluded`.

Add `best_move_excluding(board, depth, excluded_uci) -> str` as a thin Python wrapper for the test.

- [ ] **Step 4: Run tests and measure**

Run: `uv run pytest -q`
Expected: all pass.

Run: `uv run python tests/match.py --snapshot pre-singular` before, then `make ab OPPONENT=snapshots/pre-singular GAMES=600 NODES=200000`.
Expected: a positive but smaller interval than LMR gave. Singular extensions are typically worth 15 to 30 Elo, so 600 games is the minimum that will resolve it. **If the interval spans zero, revert rather than keep it**, because a wrong singular implementation costs time everywhere.

- [ ] **Step 5: Commit**

```bash
git add search.py tests/test_search.py
git commit -m "feat(search): add singular extensions with multi-cut"
```

---

### Task 15: Continuation history

**Files:**
- Modify: `search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes: Task 14.
- Produces: `CONT_HIST: int32[2, 6, 64, 6, 64]` for the one-ply and two-ply continuations, and `MOVED_PIECE: int8[STACK_PLIES]`, `MOVED_TO: int8[STACK_PLIES]` recording the move played into each ply.

**Notes:** the first index selects the one-ply or two-ply table. Continuation history contributes to the ordering score for quiet moves and to the LMR reduction adjustment, exactly as plain history does. Update it wherever `HISTORY` is updated so the two never drift.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_search.py`:

```python
def test_continuation_history_is_updated_on_a_cutoff() -> None:
    search.clear_tables()
    search.think(chess.Board(), time_left_ms=5_000, max_depth=6)
    assert search.CONT_HIST.any(), "continuation history should be populated after a search"


def test_clear_tables_resets_history() -> None:
    search.think(chess.Board(), time_left_ms=3_000, max_depth=5)
    search.clear_tables()
    assert not search.CONT_HIST.any()
    assert not search.HISTORY.any()


def test_oracle_holds_with_continuation_history() -> None:
    from tests.conftest import random_positions

    search.set_pruning(False)
    try:
        for board in random_positions(count=8, seed=111, max_plies=12):
            assert search.search_value(board, depth=4) == reference_alpha_beta(
                board, 4, -search.INF, search.INF
            )
    finally:
        search.set_pruning(True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py -k continuation -v`
Expected: FAIL with `AttributeError: module 'search' has no attribute 'CONT_HIST'`.

- [ ] **Step 3: Write the implementation**

Add the arrays, record `MOVED_PIECE` and `MOVED_TO` in the move loop, add the continuation scores into quiet move ordering, feed them into the LMR adjustment, and update them alongside `HISTORY` on cutoffs and failures. Add `clear_tables()` in Python clearing `HISTORY`, `COUNTER`, `KILLERS` and `CONT_HIST`.

- [ ] **Step 4: Run tests and measure**

Run: `uv run pytest -q`
Expected: all pass.

Run: `uv run python tests/match.py --snapshot pre-conthist` before, then `make ab OPPONENT=snapshots/pre-conthist GAMES=600 NODES=200000`.
Expected: a positive interval. Revert if it spans zero.

- [ ] **Step 5: Commit**

```bash
git add search.py tests/test_search.py
git commit -m "feat(search): add one-ply and two-ply continuation history"
```

---

### Task 16: Time management validation and tuning pass

**Files:**
- Modify: `search.py`
- Create: `tests/test_time_management.py`

**Interfaces:**
- Consumes: Task 15.
- Produces: no new interfaces. Tunes constants and proves the clock is safe.

**Notes:** a flag is the most common self-inflicted loss and it costs a whole game. This task exists because every earlier task tested the search and none of them tested the clock under adversarial conditions.

- [ ] **Step 1: Write the failing test**

Create `tests/test_time_management.py`:

```python
"""The clock. A flag loses the game outright, so these are hard requirements."""

import time

import chess
import pytest

import search


@pytest.mark.parametrize("time_left_ms", [400, 1_000, 5_000, 30_000, 120_000])
def test_never_exceeds_the_clock(time_left_ms: int) -> None:
    search.think(chess.Board(), time_left_ms=1_000)  # warm any lazy path first
    started = time.perf_counter()
    uci = search.think(chess.Board(), time_left_ms=time_left_ms, increment_ms=500)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < time_left_ms, f"used {elapsed_ms:.0f}ms of {time_left_ms}ms"
    assert chess.Move.from_uci(uci) in chess.Board().legal_moves


@pytest.mark.parametrize("time_left_ms", [0, 1, 5, 50])
def test_returns_a_legal_move_with_essentially_no_clock(time_left_ms: int) -> None:
    """Below the 300ms reserve the budget floors out. We cannot avoid flagging here,
    but we must still return a legal move rather than crash or return nothing."""
    uci = search.think(chess.Board(), time_left_ms=time_left_ms, increment_ms=500)
    assert chess.Move.from_uci(uci) in chess.Board().legal_moves


def test_budget_is_floored_when_the_clock_is_below_the_reserve() -> None:
    soft, hard = search.budget_ms(100, 500)
    assert soft > 0 and hard > 0 and soft <= hard


def test_budget_grows_with_the_clock() -> None:
    small, _ = search.budget_ms(10_000, 500)
    large, _ = search.budget_ms(100_000, 500)
    assert large > small


def test_uses_a_meaningful_share_of_a_healthy_clock() -> None:
    started = time.perf_counter()
    search.think(chess.Board(), time_left_ms=60_000, increment_ms=500)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert 1_000 < elapsed_ms < 6_000, f"used {elapsed_ms:.0f}ms, which is wasteful or reckless"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_time_management.py -v`
Expected: the very short clocks fail, because a first call still pays for any un-warmed compilation and the floor is untested.

- [ ] **Step 3: Write the implementation**

Fix whatever the tests expose. Confirm every jitted function is warmed at import so no compilation ever lands on the clock. Add the instability extension: when the best move changed at this depth or the score fell by more than 30 centipawns, allow the soft limit to stretch to 1.5 times, never past the hard limit.

- [ ] **Step 4: Run the full verification**

Run: `uv run pytest -q`
Expected: all pass.

Run: `make gate`
Expected: clean.

Run: `uv run python -m harness.arena --opponent baselines/greedy --games 20 --base-ms 120000 --increment-ms 500`
Expected: no `flag`, `crash`, `illegal` or `init` terminations.

Run: `make ab OPPONENT=snapshots/baseline GAMES=400`
Expected: a large positive Elo interval against the Task 11 baseline snapshot.

Run: `make zip`
Expected: `submission.zip` containing `agent.py` at the root alongside the engine modules, comfortably under 50 MB.

- [ ] **Step 5: Commit**

```bash
git add search.py tests/test_time_management.py
git commit -m "test(search): validate time management under adversarial clocks"
```

---

## Post-plan notes

**What is deliberately not here.** Opening books and endgame tablebases, both allowed by the rules and both judged not worth their weight in the spec. Evaluation improvements, which are the user's work. A piece-square table would make every A/B measurement in Tasks 12 to 15 more trustworthy, because a material-only evaluation cannot distinguish many of the positions the added depth reaches, but it belongs to the user's file and their judgement.

**Order matters.** Tasks 2 to 9 build a foundation whose correctness is checked against `python-chess` at every step. Do not start Task 10 with a failing perft. A move generation bug found after the search exists is far harder to isolate, and an illegal move loses a game outright.

**Test file naming differs from the spec, deliberately.** The spec listed a separate
`fuzz_movegen.py`. In this plan that coverage lives inside `tests/test_movegen.py`, which
compares the generated move set against `python-chess` across 200 varied positions, and
inside `tests/test_perft.py`, which is an exhaustive check over six positions to depth 5.
Together those are strictly stronger than a separate fuzz file would have been, so there is
no gap, only a different filename.

**A snapshot must be taken before the change it will be compared against.** Tasks 12
through 15 each snapshot in their Step 2 and measure in their Step 4. Snapshotting after
the change compares a version against itself and reports zero Elo with a confident-looking
interval.

**The oracle is the point.** `test_oracle_still_holds` reappears in Tasks 12, 13, 14 and 15 on purpose. Every technique added in those tasks reshapes the search tree while being required not to change the value of a full-width search. Without re-running that check after each one, a broken heuristic is indistinguishable from a weak one.

---

## Results (recorded 2026-09-06)

What was built, and what it measured. Every Elo figure is fixed-node self-play against a
snapshot of the version immediately before the change, at 30k nodes per move, with a 95%
interval.

| Task | Outcome |
| --- | --- |
| 1-9 foundation | perft correct on all six standard positions; `make`, Zobrist and SEE verified against python-chess |
| 6 incremental Zobrist | perft(6) 7.83 to 13.35 Mnps |
| 10 search core | 3.3 Mnps in search; first playing version |
| 11 A/B harness | scores exactly +5 =10 -5 against a copy of itself, so it is unbiased |
| 12 whole-node pruning | **+108 Elo [+61, +158]**, +19 =40 -1 |
| 13 move-loop pruning and LMR | **+191 Elo [+130, +265]**, +32 =26 -2 |
| 14 singular extensions | **-41 Elo [-94, +9]**, +12 =43 -21. **Reverted.** |
| 15 continuation history | **+4 Elo [-42, +51]**, +15 =51 -14. Kept but unproven. |
| 16 time management | found and fixed a real bug: the soft limit was computed but never enforced, so every move ran to the hard limit |

Final state: 316 tests passing, `make gate` clean, 20/20 by checkmate against `greedy` with
no flags or crashes, and a win at the real 120s + 0.5s control.

### Three things worth carrying forward

**Two of the plan's design assumptions were wrong and were corrected during the build.**
numba exposes module globals to jitted code as *readonly* arrays, so the "mutate globals"
design could not work and every mutable structure now travels in one `Work` namedtuple.
And `mypy --strict` does not accept numba's `int64`/`uint64` at call boundaries once real
bitboards flow through them, so jitted functions annotate as `Any` on the Python side and
let the numba decorator signature carry the contract.

**The two features that did not pay are the two that depend most on evaluation quality.**
Singular extensions ask whether the TT move is uniquely good, and continuation history
learns which quiet replies work. With a material-only evaluation most quiet moves score
identically, so both are largely measuring noise. Both are worth re-measuring once the
evaluation can tell positions apart; singular extensions are in the history at the parent
of commit `86066cb`.

**Import cost has drifted from 10.9s to 23.5s** as the search grew, against a 30s guard in
`tests/bench.py` and a 90s platform budget. Not a problem yet, and the guard is doing its
job by being close, but the next large addition to the jitted code will trip it.
