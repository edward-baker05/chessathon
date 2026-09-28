"""Build board-only optimization experiments from the frozen current source."""

import shutil
from pathlib import Path

BASE = Path("/tmp/chessathon-representation")


def main() -> None:
    for name in ("contiguous", "bitscan", "combined", "copyhelpers"):
        target = BASE / name
        shutil.copytree(BASE / "current", target, dirs_exist_ok=True)
        if name in ("contiguous", "combined", "copyhelpers"):
            for filename in ("position.py", "movegen.py"):
                path = target / filename
                source = path.read_text()
                for kind in ("uint64", "int8", "int32"):
                    source = source.replace(f"{kind}[:]", f"{kind}[::1]")
                    source = source.replace(f"{kind}[:, :]", f"{kind}[:, ::1]")
                path.write_text(source)
        if name in ("bitscan", "combined"):
            path = target / "bitboard.py"
            source = path.read_text().replace(
                "from numba import boolean, int64, njit, uint64",
                "from numba import boolean, int64, njit, uint64\n"
                "from numba.cpython.unsafe.numbers import trailing_zeros",
            )
            before = "return DEBRUIJN_INDEX[int(((b & (~b + ONE)) * DEBRUIJN) >> U(58))]"
            assert source.count(before) == 1
            source = source.replace(before, "return int64(trailing_zeros(b))")
            path.write_text(source)
        if name == "copyhelpers":
            path = target / "position.py"
            source = path.read_text()
            helpers = """@njit(void(uint64[::1], uint64[::1]), cache=False)
def _copy_state(src: Bits, dst: Bits) -> None:
    for i in range(NFIELDS):
        dst[i] = src[i]


@njit(void(int8[::1], int8[::1]), cache=False)
def _copy_mail(src: Bits, dst: Bits) -> None:
    for i in range(64):
        dst[i] = src[i]


"""
            marker = "@njit(void(uint64[::1], int8[::1], uint64[::1], int8[::1], int32)"
            source = source.replace(marker, helpers + marker)
            before = (
                "    for i in range(NFIELDS):\n        dst_state[i] = src_state[i]\n"
                "    for i in range(64):\n        dst_mail[i] = src_mail[i]\n"
            )
            assert source.count(before) == 2
            source = source.replace(
                before,
                "    _copy_state(src_state, dst_state)\n    _copy_mail(src_mail, dst_mail)\n",
            )
            path.write_text(source)


if __name__ == "__main__":
    main()
