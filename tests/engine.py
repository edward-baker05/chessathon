"""A driver for the engine, shared by the tests that speak UCI to it.

Skipped until the engine is built with `make`.
"""

import subprocess
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parent.parent / "build" / "engine"

pytestmark = pytest.mark.skipif(
    not ENGINE.exists(), reason="the engine is not built; run `make`"
)


class Engine:
    """One engine process, spoken to over UCI a command at a time."""

    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [str(ENGINE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def send(self, line: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def until(self, prefix: str) -> list[str]:
        """Every line up to and including the first that starts with `prefix`."""
        assert self.process.stdout is not None
        lines = []
        for line in self.process.stdout:
            lines.append(line.strip())
            if line.startswith(prefix):
                return lines
        raise AssertionError(f"the engine exited before printing {prefix!r}: {lines}")

    def close(self) -> None:
        self.send("quit")
        self.process.wait(timeout=10)


def info_field(line: str, name: str) -> str:
    tokens = line.split()
    return tokens[tokens.index(name) + 1]
