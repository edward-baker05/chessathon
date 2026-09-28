import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

from harness.rules import READ_CHUNK, WATCHDOG_GRACE_MS

RUNNER = Path(__file__).resolve().parent / "runner.py"
DRAIN_GRACE_S = 0.2


class AgentFailure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def local(directory: Path) -> "Agent":
    """Run an agent as a process on this machine, through harness/runner.py."""
    return Agent([sys.executable, str(RUNNER), str(directory.resolve())])


class Agent:
    """One agent process, spoken to over a line-based JSON protocol."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.stderr_tail = ""
        self._process: subprocess.Popen[bytes] | None = None
        self._chunks: queue.Queue[tuple[str, bytes]] = queue.Queue()
        self._readers: list[threading.Thread] = []
        self._buffer = b""
        self._tail = b""

    def start(self, init_budget_s: float) -> None:
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._process = process
        self._readers = [
            self._reader(_pipe(process.stdout), "stdout"),
            self._reader(_pipe(process.stderr), "stderr"),
        ]
        ready = self._await_line(time.monotonic() + init_budget_s)
        if ready is None:
            raise AgentFailure("init" if process.poll() is None else "crash")
        if not _is_ready(ready):
            raise AgentFailure("init")

    def move(self, fen: str, time_left_ms: int) -> str:
        if self._process is None:
            raise RuntimeError("agent moved before start")
        request = json.dumps({"fen": fen, "time_left_ms": time_left_ms}).encode()
        try:
            _pipe(self._process.stdin).write(request + b"\n")
        except BrokenPipeError:
            raise AgentFailure("crash") from None
        line = self._await_line(time.monotonic() + (time_left_ms + WATCHDOG_GRACE_MS) / 1000.0)
        if line is None:
            raise AgentFailure("flag")
        return _parse_move(line)

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.kill()
        for reader in self._readers:
            reader.join(DRAIN_GRACE_S)
        self._drain()
        self.stderr_tail = self._tail.decode("utf-8", "replace")
        _pipe(self._process.stdin).close()
        self._process.wait()
        self._process = None
        self._readers = []

    def _reader(self, stream: IO[bytes], name: str) -> threading.Thread:
        reader = threading.Thread(target=self._forward, args=(stream, name), daemon=True)
        reader.start()
        return reader

    # the reader owns its pipe, so a thread outliving stop() never reads a closed stream
    def _forward(self, stream: IO[bytes], name: str) -> None:
        with stream:
            while True:
                chunk = stream.read(READ_CHUNK)
                self._chunks.put((name, chunk))
                if not chunk:
                    return

    def _await_line(self, deadline: float) -> bytes | None:
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                name, chunk = self._chunks.get(timeout=remaining)
            except queue.Empty:
                return None
            if name == "stderr":
                self._tail += chunk
            elif not chunk:
                raise AgentFailure("crash")
            else:
                self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line

    def _drain(self) -> None:
        while True:
            try:
                name, chunk = self._chunks.get_nowait()
            except queue.Empty:
                return
            if name == "stderr":
                self._tail += chunk


def _pipe(stream: IO[bytes] | None) -> IO[bytes]:
    if stream is None:
        raise RuntimeError("the agent process exposed no pipe")
    return stream


def _is_ready(line: bytes) -> bool:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("ready") is True


def _parse_move(line: bytes) -> str:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AgentFailure("illegal") from None
    move = payload.get("move") if isinstance(payload, dict) else None
    if not isinstance(move, str):
        raise AgentFailure("illegal")
    return move
