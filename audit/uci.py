"""Synchronous offline UCI client, avoiding sandbox-blocked asyncio wakeup sockets."""

import os
import select
import subprocess
import time
from typing import Any

import chess
import chess.engine


class Engine:
    def __init__(self, path: str) -> None:
        self.process = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.buffer = b''
        self.id: dict[str, str] = {}
        self.send('uci')
        while (line := self.read()) != 'uciok':
            if line.startswith('id '):
                _, key, value = line.split(' ', 2)
                self.id[key] = value

    def send(self, line: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write((line + '\n').encode())
        self.process.stdin.flush()

    def read(self) -> str:
        assert self.process.stdout is not None
        deadline = time.monotonic() + 60
        while b'\n' not in self.buffer:
            timeout = deadline - time.monotonic()
            if timeout <= 0 or not select.select([self.process.stdout], [], [], timeout)[0]:
                raise TimeoutError('UCI did not respond in 60 seconds')
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError('UCI process exited')
            self.buffer += data
        line, self.buffer = self.buffer.split(b'\n', 1)
        return line.decode().strip()

    def configure(self, options: dict[str, Any]) -> None:
        for key, value in options.items():
            self.send(f'setoption name {key}' + ('' if value is None else f' value {value}'))
        self.send('isready')
        while self.read() != 'readyok':
            pass

    def analyse(self, board: chess.Board, limit: chess.engine.Limit,
                multipv: int | None = None,
                root_moves: list[chess.Move] | None = None) -> Any:
        self.configure({'MultiPV': multipv or 1})
        self.send(f'position fen {board.fen()}')
        command = f'go nodes {limit.nodes}'
        if root_moves:
            command += ' searchmoves ' + ' '.join(move.uci() for move in root_moves)
        self.send(command)
        lines: dict[int, Any] = {}
        while not (line := self.read()).startswith('bestmove'):
            if line.startswith('info ') and ' pv ' in line:
                info = chess.engine._parse_uci_info(line[5:], board, chess.engine.INFO_ALL)
                lines[info.get('multipv', 1)] = info
        result = [lines[key] for key in sorted(lines)]
        return result if multipv is not None else result[0]

    def quit(self) -> None:
        self.send('quit')
        self.process.wait(timeout=5)
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.close()
        self.process.stdout.close()
