#!/usr/bin/env python3
"""The engine lichess-bot launches: relays UCI between its stdin/stdout and the zygote.

Standard library only, so it starts instantly under any Python. The chess happens in a
process the zygote forks for this connection; see zygote.py.
"""

import os
import socket
import sys
import threading
from pathlib import Path

SOCKET = Path(
    os.environ.get("ENGINE_ZYGOTE_SOCKET", Path(__file__).resolve().parent / "zygote.sock")
)


def main() -> None:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(str(SOCKET))
    except OSError as error:
        sys.exit(f"cannot reach the zygote at {SOCKET} ({error}); start it with `make lichess`")

    def forward_stdin() -> None:
        while chunk := os.read(sys.stdin.fileno(), 4096):
            connection.sendall(chunk)
        connection.shutdown(socket.SHUT_WR)

    threading.Thread(target=forward_stdin, daemon=True).start()
    while chunk := connection.recv(4096):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
