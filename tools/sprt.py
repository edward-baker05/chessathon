"""Play one build against another under fastchess, until an SPRT decides.

`tests/match.py` imports the agent afresh for every game, which costs about a minute of
numba compilation per side per game, and plays a length fixed in advance. That capped a
match at a few hundred games, whose 95% interval of about +-30 Elo cannot see the 5 to 15
Elo most real changes are worth. This keeps each build warm in a zygote (lichess/zygote.py)
that forks a fresh process per game, so a game costs its own clock and nothing else, and
lets fastchess stop the match as soon as the sequential test is decided either way.

A build is one of:

  worktree        the working tree, uncommitted changes included (the default for --dev)
  a git ref       HEAD, main, a sha: the top-level .py files and weights at that commit
  a directory     a snapshot holding agent.py, e.g. snapshots/<tag>
  an executable   a native UCI engine, played directly. It is copied alone, so it has to
                  carry its network inside it.

Every build is frozen into the run directory before the first game, so editing the tree
during a match cannot change who is playing.

    make sprt                                  # the working tree against HEAD
    make sprt BASE=main~3                      # against an older commit
    make sprt DEV=../engine-cpp/build/engine   # a UCI binary against HEAD
    make sprt DEV=cpp/build/engine BASE=cpp/build/engine ARGS="--dev-option Threads=2"
                                               # the same binary, two threads against one
    uv run python tools/sprt.py --games 200    # a fixed-length run with no SPRT

The default bounds are [0, 10] in normalized Elo: H1 accepted means the change is better, H0
accepted means it is not worth 10 nElo. At this engine's draw rate 10 nElo is about 6 Elo.
fishtest's own [0, 2] is sized for a cluster: on one six-core machine, at about 600 games an
hour, [0, 10] takes up to half a day for a change near either bound and far less for a clear
winner or loser, while [0, 5] takes up to two days. A non-regression test for a
simplification is `--elo0 -10 --elo1 0`.
"""

import argparse
import collections
import contextlib
import datetime
import io
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FASTCHESS = ROOT / "third_party" / "fastchess" / "fastchess"
BOOK = ROOT / "third_party" / "books" / "UHO_Lichess_4852_v1.epd"
ZYGOTE = ROOT / "lichess" / "zygote.py"
UCI_CLIENT = ROOT / "lichess" / "uci_client.py"
RUNS = ROOT / "sprt"

WORKTREE = "worktree"
# The import compiles every jitted function. It takes about a minute alone; allow for a
# loaded machine before calling a zygote stuck.
IMPORT_TIMEOUT_S = 600.0
# An SPRT stops itself. This only bounds a test whose true value sits between the bounds.
MAX_GAMES = 40_000
# Endings that are chess. Anything else is an engine fault: a flag, a crash, an illegal move.
CLEAN_ENDINGS = {"normal", "adjudication"}
# fastchess checks every reported PV and prints the whole game when one plays on past a
# draw. That is a reporting fault, not a chess one, so it is kept in the log and counted.
PV_WARNING = "Warning; PV continues after"
QUIET = ("Started game", "Info; ", "Position; ", "Moves; ", PV_WARNING)


@dataclass
class Build:
    name: str
    spec: str
    # What the spec resolved to, for the record: a commit, a path.
    origin: str
    root: Path
    # A native UCI engine, played directly. None for a Python build served by a zygote.
    executable: Path | None = None


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def copy_python_build(source: Path, destination: Path) -> None:
    """The files the engine runs from: the top-level modules and the network."""
    for module in sorted(source.glob("*.py")):
        shutil.copy2(module, destination / module.name)
    if (source / "weights").is_dir():
        shutil.copytree(source / "weights", destination / "weights")


def extract_commit(sha: str, destination: Path) -> None:
    top_level = git("ls-tree", "--name-only", sha).splitlines()
    wanted = [name for name in top_level if name.endswith(".py") or name == "weights"]
    archive = subprocess.run(
        ["git", "archive", "--format=tar", sha, "--", *wanted],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")


def freeze(name: str, spec: str, run_dir: Path) -> Build:
    """Copy the build `spec` names into the run directory, where nothing else will touch it."""
    root = run_dir / name
    root.mkdir()
    path = Path(spec)
    if spec == WORKTREE:
        copy_python_build(ROOT, root)
        dirty = git("status", "--porcelain", "--", "*.py", "weights")
        origin = f"working tree at {git('rev-parse', '--short=12', 'HEAD')}"
        return Build(name, spec, origin + (" with changes" if dirty else ""), root)
    if path.is_file() and os.access(path, os.X_OK):
        executable = root / path.name
        shutil.copy2(path, executable)
        return Build(name, spec, str(path.resolve()), root, executable)
    if path.is_dir() and (path / "agent.py").is_file():
        copy_python_build(path, root)
        return Build(name, spec, str(path.resolve()), root)
    try:
        sha = git("rev-parse", "--verify", "--quiet", f"{spec}^{{commit}}")
    except subprocess.CalledProcessError:
        sys.exit(f"{spec!r} is not a git ref, a directory holding agent.py or an executable")
    extract_commit(sha, root)
    return Build(name, spec, f"{spec} at {sha[:12]}", root)


Zygote = tuple[subprocess.Popen[bytes], Path]


def start_zygote(build: Build, sockets: Path, run_dir: Path) -> Zygote:
    socket = sockets / f"{build.name}.sock"
    with open(run_dir / f"{build.name}-zygote.log", "wb") as log:
        process = subprocess.Popen(
            [sys.executable, str(ZYGOTE), "--root", str(build.root), "--socket", str(socket)],
            stdout=subprocess.DEVNULL,
            stderr=log,
            # Its own group, so stopping it also stops any game it forked.
            start_new_session=True,
        )
    return process, socket


def wait_for(zygotes: dict[str, Zygote], run_dir: Path) -> None:
    deadline = time.monotonic() + IMPORT_TIMEOUT_S
    waiting = dict(zygotes)
    while waiting:
        for name, (process, _) in list(waiting.items()):
            # Printed after listen(), where the socket file already exists after bind().
            log = (run_dir / f"{name}-zygote.log").read_text(errors="replace")
            if "zygote ready" in log:
                del waiting[name]
            elif process.poll() is not None:
                sys.exit(f"the {name} build failed to import:\n{log}")
        if time.monotonic() > deadline:
            sys.exit(f"still importing after {IMPORT_TIMEOUT_S:.0f} s: {', '.join(waiting)}")
        time.sleep(0.5)


def engine_command(build: Build, socket: Path | None, run_dir: Path) -> Path:
    """What fastchess runs for `build`: the binary itself, or a relay to its zygote."""
    if build.executable is not None:
        return build.executable
    assert socket is not None
    script = run_dir / f"{build.name}.sh"
    relay = [sys.executable, str(UCI_CLIENT), str(socket)]
    script.write_text(f"#!/bin/sh\nexec {shlex.join(relay)}\n")
    script.chmod(0o755)
    return script


def uci_options(options: list[str]) -> list[str]:
    """`--dev-option Threads=2` and the like, as fastchess engine arguments."""
    arguments = []
    for option in options:
        name, equals, value = option.partition("=")
        if not equals or not name.strip():
            sys.exit(f"an engine option is NAME=VALUE, not {option!r}")
        arguments.append(f"option.{name.strip()}={value}")
    return arguments


def stop(zygotes: dict[str, Zygote]) -> None:
    for process, _ in zygotes.values():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        process.wait()


def tally(pgn: Path) -> collections.Counter[str]:
    """Games by how they ended, from fastchess's Termination tags. A game without one is
    counted as normal, which is what fastchess writes for every game the rules ended."""
    counts: collections.Counter[str] = collections.Counter()
    if not pgn.exists():
        return counts
    text = pgn.read_text(errors="replace")
    games = len(re.findall(r'^\[Event "', text, flags=re.MULTILINE))
    tagged = re.findall(r'^\[Termination "([^"]*)"\]', text, flags=re.MULTILINE)
    counts.update(tagged)
    counts["normal"] += games - len(tagged)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dev", default=WORKTREE, help="the build under test")
    parser.add_argument("--base", default="HEAD", help="the build it is measured against")
    parser.add_argument("--tc", default="10+0.1", help="seconds+increment, for both sides")
    parser.add_argument("--base-tc", help="a different time control for the base only")
    parser.add_argument("--nodes", type=int, help="search a fixed number of nodes, no clock")
    for name in ("dev", "base"):
        parser.add_argument(
            f"--{name}-option",
            action="append",
            default=[],
            metavar="NAME=VALUE",
            help=f"a UCI option for the {name} build, e.g. Threads=2; may be repeated",
        )
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--games", type=int, help="play exactly this many, with no SPRT")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2 - 1),
        help="games at once; one per physical core, less one, is the default",
    )
    parser.add_argument("--book", type=Path, default=BOOK)
    parser.add_argument("--seed", type=int, default=secrets.randbelow(2**31))
    arguments = parser.parse_args()

    if not FASTCHESS.is_file() or not arguments.book.is_file():
        sys.exit("fastchess or the opening book is missing; run `make fastchess-setup`")
    if arguments.games is not None and arguments.games % 2:
        parser.error("--games must be even: every opening is played with both colours")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / stamp
    run_dir.mkdir(parents=True)
    builds = [freeze("dev", arguments.dev, run_dir), freeze("base", arguments.base, run_dir)]

    limits = {"dev": [f"tc={arguments.tc}"], "base": [f"tc={arguments.base_tc or arguments.tc}"]}
    if arguments.nodes is not None:
        limits = {name: [f"nodes={arguments.nodes}"] for name in limits}
    options = {
        "dev": uci_options(arguments.dev_option),
        "base": uci_options(arguments.base_option),
    }
    rounds = (arguments.games or MAX_GAMES) // 2
    record = {
        "started": stamp,
        "builds": {b.name: {"spec": b.spec, "origin": b.origin} for b in builds},
        "limits": limits,
        "options": options,
        "sprt": None
        if arguments.games
        else {k: getattr(arguments, k) for k in ("elo0", "elo1", "alpha", "beta")},
        "games": arguments.games,
        "concurrency": arguments.concurrency,
        "book": str(arguments.book),
        "seed": arguments.seed,
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    for build in builds:
        print(f"{build.name}: {build.origin}")
    print(f"results in {run_dir.relative_to(ROOT)}/")

    sockets = Path(tempfile.mkdtemp(prefix="sprt-"))
    zygotes = {b.name: start_zygote(b, sockets, run_dir) for b in builds if b.executable is None}
    pv_warnings = 0
    try:
        if zygotes:
            print("importing the Python builds (about a minute)", flush=True)
            wait_for(zygotes, run_dir)
        command = [str(FASTCHESS)]
        for build in builds:
            socket = zygotes[build.name][1] if build.name in zygotes else None
            executable = engine_command(build, socket, run_dir)
            command += [
                "-engine",
                f"cmd={executable}",
                f"name={build.name}",
                *limits[build.name],
                *options[build.name],
            ]
        command += [
            # A fresh process per game, as on lichess: nothing carries over between games.
            "-each", "proto=uci", "restart=on",
            "-openings", f"file={arguments.book}", "format=epd", "order=random",
            "-srand", str(arguments.seed),
            "-repeat", "-rounds", str(rounds),
            "-concurrency", str(arguments.concurrency),
            "-pgnout", f"file={run_dir / 'games.pgn'}", "timeleft=true", "nodes=true",
            "-config", f"outname={run_dir / 'config.json'}",
            "-ratinginterval", "20",
        ]  # fmt: skip
        if arguments.games is None:
            command += [
                "-sprt",
                f"elo0={arguments.elo0}",
                f"elo1={arguments.elo1}",
                f"alpha={arguments.alpha}",
                f"beta={arguments.beta}",
                "model=normalized",
            ]
        (run_dir / "command.txt").write_text(shlex.join(command) + "\n")
        with open(run_dir / "fastchess.log", "w") as log:
            fastchess = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            assert fastchess.stdout is not None
            # Flushed per line, so a run redirected to a file or tailed can be watched live.
            for line in fastchess.stdout:
                log.write(line)
                log.flush()
                if line.startswith(PV_WARNING):
                    pv_warnings += 1
                if not line.startswith(QUIET):
                    print(line, end="", flush=True)
            status = fastchess.wait()
    except KeyboardInterrupt:
        status = 130
    finally:
        stop(zygotes)
        shutil.rmtree(sockets, ignore_errors=True)

    counts = tally(run_dir / "games.pgn")
    print(f"\n{sum(counts.values())} games; endings: {dict(counts)}")
    trouble = sum(n for ending, n in counts.items() if ending not in CLEAN_ENDINGS)
    if trouble:
        print(f"WARNING: {trouble} games ended in something other than chess; see games.pgn")
    if pv_warnings:
        print(f"{pv_warnings} reported PVs ran past the game's end; see fastchess.log")
    return status


if __name__ == "__main__":
    sys.exit(main())
