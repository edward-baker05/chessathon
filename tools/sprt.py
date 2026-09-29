"""Play one build of the engine against another under fastchess, until an SPRT decides.

A match of a few hundred games has a 95% interval of about +-30 Elo, which cannot see the 5
to 15 Elo most real changes are worth. This lets fastchess stop the match as soon as the
sequential test is decided either way.

A build is one of:

  worktree        the source tree as it stands, uncommitted changes included (the default for --dev)
  a git ref       HEAD, main, a sha: the sources and the network at that commit
  an executable   a UCI engine, played as it is. It is copied alone, so it has to carry its
                  network inside it, as build/engine does.

A source build is compiled into the run directory, so `make` need not have been run
and its result is never what is measured. Every build is frozen there before the first game,
so editing the tree during a match cannot change who is playing.

    make sprt                                  # the working tree against HEAD
    make sprt BASE=main~3                      # against an older commit
    make sprt DEV=../other-engine/build/engine # a UCI binary against HEAD
    make sprt DEV=build/engine BASE=build/engine ARGS="--dev-option Threads=2"
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
import datetime
import io
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FASTCHESS = ROOT / "third_party" / "fastchess" / "fastchess"
BOOK = ROOT / "third_party" / "books" / "UHO_Lichess_4852_v1.epd"
RUNS = ROOT / "sprt"

WORKTREE = "worktree"
# What building the engine reads: the sources, the network and the tool that embeds it.
SOURCES = ["Makefile", "src", "weights/net.npz", "tools/export_net.py"]
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
    executable: Path


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def copy_worktree(destination: Path) -> None:
    for source in SOURCES:
        target = destination / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / source).is_dir():
            shutil.copytree(ROOT / source, target)
        else:
            shutil.copy2(ROOT / source, target)


def extract_commit(sha: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", sha, "--", *SOURCES],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")


def compile_engine(root: Path, name: str, run_dir: Path) -> Path:
    print(f"building {name}", flush=True)
    with open(run_dir / f"{name}-build.log", "wb") as log:
        result = subprocess.run(
            [
                "make",
                "-C",
                str(root),
                f"-j{os.cpu_count() or 1}",
                f"PYTHON={sys.executable}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        log_text = (run_dir / f"{name}-build.log").read_text(errors="replace")
        sys.exit(f"the {name} build failed:\n{log_text[-2000:]}")
    return root / "build" / "engine"


def freeze(name: str, spec: str, run_dir: Path) -> Build:
    """Build or copy what `spec` names into the run directory, where nothing else will touch it."""
    root = run_dir / name
    root.mkdir()
    path = Path(spec)
    if spec == WORKTREE:
        copy_worktree(root)
        dirty = git("status", "--porcelain", "--", *SOURCES)
        origin = f"working tree at {git('rev-parse', '--short=12', 'HEAD')}"
        origin += " with changes" if dirty else ""
        return Build(name, spec, origin, root, compile_engine(root, name, run_dir))
    if path.is_file() and os.access(path, os.X_OK):
        executable = root / path.name
        shutil.copy2(path, executable)
        return Build(name, spec, str(path.resolve()), root, executable)
    try:
        sha = git("rev-parse", "--verify", "--quiet", f"{spec}^{{commit}}")
    except subprocess.CalledProcessError:
        sys.exit(f"{spec!r} is not a git ref or an executable")
    extract_commit(sha, root)
    return Build(name, spec, f"{spec} at {sha[:12]}", root, compile_engine(root, name, run_dir))


def uci_options(options: list[str]) -> list[str]:
    """`--dev-option Threads=2` and the like, as fastchess engine arguments."""
    arguments = []
    for option in options:
        name, equals, value = option.partition("=")
        if not equals or not name.strip():
            sys.exit(f"an engine option is NAME=VALUE, not {option!r}")
        arguments.append(f"option.{name.strip()}={value}")
    return arguments


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

    pv_warnings = 0
    try:
        command = [str(FASTCHESS)]
        for build in builds:
            command += [
                "-engine",
                f"cmd={build.executable}",
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
