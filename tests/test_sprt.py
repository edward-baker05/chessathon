import os
import stat
from pathlib import Path

import pytest

from tools.sprt import copy_worktree, freeze, tally, uci_options


def test_the_working_tree_is_copied_with_what_the_build_reads(tmp_path: Path) -> None:
    copy_worktree(tmp_path)
    root = Path(__file__).resolve().parent.parent
    assert (tmp_path / "Makefile").is_file()
    assert (tmp_path / "src" / "search.cpp").read_bytes() == (
        root / "src" / "search.cpp"
    ).read_bytes()
    assert (tmp_path / "weights" / "net.npz").is_file()
    assert (tmp_path / "tools" / "export_net.py").is_file()
    # Only what the build reads: no tests, and no build directory carried across.
    assert not (tmp_path / "tests").exists()
    assert not (tmp_path / "build").exists()


def test_an_executable_is_played_directly(tmp_path: Path) -> None:
    engine = tmp_path / "engine"
    engine.write_text("#!/bin/sh\n")
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    build = freeze("dev", str(engine), run_dir)
    assert build.executable == run_dir / "dev" / "engine"
    assert os.access(build.executable, os.X_OK)


def test_an_unknown_build_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        freeze("dev", "no-such-ref-or-path", tmp_path)



def test_tally_counts_abnormal_endings(tmp_path: Path) -> None:
    pgn = tmp_path / "games.pgn"
    pgn.write_text(
        '[Event "a"]\n[Termination "normal"]\n\n1. e4 1-0\n\n'
        '[Event "b"]\n[Termination "time forfeit"]\n\n1. e4 0-1\n\n'
        '[Event "c"]\n\n1. d4 1/2-1/2\n'
    )
    assert tally(pgn) == {"normal": 2, "time forfeit": 1}


def test_engine_options_become_fastchess_arguments() -> None:
    assert uci_options(["Threads=2", "Hash=64", "Clear Hash="]) == [
        "option.Threads=2",
        "option.Hash=64",
        "option.Clear Hash=",
    ]
    with pytest.raises(SystemExit):
        uci_options(["Threads"])
