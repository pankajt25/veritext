"""Unit tests for the sources CLI subcommands (add, search, stats)."""

from pathlib import Path
import subprocess
import sys
import pytest


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_cli_sources_stats(base_dir):
    res = subprocess.run(
        [sys.executable, "-m", "veritext.sources", "stats"],
        cwd=base_dir,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Indexed Sources:" in res.stdout
    assert "Indexed Passages:" in res.stdout
    assert "HTTP Cache Entries:" in res.stdout


def test_cli_sources_add_directory(base_dir):
    ref_dir = base_dir / "data" / "references"
    res = subprocess.run(
        [sys.executable, "-m", "veritext.sources", "add", str(ref_dir)],
        cwd=base_dir,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Successfully indexed" in res.stdout


def test_cli_sources_search_offline(base_dir):
    sample = base_dir / "data" / "samples" / "sample_academic.txt"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "veritext.sources",
            "search",
            str(sample),
            "--offline",
            "--max",
            "5",
        ],
        cwd=base_dir,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Searching candidate sources for: sample_academic.txt" in res.stdout
    assert "Mode: Offline (Local Library Only)" in res.stdout


def test_cli_sources_invalid_command(base_dir):
    res = subprocess.run(
        [sys.executable, "-m", "veritext.sources", "invalid_subcommand"],
        cwd=base_dir,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
