"""Integration tests for the ingestion CLI (python -m veritext.ingest)."""

from pathlib import Path
import subprocess
import sys
import pytest


@pytest.fixture
def samples_dir():
    return Path(__file__).resolve().parent.parent / "data" / "samples"


def test_cli_help():
    res = subprocess.run(
        [sys.executable, "-m", "veritext.ingest", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "usage:" in res.stdout.lower()
    assert "veritext.ingest" in res.stdout.lower()


def test_cli_txt_sample(samples_dir):
    sample = samples_dir / "sample_academic.txt"
    res = subprocess.run(
        [sys.executable, "-m", "veritext.ingest", str(sample)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Paragraph Count:" in res.stdout
    assert "Sentence Count:" in res.stdout
    assert "Removed Sections" in res.stdout


def test_cli_missing_file():
    res = subprocess.run(
        [sys.executable, "-m", "veritext.ingest", "non_existent_file_xyz.txt"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "Error: File not found" in res.stderr
