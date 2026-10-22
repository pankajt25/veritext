"""Unit tests for the sources CLI subcommands (add, search, stats)."""

from pathlib import Path
import pytest
from veritext.sources.__main__ import main


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_cli_sources_stats(capsys):
    main(["stats"])
    captured = capsys.readouterr()
    assert "Indexed Sources:" in captured.out
    assert "Indexed Passages:" in captured.out
    assert "HTTP Cache Entries:" in captured.out


def test_cli_sources_add_directory(base_dir, capsys):
    ref_dir = base_dir / "data" / "references"
    main(["add", str(ref_dir)])
    captured = capsys.readouterr()
    assert "Successfully indexed" in captured.out


def test_cli_sources_search_offline(base_dir, capsys):
    sample = base_dir / "data" / "samples" / "sample_academic.txt"
    main(["search", str(sample), "--offline", "--max", "5"])
    captured = capsys.readouterr()
    assert "Searching candidate sources for: sample_academic.txt" in captured.out
    assert "Mode: Offline (Local Library Only)" in captured.out


def test_cli_sources_invalid_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["invalid_subcommand"])
    assert exc.value.code != 0
