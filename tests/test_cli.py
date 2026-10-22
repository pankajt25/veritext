"""Integration tests for the ingestion CLI (python -m veritext.ingest)."""

from pathlib import Path
import pytest
from veritext.ingest.__main__ import main


@pytest.fixture
def samples_dir():
    return Path(__file__).resolve().parent.parent / "data" / "samples"


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()
    assert "veritext.ingest" in captured.out.lower()


def test_cli_txt_sample(samples_dir, capsys):
    sample = samples_dir / "sample_academic.txt"
    main([str(sample)])
    captured = capsys.readouterr()
    assert "Paragraph Count:" in captured.out
    assert "Sentence Count:" in captured.out
    assert "Removed Sections" in captured.out


def test_cli_missing_file(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["non_existent_file_xyz.txt"])
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "Error: File not found" in captured.err
