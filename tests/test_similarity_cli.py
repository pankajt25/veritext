"""Tests for the similarity detection CLI interface: python -m veritext.similarity."""

import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
from veritext.similarity.__main__ import main


def test_cli_offline_and_json_output(tmp_path, monkeypatch, capsys):
    doc_path = tmp_path / "test_doc.txt"
    doc_path.write_text(
        "Modern deep learning algorithms utilize attention mechanisms for sequence processing. "
        "These computational layers replace recurrence entirely across NLP applications.",
        encoding="utf-8",
    )
    json_out = tmp_path / "result.json"

    # Test CLI invocation: python -m veritext.similarity <file> --offline --json <out.json>
    test_args = [
        "veritext.similarity",
        str(doc_path),
        "--offline",
        "--json",
        str(json_out),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("veritext.similarity.engine.find_candidates", return_value=[]):
        main()

    captured = capsys.readouterr()
    assert "OVERALL SIMILARITY SCORE" in captured.out
    assert "Characters Matched" in captured.out

    # Verify JSON output file was generated and is valid
    assert json_out.exists()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert "similarity_percentage" in data
    assert "matched_chars" in data
    assert "sources" in data
    assert "spans" in data


def test_cli_nonexistent_file(monkeypatch, capsys):
    test_args = ["veritext.similarity", "/nonexistent/path/doc.txt"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error: File not found" in captured.err
