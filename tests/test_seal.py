"""Tests for tamper-evident report sealing and verification."""

import copy
import json
import os
from pathlib import Path
import pytest

from veritext.report.__main__ import main as report_main
from veritext.report.seal import (
    compute_canonical_json,
    compute_seal,
    get_or_create_signing_key,
    seal_report_dict,
    verify_file,
    verify_report_seal,
)


def test_key_creation_and_permissions(tmp_path):
    key_path = tmp_path / "test_report_key"
    assert not key_path.exists()

    key1 = get_or_create_signing_key(key_path)
    assert len(key1) == 32
    assert key_path.exists()

    # Permissions on POSIX must be 0600 (owner read/write only)
    if os.name == "posix":
        mode = os.stat(key_path).st_mode
        assert (mode & 0o777) == 0o600

    # Subsequent call re-reads the same key
    key2 = get_or_create_signing_key(key_path)
    assert key1 == key2


def test_seal_valid_and_tampered_single_char(tmp_path):
    key_path = tmp_path / "test_report_key"
    key = get_or_create_signing_key(key_path)

    sample_report = {
        "document_name": "essay_final.docx",
        "summary": "Originality score 98%. No critical integrity issues detected.",
        "settings": {"offline": True, "mode": "auto"},
        "similarity": {"overall_percentage": 2.5, "spans": []},
        "metadata": {"student_id": "12345"},
    }

    # Seal the report
    sealed = seal_report_dict(copy.deepcopy(sample_report), key=key)
    assert "seal" in sealed
    assert sealed["seal"]["algorithm"] == "HMAC-SHA256"

    # Verify untampered report
    is_valid, msg = verify_report_seal(sealed, key=key)
    assert is_valid is True
    assert "VALID" in msg

    # Tamper 1: Modify a single character in the summary
    tampered_1 = copy.deepcopy(sealed)
    tampered_1["summary"] = "Originality score 99%. No critical integrity issues detected."
    is_valid, msg = verify_report_seal(tampered_1, key=key)
    assert is_valid is False
    assert "checksum mismatch" in msg.lower() or "modified" in msg.lower()

    # Tamper 2: Modify a number in nested dict
    tampered_2 = copy.deepcopy(sealed)
    tampered_2["similarity"]["overall_percentage"] = 2.6
    is_valid, msg = verify_report_seal(tampered_2, key=key)
    assert is_valid is False

    # Tamper 3: Modify key in metadata
    tampered_3 = copy.deepcopy(sealed)
    tampered_3["metadata"]["student_id"] = "12346"
    is_valid, msg = verify_report_seal(tampered_3, key=key)
    assert is_valid is False

    # Tamper 4: Modify signature directly
    tampered_4 = copy.deepcopy(sealed)
    tampered_4["seal"]["signature"] = "a" * 64
    is_valid, msg = verify_report_seal(tampered_4, key=key)
    assert is_valid is False
    assert "mismatch" in msg.lower()


def test_verify_cli(tmp_path, capsys):
    key_path = tmp_path / "cli_report_key"
    key = get_or_create_signing_key(key_path)

    report_data = {
        "document_name": "thesis_chapter.pdf",
        "summary": "Verified report",
        "timestamp": "2026-10-22T08:00:00Z",
    }
    sealed = seal_report_dict(copy.deepcopy(report_data), key=key)

    report_file = tmp_path / "valid_report.json"
    report_file.write_text(json.dumps(sealed, indent=2), encoding="utf-8")

    # Run CLI verify on valid report
    ret = report_main(["verify", str(report_file), "--key", str(key_path)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "VALID" in captured.out
    assert "Not formal legal proof" in captured.out

    # Modify report file
    sealed["summary"] = "Tampered summary"
    report_file.write_text(json.dumps(sealed, indent=2), encoding="utf-8")

    # Run CLI verify on tampered report
    ret = report_main(["verify", str(report_file), "--key", str(key_path)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "TAMPERED" in captured.out
