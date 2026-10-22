"""Tests for authorship metadata timeline check and tone compliance."""

from pathlib import Path
import pytest

from veritext.checks.authorship import (
    AUTHORSHIP_DISCLAIMER,
    FORBIDDEN_ACCUSATORY_WORDS,
    check_authorship_timeline,
    extract_metadata_timeline,
)
from veritext.ingest.models import Document


def test_neutral_wording_policy(tmp_path):
    """Enforces that NO accusatory or punitive words ever appear in authorship outputs."""
    sample_txt = tmp_path / "test_paper.txt"
    sample_txt.write_text("An academic paper on computational linguistics.", encoding="utf-8")

    result = check_authorship_timeline(sample_txt)

    # Check overall summary
    summary_lower = result.summary.lower()
    for word in FORBIDDEN_ACCUSATORY_WORDS:
        assert word not in summary_lower, f"Found forbidden word '{word}' in result summary"

    # Check every finding message
    for f in result.findings:
        msg_lower = f.message.lower()
        for word in FORBIDDEN_ACCUSATORY_WORDS:
            assert word not in msg_lower, f"Found forbidden word '{word}' in finding message"
        # Must be info severity
        assert f.severity == "info"

    # Check disclaimer
    disc_lower = AUTHORSHIP_DISCLAIMER.lower()
    for word in FORBIDDEN_ACCUSATORY_WORDS:
        assert word not in disc_lower


def test_authorship_disclaimer_content():
    """Validates that disclaimer explicitly states not keystroke tracking and not proof."""
    disc = AUTHORSHIP_DISCLAIMER.lower()
    assert "not perform keystroke tracking" in disc or "not keystroke tracking" in disc
    assert "not constitute proof" in disc or "not proof" in disc


def test_metadata_extraction_on_samples():
    """Tests timeline extraction on sample academic files."""
    docx_sample = Path("data/samples/sample_academic.docx")
    if docx_sample.is_file():
        meta = extract_metadata_timeline(docx_sample)
        assert "properties" in meta
        assert "events" in meta
        assert meta["disclaimer"] == AUTHORSHIP_DISCLAIMER

    pdf_sample = Path("data/samples/sample_academic.pdf")
    if pdf_sample.is_file():
        meta = extract_metadata_timeline(pdf_sample)
        assert "properties" in meta
        assert "events" in meta
        assert meta["disclaimer"] == AUTHORSHIP_DISCLAIMER


def test_authorship_with_raw_document():
    doc = Document(
        raw_text="Sample text without a filesystem path.",
        clean_text="Sample text without a filesystem path.",
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
        path=None,
    )
    res = check_authorship_timeline(doc)
    assert res.passed is True
    assert res.findings == []
