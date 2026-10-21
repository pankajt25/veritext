"""Tests for text-spinner and tortured-phrase detection (Cabanac et al. 2021)."""

import json
from pathlib import Path

from veritext.checks.spinner import check_spinner, load_tortured_phrases
from veritext.ingest.models import Document
from veritext.similarity.models import MatchSpan, SimilarityResult


def test_tortured_phrases_config_loaded():
    phrases = load_tortured_phrases()
    assert "counterfeit consciousness" in phrases
    assert phrases["counterfeit consciousness"] == "artificial intelligence"
    assert "profound learning" in phrases
    assert phrases["profound learning"] == "deep learning"
    assert "enormous information" in phrases
    assert phrases["enormous information"] == "big data"


def test_detect_known_tortured_phrases():
    tortured_text = (
        "Modern research explores counterfeit consciousness and profound learning architectures "
        "to synthesize enormous information efficiently. The working framework ensures low error."
    )
    doc = Document(
        raw_text=tortured_text,
        clean_text=tortured_text,
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )

    res = check_spinner(doc)
    assert not res.passed
    assert len(res.findings) >= 3

    for f in res.findings:
        assert f.check == "spinner"
        assert f.severity == "warning"
        assert f.message == "possible automated paraphrasing; review manually"

    found_phrases = [f.details.get("found_phrase", "").lower() for f in res.findings]
    assert "counterfeit consciousness" in found_phrases
    assert "profound learning" in found_phrases
    assert "enormous information" in found_phrases


def test_no_false_positive_on_normal_academic_text():
    normal_text = (
        "Modern research explores artificial intelligence and deep learning architectures "
        "to synthesize big data efficiently. The operating system ensures low error."
    )
    doc = Document(
        raw_text=normal_text,
        clean_text=normal_text,
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )

    res = check_spinner(doc)
    assert res.passed
    assert len(res.findings) == 0


def test_custom_tortured_phrases_file(tmp_path):
    custom_cfg = tmp_path / "custom_phrases.json"
    custom_cfg.write_text(
        json.dumps({
            "phrases": {
                "synthetic acumen": "artificial intelligence",
                "colossal telemetry": "big data"
            }
        }),
        encoding="utf-8",
    )

    text = "The approach relies on synthetic acumen and colossal telemetry."
    doc = Document(
        raw_text=text,
        clean_text=text,
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )

    res = check_spinner(doc, custom_file=custom_cfg)
    assert not res.passed
    found = [f.details.get("found_phrase", "").lower() for f in res.findings]
    assert "synthetic acumen" in found
    assert "colossal telemetry" in found


def test_semantic_synonym_substitution_flagged():
    doc_text = "The neural network synthesizes intricate representations from input vectors."
    src_text = "The neural network computes complex representations from input vectors."

    doc = Document(
        raw_text=doc_text,
        clean_text=doc_text,
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )

    # Simulated semantic match span with high similarity but low exact word overlap
    span = MatchSpan(
        type="paraphrase",
        score=0.82,
        source_id="src_1",
        source_title="Deep Learning Foundations",
        source_url="https://example.com/dl",
        doc_start=0,
        doc_end=len(doc_text),
        matched_text=doc_text,
        source_start=0,
        source_end=len(src_text),
        source_text=src_text,
    )

    sim_res = SimilarityResult(
        total_clean_chars=len(doc_text),
        matched_chars=len(doc_text),
        similarity_percentage=75.0,
        spans=[span],
        sources=[],
    )

    res = check_spinner(doc, similarity_result=sim_res)
    assert any(f.message == "possible automated paraphrasing; review manually" for f in res.findings)
