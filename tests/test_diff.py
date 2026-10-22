"""Tests for side-by-side source diff with word-level highlighting and safe escaping."""

import html
import pytest

from veritext.report.diff import (
    align_word_excerpts,
    extract_report_diffs,
    generate_source_diff,
    truncate_excerpt,
)


def test_hostile_text_escaping():
    malicious_target = "<script>alert('xss_target')</script> & <b>bold</b>"
    malicious_source = "<img src=x onerror=alert('xss_source')> & 'quotes'"

    t_html, s_html = align_word_excerpts(malicious_target, malicious_source, "exact")

    # Raw script or img tags must NEVER appear unescaped
    assert "<script>" not in t_html
    assert "<img" not in s_html
    assert "&lt;script&gt;" in t_html
    assert "&lt;img" in s_html

    # Check generate_source_diff HTML output
    diff_item = generate_source_diff(
        target_text=malicious_target,
        source_text=malicious_source,
        match_type="exact",
        source_title="<Malicious Source & Title>",
    )
    rendered = diff_item.render_html_details()
    assert "<script>" not in rendered
    assert "<img src=x" not in rendered
    assert "&lt;Malicious Source &amp; Title&gt;" in rendered


def test_fingerprints_only_notice():
    target = "This is a student submission excerpt that matches a classmate's stored fingerprints."
    diff_item = generate_source_diff(
        target_text=target,
        source_text=None,
        match_type="exact",
        source_title="Class Repo: submission_042",
        is_fingerprint_only=True,
    )

    rendered = diff_item.render_html_details()
    assert "source text not stored (fingerprints only)" in rendered
    assert "Class Repo: submission_042" in rendered


def test_bounded_excerpt_truncation():
    long_target = "word " * 500  # 2500 characters
    long_source = "source " * 500

    item = generate_source_diff(
        target_text=long_target,
        source_text=long_source,
        match_type="paraphrase",
        max_chars=1500,
    )

    assert len(item.target_excerpt) <= 1505
    assert len(item.source_excerpt) <= 1505
    assert item.target_truncated is True
    assert item.source_truncated is True

    rendered = item.render_html_details()
    assert "(truncated to 1,500 chars)" in rendered


def test_word_level_highlighting_and_styles():
    target = "The quick brown fox jumps over the lazy dog."
    source = "A fast brown fox jumps across a lazy cat."

    t_html, s_html = align_word_excerpts(target, source, "near-copy")

    # Shared words: brown, fox, jumps, lazy
    assert "brown" in t_html
    assert "jumps" in t_html
    assert 'border-bottom:2px dashed #ea580c' in t_html
    assert 'border-bottom:2px dashed #ea580c' in s_html

    # Diff styling includes accessibility badge and not colour alone
    item = generate_source_diff(target, source, "near-copy")
    rendered = item.render_html_details()
    assert "NEAR-COPY" in rendered
    assert "dashed" in rendered


def test_extract_report_diffs():
    mock_report = {
        "similarity": {
            "spans": [
                {
                    "type": "exact",
                    "score": 1.0,
                    "source_id": "src_1",
                    "source_title": "Source 1",
                    "source_url": "https://example.org/1",
                    "doc_start": 0,
                    "doc_end": 50,
                    "matched_text": "Machine learning enables computers to learn from data patterns.",
                    "source_text": "Machine learning enables computers to learn from data patterns.",
                },
                {
                    "type": "exact",
                    "score": 0.9,
                    "source_id": "class_repo_fp",
                    "source_title": "Classmate Paper",
                    "doc_start": 60,
                    "doc_end": 100,
                    "matched_text": "Fingerprint match without text stored.",
                    "source_text": None,
                    "metadata": {"fingerprints_only": True},
                },
            ]
        }
    }

    diffs = extract_report_diffs(mock_report)
    assert len(diffs) == 2
    assert diffs[0].source_id == "src_1"
    assert diffs[0].is_fingerprint_only is False
    assert diffs[1].source_id == "class_repo_fp"
    assert diffs[1].is_fingerprint_only is True
