"""Tests for text canonicalisation, homoglyph detection, zero-width stripping, and offset mapping."""

import pytest
from veritext.normalize.normalize import normalize_text


def test_empty_and_whitespace():
    res = normalize_text("")
    assert res.normalized_text == ""
    assert res.tokens == []
    assert res.offset_map == []
    assert res.anomalies == []

    res_spaces = normalize_text("    \n\t  ")
    assert res_spaces.normalized_text == ""
    assert res_spaces.tokens == []


def test_homoglyph_detection_and_folding():
    # Cyrillic 'а' (U+0430) and 'о' (U+043E) embedded in Latin "pаpеr"
    cyrillic_text = "Thіs pаpеr presents an approach."
    res = normalize_text(cyrillic_text)

    # Check anomalies detected
    homoglyphs = [a for a in res.anomalies if a.type == "homoglyph"]
    assert len(homoglyphs) >= 3

    # Normalized tokens should be standard Latin (and lemmatized)
    assert "this" in res.tokens
    assert "paper" in res.tokens
    assert "present" in res.tokens


def test_zero_width_and_control_stripping():
    # Text with zero-width space (U+200B) inside a word to evade detection, BOM (U+FEFF), and RTL override (U+202E)
    dirty_text = "Art\u200bificial Intelligence\ufeff \u202eResearch"
    res = normalize_text(dirty_text)

    # Verify anomalies recorded
    zw_anomalies = [a for a in res.anomalies if a.type == "zero_width"]
    ctrl_anomalies = [a for a in res.anomalies if a.type == "control_character"]
    assert len(zw_anomalies) >= 2
    assert len(ctrl_anomalies) >= 1

    # Tokens should be cleanly separated and repaired
    assert "artificial" in res.tokens
    assert "intelligence" in res.tokens
    assert "research" in res.tokens


def test_unicode_nfkc_normalization():
    # Ligature 'ﬁ' (U+FB01) and full-width letters
    ligature_text = "Scientific ﬁndings in ａｃａｄｅｍｉａ."
    res = normalize_text(ligature_text)

    assert "finding" in res.tokens
    assert "academia" in res.tokens


def test_lemmatization():
    text = "The neural networks are computing distributed representations across servers."
    res = normalize_text(text)

    # WordNet lemmatizer reduces plural nouns ("networks" -> "network", "servers" -> "server")
    assert "network" in res.tokens
    assert "server" in res.tokens
    assert "representation" in res.tokens


def test_offset_map_alignment():
    original_text = (
        "The transformer architecture uses self-attention. "
        "Intеllіgеncе (with Cyrillic) and clean text."
    )
    res = normalize_text(original_text)

    assert len(res.tokens) == len(res.offset_map)

    for token, (start, end) in zip(res.tokens, res.offset_map):
        assert 0 <= start < end <= len(original_text)
        orig_substring = original_text[start:end]
        # Re-normalizing the slice from original text should yield the same normalized token
        sub_norm = normalize_text(orig_substring)
        assert len(sub_norm.tokens) == 1
        assert sub_norm.tokens[0] == token


def test_punctuation_canonicalisation():
    curly_text = "“Smart quotes”, ‘single quotes’ — and em-dashes…"
    res = normalize_text(curly_text)

    assert "smart" in res.tokens
    assert "quote" in res.tokens
    assert "single" in res.tokens
