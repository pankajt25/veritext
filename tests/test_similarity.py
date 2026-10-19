"""Tests for exact match clustering, homoglyph obfuscation detection, interval union math, and JSON serialization."""

import json
import pytest
from veritext.fingerprint.index import FingerprintIndex
from veritext.fingerprint.winnow import winnow
from veritext.normalize.normalize import normalize_text
from veritext.similarity.exact import find_exact_matches
from veritext.similarity.models import MatchSpan, SimilarityResult, SourceMatch
from veritext.similarity.scorer import (
    calculate_covered_characters,
    compute_similarity_score,
    merge_character_intervals,
)
from veritext.sources.models import SourceCandidate


def test_merge_character_intervals():
    # Empty
    assert merge_character_intervals([]) == []

    # Disjoint intervals
    intervals = [(0, 10), (20, 30), (40, 50)]
    assert merge_character_intervals(intervals) == [(0, 10), (20, 30), (40, 50)]

    # Overlapping and contiguous intervals
    intervals = [(0, 15), (10, 25), (30, 40), (40, 50)]
    # (0, 15) and (10, 25) merge to (0, 25)
    # (30, 40) and (40, 50) merge to (30, 50)
    assert merge_character_intervals(intervals) == [(0, 25), (30, 50)]

    # Out of order
    intervals = [(30, 45), (5, 12), (10, 20)]
    assert merge_character_intervals(intervals) == [(5, 20), (30, 45)]

    # Character count calculation
    assert calculate_covered_characters([(0, 10), (5, 15), (20, 30)]) == 25


def test_exact_match_clustering(tmp_path):
    index = FingerprintIndex(tmp_path / "exact_test.sqlite")

    source_text = (
        "The transformer model completely dispenses with recurrence and convolutions, "
        "relying entirely on an attention mechanism to model dependencies between input and output."
    )
    src_cand = SourceCandidate(
        id="src_exact_1",
        title="Attention Paper",
        authors=["Vaswani et al."],
        url="https://arxiv.org/abs/1706.03762",
        origin="arxiv",
        year=2017,
        text=source_text,
    )
    index.index_source_candidate(src_cand)

    # Document contains exact copy embedded in introductory text
    doc_text = (
        "Introduction to Modern NLP. "
        "The transformer model completely dispenses with recurrence and convolutions, "
        "relying entirely on an attention mechanism to model dependencies between input and output. "
        "Future research explores deeper layers."
    )
    canonical = normalize_text(doc_text)
    doc_fps = winnow(canonical.tokens, k=5, w=4)

    sources_map = {src_cand.id: src_cand}
    spans = find_exact_matches(
        doc_clean_text=doc_text,
        canonical=canonical,
        doc_fingerprints=doc_fps,
        index=index,
        sources_map=sources_map,
        min_words=8,
    )

    assert len(spans) >= 1
    span = spans[0]
    assert span.type == "exact"
    assert span.source_id == "src_exact_1"
    assert "completely dispenses with recurrence and convolutions" in span.matched_text


def test_homoglyph_obfuscated_match(tmp_path):
    index = FingerprintIndex(tmp_path / "homoglyph_test.sqlite")

    source_text = (
        "Distributed gradient descent aggregates parameters across worker nodes "
        "to optimize loss functions without central memory bottlenecks."
    )
    src_cand = SourceCandidate(
        id="src_homo_1",
        title="Distributed Optimization",
        authors=["Author A"],
        url="file://ref.txt",
        origin="local_library",
        year=2020,
        text=source_text,
    )
    index.index_source_candidate(src_cand)

    # Obfuscate target text by replacing Latin 'a', 'e', 'o' with Cyrillic lookalikes
    # 'a' -> '\u0430', 'e' -> '\u0435', 'o' -> '\u043e'
    obfuscated_doc = (
        "Dіstributеd grаdіеnt dеscеnt аggrеgаtеs pаrаmеtеrs аcrоss wоrkеr nоdеs "
        "tо оptіmіzе lоss functіоns wіthоut cеntrаl mеmоry bоttlеnеcks."
    )

    canonical = normalize_text(obfuscated_doc)
    assert len(canonical.anomalies) > 5  # Homoglyphs detected

    doc_fps = winnow(canonical.tokens, k=5, w=4)
    sources_map = {src_cand.id: src_cand}

    spans = find_exact_matches(
        doc_clean_text=obfuscated_doc,
        canonical=canonical,
        doc_fingerprints=doc_fps,
        index=index,
        sources_map=sources_map,
        min_words=8,
    )

    assert len(spans) >= 1
    # Matched span should span the obfuscated text cleanly
    assert spans[0].source_id == "src_homo_1"


def test_unrelated_text_scores_zero(tmp_path):
    index = FingerprintIndex(tmp_path / "unrelated_test.sqlite")

    source_text = (
        "Quantum superposition allows qubits to represent linear combinations "
        "of computational basis states simultaneously during coherent unitaries."
    )
    src_cand = SourceCandidate(
        id="src_quantum",
        title="Quantum Physics",
        authors=["Nielsen and Chuang"],
        url="",
        origin="local_library",
        year=2010,
        text=source_text,
    )
    index.index_source_candidate(src_cand)

    unrelated_doc = (
        "Medieval agriculture in northwestern Europe relied heavily on three-field "
        "crop rotation systems and heavy wheeled iron plows."
    )
    canonical = normalize_text(unrelated_doc)
    doc_fps = winnow(canonical.tokens, k=5, w=4)

    sources_map = {src_cand.id: src_cand}
    spans = find_exact_matches(
        doc_clean_text=unrelated_doc,
        canonical=canonical,
        doc_fingerprints=doc_fps,
        index=index,
        sources_map=sources_map,
        min_words=8,
    )
    assert len(spans) == 0

    result = compute_similarity_score(unrelated_doc, spans, sources_map=sources_map)
    assert result.similarity_percentage == 0.0
    assert result.matched_chars == 0


def test_scorer_and_json_serialization():
    clean_text = "This is a document for testing hybrid scoring and JSON export capabilities."
    span1 = MatchSpan(
        type="exact",
        score=1.0,
        source_id="src_1",
        source_title="Source Alpha",
        source_url="https://alpha.org",
        doc_start=0,
        doc_end=25,
        matched_text="This is a document for te",
    )
    span2 = MatchSpan(
        type="paraphrase",
        score=0.88,
        source_id="src_1",
        source_title="Source Alpha",
        source_url="https://alpha.org",
        doc_start=20,  # Overlaps with span1 from 20 to 25!
        doc_end=45,
        matched_text="r testing hybrid scoring",
    )

    result = compute_similarity_score(clean_text, [span1, span2], min_match_chars=10)
    # Merged interval should be [0, 45], length = 45 characters without double counting!
    assert result.matched_chars == 45
    expected_pct = (45 / len(clean_text)) * 100.0
    assert abs(result.similarity_percentage - expected_pct) < 0.1

    # Check JSON export
    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["matched_chars"] == 45
    assert len(parsed["sources"]) == 1
    assert parsed["sources"][0]["source_id"] == "src_1"
