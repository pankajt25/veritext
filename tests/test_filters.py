"""Tests for extended exclusion filters, instant recalculation, and allowlist persistence."""

from pathlib import Path
from veritext.report.models import FullReport, HighlightSpan
from veritext.similarity.filters import (
    ExclusionFilters,
    add_to_allowlist,
    load_allowlist,
    recalculate_report_filters,
    recalculate_similarity,
    remove_from_allowlist,
)
from veritext.similarity.models import MatchSpan, SimilarityResult, SourceMatch


def test_allowlist_persistence(tmp_path):
    cfg = tmp_path / "allowlist.json"
    assert load_allowlist(cfg) == []

    # Add phrase
    add_to_allowlist("data availability statement", cfg)
    add_to_allowlist("authors declare no competing interests", cfg)

    loaded = load_allowlist(cfg)
    assert len(loaded) == 2
    assert "data availability statement" in loaded

    # Remove phrase
    remove_from_allowlist("data availability statement", cfg)
    loaded_after = load_allowlist(cfg)
    assert len(loaded_after) == 1
    assert "data availability statement" not in loaded_after
    assert "authors declare no competing interests" in loaded_after


def test_exclusion_filters_recalculation_instant():
    clean_text = (
        "0123456789" * 10
    )  # 100 characters

    # Create spans:
    # Span 1: source_1 (char 0 to 30) - length 30
    span1 = MatchSpan(
        type="exact",
        score=1.0,
        source_id="source_1",
        source_title="Source One",
        source_url="https://src1.org",
        doc_start=0,
        doc_end=30,
        matched_text="0123456789" * 3,
    )
    # Span 2: source_2 (char 40 to 70) - length 30
    span2 = MatchSpan(
        type="exact",
        score=1.0,
        source_id="source_2",
        source_title="Source Two",
        source_url="https://src2.org",
        doc_start=40,
        doc_end=70,
        matched_text="0123456789" * 3,
    )
    # Span 3: small span (char 80 to 90) - length 10 (<25)
    span3 = MatchSpan(
        type="exact",
        score=1.0,
        source_id="source_3",
        source_title="Source Three",
        source_url="",
        doc_start=80,
        doc_end=90,
        matched_text="0123456789",
    )
    # Span 4: template prompt match (char 90 to 100) - length 10
    span4 = MatchSpan(
        type="exact",
        score=1.0,
        source_id="source_4",
        source_title="Source Four",
        source_url="",
        doc_start=90,
        doc_end=100,
        matched_text="assignment prompt boilerplate",
    )

    all_spans = [span1, span2, span3, span4]

    initial_sim = SimilarityResult(
        total_clean_chars=100,
        matched_chars=70,
        similarity_percentage=70.0,
        spans=all_spans,
        sources=[
            SourceMatch("source_1", "Source One", "https://src1.org", "web", 30, 30.0, 1, [span1]),
            SourceMatch("source_2", "Source Two", "https://src2.org", "web", 30, 30.0, 1, [span2]),
        ],
        metadata={"all_original_spans": all_spans},
    )

    # 1. Base filter: exclude small matches only
    f_base = ExclusionFilters(exclude_small_matches=True, min_match_chars=25)
    r_base = recalculate_similarity(initial_sim, clean_text, f_base)
    # span1 (30) + span2 (30) = 60 chars = 60%
    assert r_base.similarity_percentage == 60.0
    assert len(r_base.sources) == 2

    # 2. Exclude source_1 in one click
    f_ex_source = ExclusionFilters(
        exclude_small_matches=True,
        min_match_chars=25,
        exclude_sources={"source_1"},
    )
    r_ex_source = recalculate_similarity(initial_sim, clean_text, f_ex_source)
    # only span2 (30) survives = 30%
    assert r_ex_source.similarity_percentage == 30.0
    assert len(r_ex_source.sources) == 1
    assert r_ex_source.sources[0].source_id == "source_2"

    # 3. Exclude template
    f_tpl = ExclusionFilters(
        exclude_small_matches=False,
        exclude_template=True,
        template_text="Please submit your assignment prompt boilerplate here.",
    )
    r_tpl = recalculate_similarity(initial_sim, clean_text, f_tpl)
    # span4 is excluded
    surviving_ids = [s.source_id for s in r_tpl.spans]
    assert "source_4" not in surviving_ids

    # 4. Exclude allowlist
    span_al = MatchSpan(
        type="exact",
        score=1.0,
        source_id="src_al",
        source_title="Src AL",
        source_url="",
        doc_start=0,
        doc_end=35,
        matched_text="data availability statement is clear",
    )
    sim_with_al = SimilarityResult(
        total_clean_chars=100,
        matched_chars=35,
        similarity_percentage=35.0,
        spans=[span_al],
        sources=[],
        metadata={"all_original_spans": [span_al]},
    )
    f_al = ExclusionFilters(
        exclude_small_matches=False,
        exclude_allowlist=True,
        allowlist=["data availability statement"],
    )
    r_al = recalculate_similarity(sim_with_al, clean_text, f_al)
    assert r_al.similarity_percentage == 0.0


def test_recalculate_report_filters():
    doc_text = "Standard academic analysis on machine learning algorithms."
    span = MatchSpan(
        type="exact",
        score=1.0,
        source_id="src_99",
        source_title="Source 99",
        source_url="",
        doc_start=0,
        doc_end=20,
        matched_text=doc_text[:20],
    )
    hl = HighlightSpan(
        id="span_1",
        start=0,
        end=20,
        category="exact match",
        severity="high",
        message="Exact match",
        source="Source 99",
        details={"engine": "similarity", "source_id": "src_99"},
    )
    sim = SimilarityResult(
        total_clean_chars=len(doc_text),
        matched_chars=20,
        similarity_percentage=35.0,
        spans=[span],
        sources=[SourceMatch("src_99", "Source 99", "", "local", 20, 35.0, 1, [span])],
        metadata={"all_original_spans": [span]},
    )

    report = FullReport(
        document_name="doc.txt",
        document_path="/tmp/doc.txt",
        original_text=doc_text,
        clean_text=doc_text,
        similarity=sim,
        highlights=[hl],
        summary="Initial summary",
    )

    # Exclude source 99
    filters = ExclusionFilters(exclude_sources={"src_99"})
    updated_report = recalculate_report_filters(report, filters)

    assert updated_report.similarity.similarity_percentage == 0.0
    assert updated_report.highlights[0].ignored is True
    assert "Excluded sources (1)" in updated_report.settings["active_filters"]
    assert "0.0%" in updated_report.summary
