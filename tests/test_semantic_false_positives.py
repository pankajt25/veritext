"""Tests verifying semantic false-positive fixes, headline score rules, and lexical anchors.

Ensures:
1. Headline score ignores semantic matches by default.
2. The opt-in flag include_semantic_in_score includes them with explicit label 'score includes unverified semantic leads'.
3. The anchor rule counts shared rare terms, numbers, and distinctive entities correctly and rejects topic-only overlap.
4. The benchmark runner prints its effective settings.
"""

from unittest.mock import MagicMock, patch
import pytest

from veritext.eval.runner import BenchmarkRunner
from veritext.similarity.anchors import (
    EXPLANATORY_LINE,
    INTERPRETATION_GUIDE,
    evaluate_span_anchors,
    extract_content_lemmas,
    extract_numbers,
)
from veritext.similarity.models import MatchSpan, SimilarityResult, SourceMatch
from veritext.similarity.scorer import compute_similarity_score


def test_headline_score_ignores_semantic_matches_by_default():
    """Headline similarity score must ONLY count exact and near-copy matches by default."""
    clean_text = "This is a document evaluating headline similarity scoring rules and separate leads."

    span_exact = MatchSpan(
        type="exact",
        score=1.0,
        source_id="src_1",
        source_title="Source 1",
        source_url="",
        doc_start=0,
        doc_end=20,
        matched_text="This is a document e",
    )
    span_semantic = MatchSpan(
        type="possible paraphrase (review)",
        score=0.82,
        source_id="src_2",
        source_title="Source 2",
        source_url="",
        doc_start=25,
        doc_end=65,
        matched_text="valuating headline similarity scoring ru",
        metadata={
            "is_single_window": False,
            "consecutive_windows": 3,
            "lead_level": "strong lead",
        },
    )

    # Default calculation: semantic span is excluded from headline score
    result = compute_similarity_score(clean_text, [span_exact, span_semantic], min_match_chars=10)

    # Only span_exact (20 chars) should be counted in headline score
    assert result.matched_chars == 20
    expected_pct = (20 / len(clean_text)) * 100.0
    assert abs(result.similarity_percentage - expected_pct) < 0.1
    assert result.metadata["score_label"] == "exact & near-copy matches only"

    # Semantic matches must be recorded separately as paraphrase leads
    assert result.metadata["paraphrase_leads_count"] == 1
    assert result.metadata["paraphrase_leads_percentage"] > 0.0
    assert result.metadata["strong_leads_count"] == 1
    assert result.metadata["weak_leads_count"] == 0
    assert result.metadata["explanatory_line"] == EXPLANATORY_LINE
    assert result.metadata["interpretation_guide"] == INTERPRETATION_GUIDE

    # Per-source leads verification
    src2 = next((s for s in result.sources if s.source_id == "src_2"), None)
    assert src2 is not None
    assert src2.percentage == 0.0  # 0% headline overlap
    assert src2.lead_count == 1
    assert src2.strong_lead_count == 1


def test_optin_flag_includes_semantic_with_explicit_label():
    """When include_semantic_in_score=True, score includes semantic leads with explicit label."""
    clean_text = "This is a document evaluating headline similarity scoring rules and separate leads."

    span_exact = MatchSpan(
        type="exact",
        score=1.0,
        source_id="src_1",
        source_title="Source 1",
        source_url="",
        doc_start=0,
        doc_end=20,
        matched_text="This is a document e",
    )
    span_semantic = MatchSpan(
        type="possible paraphrase (review)",
        score=0.82,
        source_id="src_1",
        source_title="Source 1",
        source_url="",
        doc_start=20,
        doc_end=50,
        matched_text="valuating headline similarity ",
        metadata={
            "is_single_window": False,
            "consecutive_windows": 2,
            "lead_level": "weak lead",
        },
    )

    result = compute_similarity_score(
        clean_text,
        [span_exact, span_semantic],
        min_match_chars=10,
        include_semantic_in_score=True,
    )

    # Merged interval [0, 50] = 50 chars
    assert result.matched_chars == 50
    expected_pct = (50 / len(clean_text)) * 100.0
    assert abs(result.similarity_percentage - expected_pct) < 0.1

    # Explicit label must state unverified semantic leads
    assert result.metadata["score_label"] == "score includes unverified semantic leads"
    assert result.metadata["include_semantic_in_score"] is True


def test_anchor_rule_counts_shared_rare_terms_and_rejects_topic_only():
    """Anchor rule must count shared rare words, numbers, and entities, and reject topic-only overlap."""
    ref_df = {
        "attention": 15,    # Ubiquitous domain word
        "transformer": 12,  # Common word
        "mechanism": 10,    # Common word
        "sequence": 14,     # Common word
        "network": 18,      # Common word
        "model": 20,        # Common word
        "self": 25,         # Common word
        "recurrence": 2,    # Rare word (DF <= 2)
        "bottleneck": 1,    # Rare word (DF <= 2)
        "vaswani": 1,       # Named entity / rare term
    }

    # Case 1: Real paraphrase sharing rare words, numbers, and distinctive entities
    doc_real_para = (
        "In 2017, Vaswani introduced multi-head self-attention to bypass the "
        "informational recurrence bottleneck in sequence modeling."
    )
    src_real_para = (
        "Vaswani et al. (2017) demonstrated that removing recurrence eliminates "
        "the informational bottleneck via pairwise self-attention."
    )

    eval_real = evaluate_span_anchors(
        doc_text=doc_real_para,
        source_text=src_real_para,
        cosine_score=0.85,
        reference_df=ref_df,
        min_rare_anchors=2,
        min_content_ratio=0.15,
    )

    assert eval_real["is_strong_lead"] is True
    assert eval_real["lead_level"] == "strong lead"
    assert "recurrence" in eval_real["shared_rare_words"]
    assert "bottleneck" in eval_real["shared_rare_words"]
    assert "2017" in eval_real["shared_numbers"]

    # Case 2: Same-topic original sharing ubiquitous technical terminology but no anchors
    doc_same_topic = (
        "Sequence modeling in modern neural networks employs self-attention mechanisms "
        "to compute token representations dynamically across large contexts."
    )
    src_same_topic = (
        "Deep learning transformer networks utilize self-attention mechanisms to "
        "model dependencies across complex sequence structures."
    )

    eval_topic = evaluate_span_anchors(
        doc_text=doc_same_topic,
        source_text=src_same_topic,
        cosine_score=0.81,
        reference_df=ref_df,
        min_rare_anchors=2,
        min_content_ratio=0.30,  # Strict ratio or rare anchor requirement
    )

    # Shared lemmas are only generic domain words ('attention', 'mechanism', 'sequence'), no rare words
    assert len(eval_topic["shared_rare_words"]) == 0
    assert len(eval_topic["shared_numbers"]) == 0
    assert eval_topic["is_strong_lead"] is False
    assert eval_topic["lead_level"] == "weak lead"


def test_benchmark_prints_effective_settings(capsys, tmp_path):
    """The benchmark runner must print its effective semantic settings at startup."""
    runner = BenchmarkRunner(
        out_dir=tmp_path / "test_eval_out",
        semantic=True,
        ai=False,
        crosslang=False,
    )

    # Mock setup and evaluations to test runner startup banner
    with patch.object(runner, "setup", return_value=None), \
         patch.object(runner, "_evaluate_plagiarism", return_value={"winnowing_only": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "fpr": 0.0}}), \
         patch.object(runner, "_evaluate_ai", return_value={"stylometry_only": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "fpr_human_overall": 0.0, "fpr_non_native_esl": 0.0}}), \
         patch.object(runner, "_evaluate_spinner", return_value={"precision": 1.0, "recall": 1.0}), \
         patch.object(runner, "_evaluate_collusion", return_value={"expected_clusters_found": True}), \
         patch.object(runner, "_evaluate_seals", return_value={"tamper_detection_rate": 1.0}), \
         patch.object(runner, "_evaluate_crosslang", return_value=None), \
         patch.object(runner, "_save_results", return_value=None):

        summary = runner.run()

    captured = capsys.readouterr().out
    assert "Starting Veritext Evaluation Benchmark" in captured
    assert "Effective Semantic Settings:" in captured
    assert "Threshold (cosine):            0.78" in captured
    assert "Min consecutive windows:       2" in captured
    assert "Include semantic in headline:  False" in captured
    assert "Lexical anchor rule:           Evaluated" in captured

    # Verify settings recorded in summary JSON
    assert "effective_semantic_settings" in summary
    assert summary["effective_semantic_settings"]["cosine_threshold"] == 0.78
    assert summary["effective_semantic_settings"]["min_consecutive_windows"] == 2
    assert summary["effective_semantic_settings"]["include_semantic_in_headline_by_default"] is False
