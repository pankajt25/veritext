"""Tests for review workflow, annotations, rubric assessments, and views."""

import json
from pathlib import Path
import pytest

from veritext.review import (
    FindingReview,
    ReviewSession,
    ReviewStatus,
    RubricAssessment,
    RubricCriterion,
    RubricScore,
    format_finding_for_view,
    format_report_for_view,
    format_span_for_view,
    get_default_criteria,
    load_review_session,
    load_rubric_template,
    sanitize_text_for_student,
    save_review_session,
)
from veritext.similarity.filters import ExclusionFilters, recalculate_similarity_with_review
from veritext.similarity.models import MatchSpan, SimilarityResult, SourceMatch


def test_review_session_persistence(tmp_path):
    session = ReviewSession(
        report_id="rep_12345",
        document_name="essay.pdf",
        author="Prof. Smith",
    )
    session.set_review(
        target_id="span_10_50",
        status="accepted",
        comment="Genuine unquoted match from Wikipedia",
        author="Prof. Smith",
        target_type="span",
        start=10,
        end=50,
    )
    session.set_review(
        target_id="finding_spinner_100_140",
        status="ignored",
        comment="False positive, legitimate terminology",
        target_type="finding",
        start=100,
        end=140,
    )

    # Save
    saved_path = save_review_session(session, storage_dir=tmp_path)
    assert saved_path.is_file()
    assert saved_path.name == "rep_12345.json"

    # Reload
    reloaded = load_review_session("rep_12345", storage_dir=tmp_path)
    assert reloaded is not None
    assert reloaded.report_id == "rep_12345"
    assert reloaded.author == "Prof. Smith"
    assert len(reloaded.reviews) == 2
    assert reloaded.is_accepted("span_10_50")
    assert reloaded.is_ignored("finding_spinner_100_140")
    assert "finding_spinner_100_140" in reloaded.get_ignored_target_ids()


def test_student_vs_instructor_view():
    report_data = {
        "report_id": "rep_test",
        "overall_risk": "high",
        "summary": "Plagiarism detected: 45% similarity with suspicious collusion and tortured phrases.",
        "similarity": {
            "spans": [
                {
                    "type": "exact",
                    "source_id": "src_1",
                    "source_title": "Academic Journal",
                    "doc_start": 0,
                    "doc_end": 20,
                    "matched_text": "Exact copied sentence.",
                    "metadata": {"id": "span_0_20"},
                }
            ]
        },
        "integrity": {
            "findings": [
                {
                    "id": "find_1",
                    "check": "spinner_tortured_phrases",
                    "severity": "warning",
                    "message": "Tortured phrase 'counterfeit consciousness' detected (possible automated paraphrasing).",
                    "start": 50,
                    "end": 80,
                }
            ]
        },
    }

    # Instructor View
    inst_view = format_report_for_view(report_data, view_mode="instructor")
    assert inst_view["view_mode"] == "instructor"
    assert inst_view["overall_risk"] == "high"
    assert "Plagiarism" in inst_view["summary"]
    assert inst_view["similarity"]["spans"][0]["guidance"] is None

    # Student View
    stud_view = format_report_for_view(report_data, view_mode="student")
    assert stud_view["view_mode"] == "student"
    assert "Plagiarism" not in stud_view["summary"]
    assert "unattributed matching text" in stud_view["summary"]
    assert stud_view["overall_risk"] == "Review and Revision Recommended"

    # Span in student view contains developmental guidance
    span = stud_view["similarity"]["spans"][0]
    assert span["guidance"] is not None
    assert "quotation marks" in span["guidance"] or "citation" in span["guidance"]

    # Finding in student view has sanitized message and constructive guidance
    finding = stud_view["integrity"]["findings"][0]
    assert "Tortured phrase" not in finding["message"]
    assert "unconventional phrasing" in finding["message"]
    assert finding["display_severity"] == "Action Recommended"
    assert "Guidance:" in finding["guidance"]


def test_rubric_scoring_and_assessment():
    criteria = get_default_criteria()
    assert len(criteria) >= 4
    criterion_ids = {c.id for c in criteria}
    assert "originality" in criterion_ids
    assert "citation_quality" in criterion_ids
    assert "use_of_sources" in criterion_ids
    assert "writing_clarity" in criterion_ids

    # Fill scores
    scores = {
        "originality": RubricScore(criterion_id="originality", score=3, comment="Good originality."),
        "citation_quality": RubricScore(criterion_id="citation_quality", score=4, comment="Exemplary references."),
        "use_of_sources": RubricScore(criterion_id="use_of_sources", score=3, comment="Well synthesized."),
        "writing_clarity": RubricScore(criterion_id="writing_clarity", score=2, comment="A few passive clauses."),
    }
    assessment = RubricAssessment(
        name="Standard Academic Integrity & Writing Rubric",
        scores=scores,
        overall_comment="Well written paper overall.",
    )
    assert assessment.total_score == 12
    assert assessment.max_possible == 16
    assert assessment.percentage == 75.0

    d = assessment.to_dict()
    assert d["total_score"] == 12
    assert d["percentage"] == 75.0


def test_ignored_findings_excluded_from_similarity_recalculation():
    clean_text = "This is a sentence containing sixty characters for testing here."
    span1 = MatchSpan(
        type="exact",
        score=1.0,
        source_id="src_a",
        source_title="Source A",
        source_url="http://a.com",
        doc_start=0,
        doc_end=20,
        matched_text=clean_text[0:20],
        metadata={"id": "span_0_20"},
    )
    span2 = MatchSpan(
        type="exact",
        score=1.0,
        source_id="src_b",
        source_title="Source B",
        source_url="http://b.com",
        doc_start=30,
        doc_end=50,
        matched_text=clean_text[30:50],
        metadata={"id": "span_30_50"},
    )

    sim = SimilarityResult(
        total_clean_chars=len(clean_text),
        matched_chars=40,
        similarity_percentage=40.0,
        spans=[span1, span2],
        sources=[
            SourceMatch("src_a", "Source A", "http://a.com", "academic", 20, 20.0, 1, [span1]),
            SourceMatch("src_b", "Source B", "http://b.com", "academic", 20, 20.0, 1, [span2]),
        ],
        metadata={"all_original_spans": [span1, span2]},
    )

    # Initial recalculation: both spans active
    filters = ExclusionFilters(exclude_small_matches=False)
    res_all = recalculate_similarity_with_review(sim, clean_text, filters=filters)
    assert len(res_all.spans) == 2
    assert round(res_all.similarity_percentage, 1) > 30.0

    # Mark span_0_20 as ignored in a ReviewSession
    session = ReviewSession(report_id="test_sim")
    session.set_review(
        target_id="span_0_20",
        status=ReviewStatus.IGNORED,
        comment="Legitimate boilerplate",
    )

    res_filtered = recalculate_similarity_with_review(sim, clean_text, review_session=session, filters=filters)
    # Span1 should be excluded now
    assert len(res_filtered.spans) == 1
    assert res_filtered.spans[0].source_id == "src_b"
    assert res_filtered.similarity_percentage < res_all.similarity_percentage
