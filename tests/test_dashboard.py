"""Tests for combined dashboard (paragraph heat strip and cohort portfolio table)."""

import pytest

from veritext.report.dashboard import (
    generate_paragraph_heat_strip,
    generate_portfolio_csv,
    render_heat_strip_html,
    render_portfolio_html,
    split_text_into_paragraphs,
)
from veritext.similarity.models import MatchSpan, SimilarityResult, SourceMatch


def test_split_text_into_paragraphs():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
    paras = split_text_into_paragraphs(text)
    assert len(paras) == 3
    assert paras[0][3] == "First paragraph here."
    assert paras[1][3] == "Second paragraph here."
    assert paras[2][3] == "Third paragraph."


def test_generate_paragraph_heat_strip():
    text = (
        "Paragraph one is completely original and contains no matches.\n\n"
        "Paragraph two contains a matched span from an external academic source.\n\n"
        "Paragraph three is another block of original text."
    )
    paras = split_text_into_paragraphs(text)
    p2_start = paras[1][1]
    p2_end = paras[1][2]

    span = MatchSpan(
        type="exact",
        score=1.0,
        source_id="src_wiki",
        source_title="Wikipedia",
        source_url="https://en.wikipedia.org",
        doc_start=p2_start,
        doc_end=p2_start + 30,
        matched_text=text[p2_start : p2_start + 30],
    )
    sim = SimilarityResult(
        total_clean_chars=len(text),
        matched_chars=30,
        similarity_percentage=15.0,
        spans=[span],
        sources=[SourceMatch("src_wiki", "Wikipedia", "https://en.wikipedia.org", "wikipedia", 30, 15.0, 1, [span])],
    )

    heat_strip = generate_paragraph_heat_strip(text, similarity=sim)
    assert len(heat_strip) == 3
    # P1: 0% similarity
    assert heat_strip[0]["similarity_percentage"] == 0.0
    # P2: > 0% similarity
    assert heat_strip[1]["similarity_percentage"] > 0.0
    # P3: 0% similarity
    assert heat_strip[2]["similarity_percentage"] == 0.0

    # HTML render check
    html = render_heat_strip_html(heat_strip)
    assert "heat-strip-container" in html
    assert "Wikipedia" not in html  # heat strip displays bars and flags


def test_portfolio_csv_and_html():
    items = [
        {
            "file_name": "student_alpha.pdf",
            "similarity_percentage": 12.5,
            "ai_probability": 5.0,
            "integrity_findings_count": 0,
            "max_collusion_similarity": 8.0,
            "status": "Passed",
        },
        {
            "file_name": "student_beta.docx",
            "similarity_percentage": 52.0,
            "ai_probability": 84.0,
            "integrity_findings_count": 3,
            "max_collusion_similarity": 68.0,
            "status": "Needs Review",
        },
    ]

    csv_data = generate_portfolio_csv(items)
    assert "student_alpha.pdf,12.5,5.0,0,8.0,Passed" in csv_data
    assert "student_beta.docx,52.0,84.0,3,68.0,Needs Review" in csv_data

    html_data = render_portfolio_html(items)
    assert "portfolio-container" in html_data
    assert "student_alpha.pdf" in html_data
    assert "sortPortfolioTable" in html_data
    assert "filterPortfolioTable" in html_data
    assert "data:text/csv;base64," in html_data
