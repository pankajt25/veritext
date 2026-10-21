"""Fast, offline tests for Stage 6 report generator, exports, highlighting, and Streamlit UI."""

import html
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from streamlit.testing.v1 import AppTest

from veritext.ai_detect.models import AIResult, SentenceAIResult
from veritext.checks.models import CheckResult, Finding, IntegrityReport
from veritext.ingest.models import Document, Sentence
from veritext.report.builder import build_report
from veritext.report.highlight import CATEGORY_STYLES, render_highlighted_html, render_legend_html
from veritext.report.html_export import export_report_html
from veritext.report.models import FullReport, HighlightSpan
from veritext.report.pdf_export import export_report_pdf, sanitize_text_for_pdf
from veritext.similarity.models import MatchSpan, SimilarityResult, SourceMatch
from veritext.ui.services import run_pipeline


@pytest.fixture
def sample_report():
    raw = "The quick brown fox jumps over the lazy dog. In contrast, automated neural systems analyze language."
    spans = [
        HighlightSpan(
            id="span_1",
            start=4,
            end=19,
            category="exact match",
            severity="high",
            message="Verbatim match with encyclopedia",
            source="Encyclopedia Britannica",
        ),
        HighlightSpan(
            id="span_2",
            start=45,
            end=99,
            category="likely AI",
            severity="warning",
            message="Low burstiness sentence",
            source="Ensemble Model",
        ),
        HighlightSpan(
            id="span_3",
            start=70,
            end=84,
            category="hidden content",
            severity="high",
            message="Microscopic font detected",
            source="DOCX Inspector",
        ),
    ]
    sim = SimilarityResult(
        document_path="sample.txt",
        total_clean_chars=len(raw),
        matched_chars=15,
        similarity_percentage=15.0,
        sources=[
            SourceMatch(
                source_id="src_1",
                source_title="Encyclopedia Britannica",
                source_url="https://example.com/source",
                origin="web",
                matched_chars=15,
                percentage=15.0,
                match_count=1,
                spans=[],
            )
        ],
    )
    ai = AIResult(
        status="success",
        word_count=len(raw.split()),
        overall_probability=75.4,
        confidence_band="medium",
        reduced_confidence=False,
        mode="full",
        explanations=["Low token perplexity variance"],
        sentences=[
            SentenceAIResult(
                text="In contrast, automated neural systems analyze language.",
                doc_start=45,
                doc_end=99,
                score=0.88,
                label="likely AI",
            )
        ],
    )
    integ = IntegrityReport(
        document_path="sample.txt",
        results=[
            CheckResult(
                check_name="hidden_content",
                findings=[
                    Finding(
                        check="hidden_content",
                        severity="high",
                        message="Microscopic font detected",
                        start=70,
                        end=84,
                    )
                ],
                summary="Detected 1 hidden run",
                passed=False,
            )
        ],
        summary="1 high severity issue",
        passed=False,
    )
    return FullReport(
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        original_text=raw,
        clean_text=raw,
        similarity=sim,
        ai_detect=ai,
        integrity=integ,
        highlights=spans,
        summary="Test Executive Summary",
    )


# ==============================================================================
# 1. Report Builder & Overlap Merging
# ==============================================================================

def test_report_builder_merging_spans():
    raw_text = "Natural language processing evaluates linguistic structure and syntactic patterns."
    doc = Document(
        raw_text=raw_text,
        clean_text=raw_text,
        sentences=[Sentence(text=raw_text, start=0, end=len(raw_text))],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )
    sim = SimilarityResult(
        document_path="test.txt",
        total_clean_chars=len(raw_text),
        matched_chars=20,
        similarity_percentage=25.0,
        sources=[
            SourceMatch(
                source_id="s1",
                source_title="NLP Primer",
                source_url="",
                origin="local",
                matched_chars=20,
                percentage=25.0,
                match_count=1,
                spans=[
                    MatchSpan(
                        type="exact",
                        score=1.0,
                        source_id="s1",
                        source_title="NLP Primer",
                        source_url="",
                        doc_start=0,
                        doc_end=27,
                        matched_text="Natural language processing",
                    )
                ],
            )
        ],
    )
    ai = AIResult(
        status="success",
        word_count=len(raw_text.split()),
        overall_probability=82.0,
        confidence_band="high",
        reduced_confidence=False,
        mode="full",
        sentences=[
            SentenceAIResult(
                text=raw_text,
                doc_start=0,
                doc_end=len(raw_text),
                score=0.82,
                label="likely AI",
            )
        ],
    )
    integ = IntegrityReport(
        document_path="test.txt",
        results=[
            CheckResult(
                check_name="citations",
                findings=[
                    Finding(
                        check="citations",
                        severity="warning",
                        message="Uncited empirical claim",
                        start=38,
                        end=58,
                    )
                ],
            )
        ],
    )

    report = build_report(doc, similarity=sim, ai=ai, integrity=integ)
    assert isinstance(report, FullReport)
    assert len(report.highlights) == 3

    categories = {h.category for h in report.highlights}
    assert "exact match" in categories
    assert "likely AI" in categories
    assert "citation issue" in categories

    # All highlights sorted by start offset
    for i in range(len(report.highlights) - 1):
        assert report.highlights[i].start <= report.highlights[i + 1].start


# ==============================================================================
# 2. HTML Escaping & Overlapping Spans
# ==============================================================================

def test_html_escaping_hostile_document():
    hostile_text = (
        'Malicious payload: <script>alert("xss")</script> & <img src=x onerror=alert(1)> '
        'plus attribute injection: "><svg onload=alert(2)>'
    )
    spans = [
        HighlightSpan(
            id="span_h1",
            start=19,
            end=49,
            category="hidden content",
            severity="high",
            message='Dangerous tag <script> in message "quoted"',
        )
    ]
    rendered = render_highlighted_html(hostile_text, spans)

    # Must NOT contain raw script tag or unescaped angle brackets
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<img" not in rendered
    assert "&lt;img" in rendered
    assert "<svg" not in rendered


def test_overlapping_spans_rendering():
    text = "Machine learning models analyze natural language syntax."
    spans = [
        HighlightSpan(
            id="sp1",
            start=0,
            end=23,  # "Machine learning models"
            category="exact match",
            severity="high",
            message="Exact copy",
        ),
        HighlightSpan(
            id="sp2",
            start=17,
            end=39,  # "models analyze natural"
            category="likely AI",
            severity="warning",
            message="AI pattern",
        ),
    ]
    rendered = render_highlighted_html(text, spans)
    assert rendered.count("<mark") >= 2
    assert "hl-exact-match" in rendered
    assert "hl-likely-AI" in rendered


def test_palette_and_legend_present():
    legend = render_legend_html()
    for cat_name, style in CATEGORY_STYLES.items():
        assert style["label"] in legend
        assert style["bg"] in legend
        assert "border-bottom" in legend


# ==============================================================================
# 3. HTML Export Self-Containment
# ==============================================================================

def test_html_export_self_contained(sample_report):
    html_out = export_report_html(sample_report)
    assert "<!DOCTYPE html>" in html_out
    assert sample_report.document_name in html_out
    assert "Plagiarism Similarity" in html_out
    assert "AI Content Probability" in html_out

    # Ensure completely self-contained: no external stylesheets, scripts, or fonts
    assert "<link rel=\"stylesheet\" href=\"http" not in html_out
    assert "<script src=\"http" not in html_out
    assert "@import url(\"http" not in html_out


# ==============================================================================
# 4. PDF Export & Non-Latin Character Safety
# ==============================================================================

def test_pdf_export_survives_non_latin(sample_report):
    # Injected non-Latin characters: Cyrillic, Greek, zero-width, Chinese, emoji
    sample_report.original_text = (
        "Standard English text. "
        "Adversarial Cyrillic: syst\u0435m, Greek: \u03b1\u03b2\u03b3. "
        "Zero-width: zero\u200bwidth. Chinese: \u4e2d\u6587. Emoji: \U0001f680."
    )
    pdf_bytes = export_report_pdf(sample_report)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_sanitize_text_for_pdf():
    input_str = "Curly quotes: “hello” & ‘world’ — em-dash and Cyrillic: systеm"
    sanitized = sanitize_text_for_pdf(input_str)
    assert '"hello"' in sanitized
    assert "'world'" in sanitized
    assert "-" in sanitized
    # Non-Latin Cyrillic 'е' replaced by '?'
    assert "?" in sanitized


# ==============================================================================
# 5. JSON Round-Trip & Ignored Spans
# ==============================================================================

def test_json_round_trip(sample_report):
    json_str = sample_report.to_json()
    parsed = json.loads(json_str)
    assert parsed["document_name"] == sample_report.document_name

    restored = FullReport.from_dict(parsed)
    assert restored.document_name == sample_report.document_name
    assert len(restored.highlights) == len(sample_report.highlights)
    assert restored.similarity.similarity_percentage == sample_report.similarity.similarity_percentage
    assert restored.ai_detect.overall_probability == sample_report.ai_detect.overall_probability


def test_ignored_findings_excluded_from_exports(sample_report):
    ignored = {"span_1"}
    # HTML export with span_1 ignored
    html_out = export_report_html(sample_report, ignored_ids=ignored)
    assert 'class="hl-exact-match"' not in html_out
    assert 'class="hl-likely-AI"' in html_out

    # PDF export with span_1 ignored
    pdf_bytes = export_report_pdf(sample_report, ignored_ids=ignored)
    assert len(pdf_bytes) > 500


# ==============================================================================
# 6. Service Layer & Adaptive Profiling
# ==============================================================================

def test_services_run_pipeline_profile_and_plan():
    # Generate text with 5500 words
    long_text = "word " * 5500
    with patch("veritext.ui.services.check_document") as mock_sim, \
         patch("veritext.ui.services.detect_ai") as mock_ai, \
         patch("veritext.ui.services.run_all_checks") as mock_integ:

        mock_sim.return_value = None
        mock_ai.return_value = None
        mock_integ.return_value = None

        report = run_pipeline(
            target=long_text.encode("utf-8"),
            filename="long_doc.txt",
            settings={"use_ai": False, "use_integrity": False},
        )
        assert report.profile is not None
        assert report.profile.get("word_count") == 5500
        assert report.plan is not None
        assert len(report.original_text.split()) == 5500


# ==============================================================================
# 7. Streamlit UI Smoke Test (AppTest)
# ==============================================================================

def test_streamlit_app_smoke():
    app_path = Path(__file__).resolve().parent.parent / "src" / "veritext" / "ui" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=60)

    # Verify app loads without unhandled exceptions
    assert not at.exception

    # Verify permanent responsible use disclaimer is visible
    assert any("Responsible Use" in info.value for info in at.info)

    # Click run button without uploading file -> should show error
    at.sidebar.button[0].click().run(timeout=60)
    assert any("Please upload a document" in err.value for err in at.error)


# ==============================================================================
# 8. Report CLI Smoke Test
# ==============================================================================

def test_report_cli_smoke(sample_report):
    from veritext.report.__main__ import main

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_file = tmp_path / "test_doc.txt"
        test_file.write_text("This is an academic test document for the CLI runner.", encoding="utf-8")

        code = main([str(test_file), "--out-dir", str(tmp_path), "--offline"])
        assert code == 0

        # Verify all 3 files generated
        json_files = list(tmp_path.glob("*.json"))
        html_files = list(tmp_path.glob("*.html"))
        pdf_files = list(tmp_path.glob("*.pdf"))

        assert len(json_files) == 1
        assert len(html_files) == 1
        assert len(pdf_files) == 1
