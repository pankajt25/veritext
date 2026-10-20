"""Tests for Stage 5 integrity checks (hidden content, citations/quotes, self-plagiarism, code similarity)."""

import json
from pathlib import Path
import tempfile
import pytest

from veritext.checks.citations import (
    check_citations_and_quotes,
    detect_in_text_citations,
    parse_reference_section,
)
from veritext.checks.code_similarity import (
    compare_code_directory,
    compare_code_files,
    tokenize_file,
    tokenize_python,
    tokenize_generic,
)
from veritext.checks.hidden import check_hidden_content
from veritext.checks.models import CheckResult, Finding, IntegrityReport
from veritext.checks.runner import run_all_checks
from veritext.checks.selfcheck import add_previous, check_self_plagiarism
from veritext.ingest.models import Document, RemovedSection, Sentence
from veritext.similarity.models import MatchSpan, SourceMatch, SimilarityResult


@pytest.fixture
def demo_dir():
    return Path(__file__).resolve().parent.parent / "data" / "samples" / "checks_demo"


# ==============================================================================
# 1. Hidden Content & Obfuscation Tests
# ==============================================================================

def test_hidden_character_grouping_and_severities():
    # 1. Stray invisible character -> info
    doc_stray = Document(
        raw_text="This is a simple text with a single zero\u200bwidth space.",
        clean_text="This is a simple text with a single zero width space.",
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )
    res_stray = check_hidden_content(doc_stray)
    assert len(res_stray.findings) == 1
    assert res_stray.findings[0].severity == "info"

    # 2. Many invisible characters (>= 3) -> high
    doc_many = Document(
        raw_text="Adversarial text with zero\u200bwidth\u200bspace and word\u2060joiners.",
        clean_text="Adversarial text with zero width space and word joiners.",
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )
    res_many = check_hidden_content(doc_many)
    assert any(f.severity == "high" for f in res_many.findings)

    # 3. Mixed-script word with Cyrillic homoglyph -> high
    doc_mixed = Document(
        raw_text="The automated syst\u0435m analyzed the document.",
        clean_text="The automated system analyzed the document.",
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )
    res_mixed = check_hidden_content(doc_mixed)
    mixed_findings = [f for f in res_mixed.findings if f.details.get("category") == "mixed_script_word"]
    assert len(mixed_findings) == 1
    f = mixed_findings[0]
    assert f.severity == "high"
    assert "syst\u0435m" in f.message
    # Check start and end match original text
    assert doc_mixed.raw_text[f.start:f.end] == "syst\u0435m"


def test_docx_hidden_run_detection(demo_dir):
    docx_path = demo_dir / "hidden_content_demo.docx"
    assert docx_path.exists(), "DOCX demo file must exist"

    res = check_hidden_content(docx_path)
    assert not res.passed
    assert any("white" in f.message.lower() for f in res.findings)
    assert any("under 2 pt" in f.message.lower() or "1.0 pt" in f.message.lower() for f in res.findings)


def test_pdf_color_and_size_detection():
    # Generate a small test PDF with fpdf2
    import fpdf
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = Path(tmp.name)

    try:
        pdf = fpdf.FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, text="Visible normal text.")
        pdf.ln(10)

        # White text
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, text="White hidden text in PDF.")
        pdf.ln(10)

        # Tiny font size (< 2 pt)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", size=1)
        pdf.cell(0, 5, text="Tiny microscopic PDF payload.")

        pdf.output(str(pdf_path))

        res = check_hidden_content(pdf_path)
        assert len(res.findings) >= 1
        assert any(f.severity == "high" for f in res.findings)
        assert any(
            "white" in f.message.lower() or "under 2 pt" in f.message.lower()
            for f in res.findings
        )
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


# ==============================================================================
# 2. Citation and Quote Tests
# ==============================================================================

def test_citation_style_regexes():
    text = (
        "Previous algorithms [1] and extended schemes [2, 4] or [5-8] "
        "were reviewed in (Smith, 2020) and (Johnson & Lee, 2021). "
        "Furthermore, Smith et al. (2022) and Brown (2019) demonstrated efficiency."
    )
    citations = detect_in_text_citations(text)
    styles = {c.style for c in citations}
    assert "numeric" in styles
    assert "author_year" in styles

    # Check numeric expansion
    num_cits = [c for c in citations if c.style == "numeric"]
    assert any("1" in c.keys for c in num_cits)
    assert any("2" in c.keys and "4" in c.keys for c in num_cits)
    assert any("5" in c.keys and "6" in c.keys and "7" in c.keys and "8" in c.keys for c in num_cits)

    # Check author-year extraction
    ay_cits = [c for c in citations if c.style == "author_year"]
    all_ay_keys = [k for c in ay_cits for k in c.keys]
    assert any("smith_2020" in k for k in all_ay_keys)
    assert any("johnson_2021" in k for k in all_ay_keys)
    assert any("smith_2022" in k for k in all_ay_keys)
    assert any("brown_2019" in k for k in all_ay_keys)


def test_missing_uncited_and_duplicate_references():
    content = (
        "Neural networks are powerful [1]. "
        "However, some claims reference missing works [5].\n\n"
        "References\n\n"
        "[1] Adams, J. (2019). Foundations of Machine Learning. Oxford.\n"
        "[2] Baker, R. (2020). Deep Networks in Practice. MIT.\n"
        "[2] Baker, R. (2020). Deep Networks in Practice. MIT.\n"
    )
    doc = Document(
        raw_text=content,
        clean_text=content,
        sentences=[],
        paragraphs=[],
        removed_sections=[
            RemovedSection(
                type="references",
                content="[1] Adams, J. (2019). Foundations of Machine Learning. Oxford.\n[2] Baker, R. (2020). Deep Networks in Practice. MIT.\n[2] Baker, R. (2020). Deep Networks in Practice. MIT.\n",
                start=content.find("References"),
                end=len(content),
            )
        ],
        file_type="txt",
    )
    res = check_citations_and_quotes(doc)

    issues = [f.details.get("issue") for f in res.findings]
    assert "missing_reference" in issues
    assert "uncited_reference" in issues
    assert "duplicate_reference" in issues


def test_inconsistent_citation_style():
    content = (
        "Classic works [1] laid the foundation. Later, Smith (2020) advanced the theory.\n\n"
        "References\n\n"
        "[1] Doe, J. (2015). Early AI.\n"
        "Smith, A. (2020). Modern AI.\n"
    )
    doc = Document(
        raw_text=content,
        clean_text=content,
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )
    res = check_citations_and_quotes(doc)
    issues = [f.details.get("issue") for f in res.findings]
    assert "style_inconsistency" in issues


def test_uncited_claim_heuristics():
    # 1. Uncited claim with percentage and trigger phrase -> should be flagged as warning (never high)
    s1_text = "Studies show that adoption rates increased by 45% in urban districts."
    s1 = Sentence(text=s1_text, start=0, end=len(s1_text))
    doc_uncited = Document(
        raw_text=s1_text,
        clean_text=s1_text,
        sentences=[s1],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )
    res_uncited = check_citations_and_quotes(doc_uncited)
    claim_findings = [f for f in res_uncited.findings if f.details.get("issue") == "uncited_claim"]
    assert len(claim_findings) == 1
    assert claim_findings[0].severity == "warning"
    assert claim_findings[0].severity != "high"

    # 2. Normal cited sentence -> FALSE POSITIVE CHECK: MUST NOT BE FLAGGED
    s2_text = "Studies show that adoption rates increased by 45% in urban districts [1]."
    s2 = Sentence(text=s2_text, start=0, end=len(s2_text))
    doc_cited = Document(
        raw_text=s2_text,
        clean_text=s2_text,
        sentences=[s2],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )
    res_cited = check_citations_and_quotes(doc_cited)
    claim_findings_cited = [f for f in res_cited.findings if f.details.get("issue") == "uncited_claim"]
    assert len(claim_findings_cited) == 0, "Sentence with citation must not be flagged as uncited claim"


def test_long_quote_detection():
    long_quote_text = (
        '"When natural language models generate extensive text without constraints, '
        'they demonstrate an extraordinarily uniform distribution of n-gram transition '
        'probabilities that diverges significantly from authentic human authors writing '
        'under spontaneous conversational conditions across multiple independent domains."'
    )
    assert len(long_quote_text.split()) > 25

    doc = Document(
        raw_text=f"The committee noted: {long_quote_text}",
        clean_text="The committee noted: ",
        sentences=[],
        paragraphs=[],
        removed_sections=[
            RemovedSection(
                type="quote",
                content=long_quote_text,
                start=21,
                end=21 + len(long_quote_text),
            )
        ],
        file_type="txt",
    )
    res = check_citations_and_quotes(doc)
    quote_findings = [f for f in res.findings if f.details.get("issue") == "long_uncited_quote"]
    assert len(quote_findings) == 1
    assert quote_findings[0].severity == "warning"


def test_uncited_verbatim_copy_detection():
    doc_text = "In modern distributed computing, edge devices aggregate telemetry data locally to minimize bandwidth consumption."
    doc = Document(
        raw_text=doc_text,
        clean_text=doc_text,
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )
    # Mock Stage 3 SimilarityResult with exact span not in quotation marks and without citation
    span = MatchSpan(
        type="exact",
        score=1.0,
        source_id="src_99",
        source_title="Distributed Systems Guide",
        source_url="",
        doc_start=3,
        doc_end=len(doc_text) - 5,
        matched_text=doc_text[3:len(doc_text) - 5],
    )
    source_match = SourceMatch(
        source_id="src_99",
        source_title="Distributed Systems Guide",
        source_url="",
        origin="web",
        matched_chars=len(span.matched_text),
        percentage=90.0,
        match_count=1,
        spans=[span],
    )
    sim_res = SimilarityResult(
        document_path="doc1.txt",
        total_clean_chars=len(doc_text),
        matched_chars=len(span.matched_text),
        similarity_percentage=90.0,
        spans=[span],
        sources=[source_match],
    )

    res = check_citations_and_quotes(doc, similarity_result=sim_res)
    uncited_copy = [f for f in res.findings if f.details.get("issue") == "uncited_verbatim_copy"]
    assert len(uncited_copy) == 1
    assert "possible uncited copied text; add quotation marks and a citation" in uncited_copy[0].message


# ==============================================================================
# 3. Self-Plagiarism Tests
# ==============================================================================

def test_self_check_overlap(demo_dir):
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        prev_file = demo_dir / "self_plag_previous.txt"
        new_file = demo_dir / "self_plag_new.txt"
        assert prev_file.exists() and new_file.exists()

        # 1. Index previous file
        indexed = add_previous(prev_file, db_path=db_path)
        assert len(indexed) == 1

        # 2. Check new file
        res = check_self_plagiarism(new_file, db_path=db_path)
        assert len(res.findings) >= 1
        assert res.metadata["total_overlap_pct"] > 20.0
        assert not res.passed
    finally:
        if db_path.exists():
            db_path.unlink()


# ==============================================================================
# 4. Code Similarity Tests
# ==============================================================================

def test_code_similarity_python(demo_dir):
    code_dir = demo_dir / "code_demo"
    orig = code_dir / "original.py"
    renamed = code_dir / "renamed_copy.py"
    unrelated = code_dir / "unrelated.py"
    template = code_dir / "boilerplate_template.py"

    assert orig.exists() and renamed.exists() and unrelated.exists()

    # 1. Renamed-variable copy scores high (> 75%)
    sim_renamed = compare_code_files(orig, renamed)
    assert sim_renamed.similarity_pct > 75.0
    assert len(sim_renamed.matching_ranges) >= 1

    # 2. Unrelated file scores very low (< 10%)
    sim_unrelated = compare_code_files(orig, unrelated)
    assert sim_unrelated.similarity_pct < 10.0

    # 3. Directory comparison
    dir_results = compare_code_directory(code_dir, template_file=template)
    assert len(dir_results) >= 1
    top_pair = dir_results[0]
    assert {top_pair.file_a, top_pair.file_b} == {"original.py", "renamed_copy.py"}
    assert top_pair.similarity_pct > 70.0


def test_code_tokenizers():
    # Python normalization
    py_code = "def calculate_sum(val_a, val_b):\n    # Add numbers\n    return val_a + val_b\n"
    py_tokens = tokenize_python(py_code)
    py_token_str = [t.token for t in py_tokens]
    assert "KW:def" in py_token_str
    assert "KW:return" in py_token_str
    assert "ID" in py_token_str
    assert "+" in py_token_str

    # Generic C / Java normalization
    c_code = "int compute_total(int count, int factor) {\n    // comment\n    return count * factor;\n}\n"
    c_tokens = tokenize_generic(c_code, language="c")
    c_token_str = [t.token for t in c_tokens]
    assert "KW:int" in c_token_str
    assert "KW:return" in c_token_str
    assert "ID" in c_token_str
    assert "*" in c_token_str


# ==============================================================================
# 5. Serialization and Runner Tests
# ==============================================================================

def test_runner_and_json_serialization(demo_dir):
    sample = demo_dir / "homoglyphs_zero_width_demo.txt"
    report = run_all_checks(sample, previous=False)

    assert isinstance(report, IntegrityReport)
    assert not report.passed
    assert len(report.results) >= 2

    # Verify JSON serialization
    report_dict = report.to_dict()
    assert "results" in report_dict
    assert "summary" in report_dict

    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["document_path"] == str(sample.resolve())
    assert parsed["passed"] is False


def test_cli_smoke(demo_dir):
    from veritext.checks.__main__ import main

    # 1. CLI on homoglyphs sample
    sample = str(demo_dir / "homoglyphs_zero_width_demo.txt")
    code = main([sample, "--no-previous"])
    # Return code 2 indicates flagged issues, 0 indicates clean
    assert code in (0, 2)

    # 2. CLI code comparison
    code_dir = str(demo_dir / "code_demo")
    code_res = main(["code", code_dir])
    assert code_res in (0, 2)
