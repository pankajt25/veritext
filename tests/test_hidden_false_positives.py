"""Fast offline unit tests for Part B: Fewer False Positives in Hidden-Content Checks.

Tests background shading resolution, spacer filtering, finding grouping,
homoglyph/zero-width severity preservation, and Stage 5 demo compatibility.
"""

from pathlib import Path
import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Pt, RGBColor
import pytest

from veritext.checks.hidden import check_hidden_content


def test_white_text_on_dark_table_header_not_flagged(tmp_path: Path):
    """White text on dark table headers (#002060) is visible and must NOT be flagged."""
    doc = docx.Document()
    table = doc.add_table(rows=2, cols=2)
    hdr_cell = table.cell(0, 0)
    # Dark navy shading (#002060)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="002060"/>')
    hdr_cell._tc.get_or_add_tcPr().append(shd)

    p = hdr_cell.paragraphs[0]
    r = p.add_run("Executive Summary Header")
    r.font.color.rgb = RGBColor(255, 255, 255)

    docx_path = tmp_path / "white_on_dark.docx"
    doc.save(str(docx_path))

    res = check_hidden_content(docx_path)
    # Should not produce any high findings for white font on dark background
    high_findings = [f for f in res.findings if f.severity == "high"]
    assert len(high_findings) == 0


def test_white_text_on_white_background_is_flagged(tmp_path: Path):
    """White text on unset or white background is hidden/adversarial and MUST be flagged."""
    doc = docx.Document()
    p = doc.add_paragraph()
    r = p.add_run("Malicious hidden evasion payload text injected here.")
    r.font.color.rgb = RGBColor(255, 255, 255)

    docx_path = tmp_path / "white_on_white.docx"
    doc.save(str(docx_path))

    res = check_hidden_content(docx_path)
    high_findings = [f for f in res.findings if f.severity == "high"]
    assert len(high_findings) >= 1
    assert any("white" in f.message.lower() for f in high_findings)


def test_tiny_font_spacer_run_not_flagged(tmp_path: Path):
    """Tiny-font runs containing only whitespace / spacers must NOT be flagged."""
    doc = docx.Document()
    p = doc.add_paragraph()
    r1 = p.add_run("Normal visible body text.")
    r_spacer = p.add_run("   \t  ")
    r_spacer.font.size = Pt(1.0)

    docx_path = tmp_path / "tiny_spacer.docx"
    doc.save(str(docx_path))

    res = check_hidden_content(docx_path)
    tiny_findings = [f for f in res.findings if "font size under 2 pt" in f.message]
    assert len(tiny_findings) == 0


def test_real_tiny_font_word_run_is_flagged(tmp_path: Path):
    """Tiny-font runs containing visible words exceeding character threshold MUST be flagged."""
    doc = docx.Document()
    p = doc.add_paragraph()
    r = p.add_run("Microscopic hidden sentence payload inserted for evasion test.")
    r.font.size = Pt(1.0)

    docx_path = tmp_path / "tiny_payload.docx"
    doc.save(str(docx_path))

    res = check_hidden_content(docx_path)
    high_findings = [f for f in res.findings if f.severity == "high"]
    assert len(high_findings) >= 1
    assert any("font size under 2 pt" in f.message for f in high_findings)


def test_repeated_header_cells_grouped_into_one_finding(tmp_path: Path):
    """30 identical styled header cells must produce ONE grouped finding rather than 30 separate findings."""
    doc = docx.Document()
    # Create table with 30 header cells with white text on unset/light background (suspicious)
    table = doc.add_table(rows=6, cols=5)
    for row in table.rows:
        for cell in row.cells:
            p = cell.paragraphs[0]
            r = p.add_run("Header")
            r.font.color.rgb = RGBColor(255, 255, 255)

    docx_path = tmp_path / "repeated_headers.docx"
    doc.save(str(docx_path))

    res = check_hidden_content(docx_path)
    # The 30 cells should be grouped into a single finding
    docx_findings = [f for f in res.findings if f.details.get("source") == "docx_run"]
    assert len(docx_findings) == 1
    assert docx_findings[0].details.get("count") == 30
    assert "across 30 locations" in docx_findings[0].message


def test_homoglyphs_and_zero_width_remain_high_severity():
    """Cyrillic homoglyphs in Latin words and genuine zero-width evasion remain high severity."""
    # Cyrillic 'а' (\u0430) and 'е' (\u0435) in Latin words
    text_homoglyphs = "Th\u0435 r\u0435sult w\u0430s un\u0435xp\u0435ct\u0435d in this context."
    res_h = check_hidden_content(text_homoglyphs)
    high_h = [f for f in res_h.findings if f.severity == "high"]
    assert len(high_h) >= 1
    assert any("Cyrillic" in f.message for f in high_h)

    # Genuine zero-width evasion characters (\u200b) >= 3 characters
    text_zw = "Sec\u200bur\u200bity prot\u200bocol evasion"
    res_zw = check_hidden_content(text_zw)
    high_zw = [f for f in res_zw.findings if f.severity == "high"]
    assert len(high_zw) >= 1
    assert any("zero-width" in f.message.lower() for f in high_zw)


def test_soft_hyphens_and_nbsp_classified_as_info_never_high():
    """Soft hyphens (\u00ad) and non-breaking spaces must produce info findings, never high or warning."""
    text_hyphens = "The op\u00ader\u00ada\u00adtion\u00adal in\u00adfra\u00adstruc\u00adture was tested."
    res = check_hidden_content(text_hyphens)
    high_f = [f for f in res.findings if f.severity == "high"]
    warn_f = [f for f in res.findings if f.severity == "warning"]
    assert len(high_f) == 0
    assert len(warn_f) == 0
    assert res.passed is True


def test_stage5_demo_expectations_hold():
    """The original demo hidden_content_demo.docx must still produce its exact 2 high findings."""
    demo_path = Path("data/samples/checks_demo/hidden_content_demo.docx")
    assert demo_path.exists()

    res = check_hidden_content(demo_path)
    high_findings = [f for f in res.findings if f.severity == "high"]
    assert len(high_findings) == 2
    assert res.passed is False
