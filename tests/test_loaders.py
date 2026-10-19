"""Unit tests for document loaders."""

from pathlib import Path
import pytest
from veritext.ingest.loaders import (
    get_loader,
    load_docx,
    load_md,
    load_pdf,
    load_tex,
    load_txt,
)


@pytest.fixture
def samples_dir():
    return Path(__file__).resolve().parent.parent / "data" / "samples"


def test_load_txt(samples_dir):
    txt_path = samples_dir / "sample_academic.txt"
    text = load_txt(txt_path)
    assert "Adversarial Robustness in Automated Plagiarism Detection" in text
    assert "Abstract" in text
    assert len(text) > 200


def test_load_md(samples_dir):
    md_path = samples_dir / "sample_academic.md"
    text = load_md(md_path)
    assert "Semantic Integrity Verification in Academic Writing" in text
    assert "Abstract" in text
    assert len(text) > 200


def test_load_tex(samples_dir):
    tex_path = samples_dir / "sample_academic.tex"
    text = load_tex(tex_path)
    # The loader runs process_latex, so math/code/citations are sanitized
    assert "Neural Representations and Text Similarity Analysis" in text or "Document comparison relies" in text
    assert "\\begin{equation}" not in text
    assert "\\begin{verbatim}" not in text


def test_load_docx(samples_dir):
    docx_path = samples_dir / "sample_academic.docx"
    if not docx_path.exists():
        pytest.skip("sample_academic.docx not generated yet")
    text = load_docx(docx_path)
    assert "Evaluating Integrity in Generative Text Systems" in text
    assert "Abstract" in text


def test_load_pdf(samples_dir):
    pdf_path = samples_dir / "sample_academic.pdf"
    if not pdf_path.exists():
        pytest.skip("sample_academic.pdf not generated yet")
    text = load_pdf(pdf_path)
    assert "Heuristic Approaches to Document Verification" in text or "Abstract" in text


def test_get_loader():
    assert get_loader("test.txt") == load_txt
    assert get_loader("test.md") == load_md
    assert get_loader("test.tex") == load_tex
    assert get_loader("test.docx") == load_docx
    assert get_loader("test.pdf") == load_pdf

    with pytest.raises(ValueError, match="Unsupported file format"):
        get_loader("test.xyz")
