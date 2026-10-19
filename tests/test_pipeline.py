"""End-to-end integration tests for the load_document pipeline."""

from pathlib import Path
import pytest
from veritext.ingest.pipeline import load_document


@pytest.fixture
def samples_dir():
    return Path(__file__).resolve().parent.parent / "data" / "samples"


def _verify_document(doc, expected_ext):
    assert doc.file_type == expected_ext
    assert len(doc.raw_text) > 0
    assert len(doc.clean_text) > 0
    assert len(doc.sentences) > 0
    assert len(doc.paragraphs) > 0

    # Critical check: offset accuracy in clean_text
    for s in doc.sentences:
        assert doc.clean_text[s.start:s.end] == s.text, f"Sentence offset mismatch: '{s.text}'"

    for p in doc.paragraphs:
        assert doc.clean_text[p.start:p.end] == p.text, f"Paragraph offset mismatch: '{p.text}'"

    # Verify dictionary serialization works
    doc_dict = doc.to_dict()
    assert doc_dict["file_type"] == expected_ext
    assert len(doc_dict["sentences"]) == len(doc.sentences)


def test_load_document_txt(samples_dir):
    path = samples_dir / "sample_academic.txt"
    doc = load_document(path)
    _verify_document(doc, "txt")
    # Verify references were removed
    types = [r.type for r in doc.removed_sections]
    assert "references" in types


def test_load_document_md(samples_dir):
    path = samples_dir / "sample_academic.md"
    doc = load_document(path)
    _verify_document(doc, "md")
    types = [r.type for r in doc.removed_sections]
    assert "references" in types
    assert "quote" in types


def test_load_document_tex(samples_dir):
    path = samples_dir / "sample_academic.tex"
    doc = load_document(path)
    _verify_document(doc, "tex")
    types = [r.type for r in doc.removed_sections]
    assert "latex_math" in types or "code_block" in types or "citation" in types


def test_load_document_docx(samples_dir):
    path = samples_dir / "sample_academic.docx"
    if not path.exists():
        pytest.skip("sample_academic.docx not generated yet")
    doc = load_document(path)
    _verify_document(doc, "docx")


def test_load_document_pdf(samples_dir):
    path = samples_dir / "sample_academic.pdf"
    if not path.exists():
        pytest.skip("sample_academic.pdf not generated yet")
    doc = load_document(path)
    _verify_document(doc, "pdf")


def test_load_document_not_found():
    with pytest.raises(FileNotFoundError):
        load_document("non_existent_document_12345.txt")


def test_load_document_unsupported_format(tmp_path):
    bad_file = tmp_path / "test.unsupported"
    bad_file.write_text("Some text")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(bad_file)
