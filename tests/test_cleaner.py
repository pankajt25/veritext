"""Unit tests for document cleaning (references and quote removal)."""

import pytest
from veritext.ingest.cleaner import clean_text, remove_quotes, remove_references


def test_remove_references_standard():
    text = (
        "Main discussion points.\n\n"
        "References\n"
        "1. Turing, A. M. (1950). Computing Machinery and Intelligence.\n"
        "2. Shannon, C. E. (1948). A Mathematical Theory of Communication.\n"
    )
    cleaned, removed = remove_references(text)
    assert "Main discussion points." in cleaned
    assert "Turing, A. M." not in cleaned
    assert len(removed) == 1
    assert removed[0].type == "references"
    assert "Turing" in removed[0].content


def test_remove_references_with_appendix():
    text = (
        "Main text content.\n\n"
        "## References\n"
        "[1] First Paper\n"
        "[2] Second Paper\n\n"
        "## Appendix A\n"
        "Additional experimental details.\n"
    )
    cleaned, removed = remove_references(text)
    assert "Main text content." in cleaned
    assert "First Paper" not in cleaned
    assert "Appendix A" in cleaned
    assert "Additional experimental details." in cleaned
    assert len(removed) == 1
    assert "First Paper" in removed[0].content


def test_remove_latex_bibliography():
    text = (
        "Article body before bib.\n\n"
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{ref1} Author Name, Title.\n"
        "\\end{thebibliography}\n"
    )
    cleaned, removed = remove_references(text)
    assert "Article body before bib." in cleaned
    assert "\\bibitem{ref1}" not in cleaned
    assert len(removed) == 1
    assert removed[0].type == "references"


def test_remove_quotes_double_and_curly():
    text = (
        'The philosopher stated, "I think, therefore I am," during his inquiry. '
        'Similarly, another wrote, “Language is the house of Being,” in his essay.'
    )
    cleaned, removed = remove_quotes(text)
    assert '"I think, therefore I am,"' not in cleaned
    assert "Language is the house of Being" not in cleaned
    assert len(removed) == 2
    assert all(r.type == "quote" for r in removed)
    assert "The philosopher stated," in cleaned
    assert "during his inquiry." in cleaned


def test_remove_quotes_markdown_blockquote():
    text = (
        "Introduction to the concept.\n\n"
        "> This is a direct blockquote spanning\n"
        "> multiple lines in markdown format.\n\n"
        "Concluding thoughts follow."
    )
    cleaned, removed = remove_quotes(text)
    assert "direct blockquote spanning" not in cleaned
    assert "Introduction to the concept." in cleaned
    assert "Concluding thoughts follow." in cleaned
    assert len(removed) == 1
    assert removed[0].type == "quote"


def test_clean_text_end_to_end():
    text = (
        "Scholars assert that integrity is vital.\n\n"
        'As one expert argued, "Honesty fosters genuine scholarship and progress."\n\n'
        "Our analysis confirms this trend.\n\n"
        "References\n"
        "1. Smith, J. (2020). Academic Ethics."
    )
    clean, removed = clean_text(text)
    assert "Scholars assert that integrity is vital." in clean
    assert "Our analysis confirms this trend." in clean
    assert "Honesty fosters genuine scholarship" not in clean
    assert "Smith, J. (2020)" not in clean
    types = [r.type for r in removed]
    assert "references" in types
    assert "quote" in types
