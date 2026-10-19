"""Unit tests for LaTeX parsing and sanitization."""

import pytest
from veritext.ingest.latex import (
    process_latex,
    remove_latex_citations,
    remove_latex_code_blocks,
    remove_latex_math,
)


def test_remove_latex_math_inline():
    text = "The speed of light is $c = 3 \\times 10^8$ m/s in vacuum."
    cleaned, removed = remove_latex_math(text)
    assert "$c = 3 \\times 10^8$" in [r.content for r in removed]
    assert "$c = 3" not in cleaned
    assert "The speed of light is" in cleaned
    assert "m/s in vacuum" in cleaned


def test_remove_latex_math_environments():
    text = (
        "Consider the following loss function:\n"
        "\\begin{equation}\n"
        "L = -\\sum y \\log(\\hat{y})\n"
        "\\end{equation}\n"
        "This is used in classification."
    )
    cleaned, removed = remove_latex_math(text)
    assert len(removed) >= 1
    assert "equation" in removed[0].content
    assert "\\begin{equation}" not in cleaned
    assert "Consider the following loss function:" in cleaned
    assert "This is used in classification." in cleaned


def test_remove_latex_display_math():
    text = "Formula: \\[ E = mc^2 \\] is famous."
    cleaned, removed = remove_latex_math(text)
    assert len(removed) == 1
    assert "\\[ E = mc^2 \\]" in removed[0].content
    assert "Formula:" in cleaned
    assert "is famous." in cleaned


def test_remove_latex_code_blocks():
    text = (
        "Here is the algorithm:\n"
        "\\begin{verbatim}\n"
        "def hello():\n"
        "    print('world')\n"
        "\\end{verbatim}\n"
        "End of algorithm."
    )
    cleaned, removed = remove_latex_code_blocks(text)
    assert len(removed) >= 1
    assert "verbatim" in removed[0].content
    assert "def hello():" not in cleaned
    assert "Here is the algorithm:" in cleaned
    assert "End of algorithm." in cleaned


def test_remove_latex_citations():
    text = (
        "Previous studies \\cite{smith2020} and others \\citep[e.g.][p. 12]{jones2021} "
        "have shown significant improvements."
    )
    cleaned, removed = remove_latex_citations(text)
    assert len(removed) == 2
    assert "\\cite{smith2020}" in [r.content for r in removed]
    assert "smith2020" not in cleaned
    assert "jones2021" not in cleaned
    assert "Previous studies" in cleaned
    assert "have shown significant improvements." in cleaned


def test_process_latex_end_to_end():
    latex_sample = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{Introduction}\n"
        "Neural networks \\cite{vaswani2017} optimize functions where $f(x) = y$.\n"
        "\\begin{verbatim}\n"
        "x = torch.tensor([1, 2, 3])\n"
        "\\end{verbatim}\n"
        "Training concludes successfully.\n"
        "\\end{document}"
    )
    plain, removed = process_latex(latex_sample)
    assert "torch.tensor" not in plain
    assert "$f(x) = y$" not in plain
    assert "vaswani2017" not in plain
    assert "Neural networks" in plain
    assert "Training concludes successfully" in plain
    assert len(removed) >= 3
