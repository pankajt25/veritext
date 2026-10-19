"""Unit tests for sentence and paragraph splitting with character offset accuracy."""

import pytest
from veritext.ingest.splitter import split_paragraphs, split_sentences


def test_paragraph_splitting_offsets():
    text = (
        "Paragraph one begins here. It contains multiple statements.\n\n"
        "Paragraph two is distinctly separated by blank lines.\n\n"
        "Paragraph three concludes the sample section."
    )
    paragraphs = split_paragraphs(text)
    assert len(paragraphs) == 3

    for p in paragraphs:
        # Crucial: verify exact offset slicing matches the paragraph text
        assert text[p.start:p.end] == p.text
        assert p.text == str(p)

    assert "Paragraph one begins here" in paragraphs[0].text
    assert "Paragraph two is distinctly separated" in paragraphs[1].text
    assert "Paragraph three concludes" in paragraphs[2].text


def test_sentence_splitting_offsets():
    text = (
        "Artificial intelligence is transforming academic institutions rapidly. "
        "Researchers must adapt their evaluation metrics accordingly. "
        "Does modern pedagogy account for synthetic writing? Indeed, it must."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 4

    for s in sentences:
        # Crucial: verify exact offset slicing matches the sentence text
        assert text[s.start:s.end] == s.text

    assert sentences[0].text == "Artificial intelligence is transforming academic institutions rapidly."
    assert sentences[1].text == "Researchers must adapt their evaluation metrics accordingly."
    assert sentences[2].text == "Does modern pedagogy account for synthetic writing?"
    assert sentences[3].text == "Indeed, it must."


def test_sentence_tuple_unpacking_and_indexing():
    text = "First sentence here. Second sentence follows."
    sentences = split_sentences(text)
    assert len(sentences) == 2

    # Unpacking test
    t, start, end = sentences[0]
    assert t == sentences[0].text
    assert start == sentences[0].start
    assert end == sentences[0].end
    assert text[start:end] == t

    # Indexing test
    assert sentences[1][0] == sentences[1].text
    assert sentences[1][1] == sentences[1].start
    assert sentences[1][2] == sentences[1].end


def test_splitter_empty_and_whitespace():
    assert split_paragraphs("") == []
    assert split_paragraphs("   \n\n   ") == []
    assert split_sentences("") == []
    assert split_sentences("   \n\t  ") == []


def test_multiline_paragraph_offsets():
    text = (
        "This is a paragraph that spans\n"
        "across several physical lines within\n"
        "a single conceptual block.\n\n"
        "And here is another distinct paragraph."
    )
    paragraphs = split_paragraphs(text)
    assert len(paragraphs) == 2
    for p in paragraphs:
        assert text[p.start:p.end] == p.text
