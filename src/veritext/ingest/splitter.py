"""Sentence and paragraph splitting with exact character offset tracking."""

import re
from typing import List, Tuple
from veritext.ingest.models import Paragraph, Sentence


def _ensure_nltk_punkt():
    """Ensures NLTK punkt tokenizer models are available."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except (LookupError, OSError):
            nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


def split_paragraphs(text: str) -> List[Paragraph]:
    """
    Splits text into paragraphs based on double newlines while tracking exact character offsets.
    Guarantees text[start:end] == paragraph.text.
    """
    paragraphs: List[Paragraph] = []
    if not text:
        return paragraphs

    # Regex finding non-empty blocks between double newlines
    # Matches any non-whitespace contiguous block across lines
    pattern = re.compile(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*")

    for match in pattern.finditer(text):
        raw_start, raw_end = match.start(), match.end()
        raw_str = text[raw_start:raw_end]
        stripped = raw_str.strip()
        if not stripped:
            continue

        leading = len(raw_str) - len(raw_str.lstrip())
        trailing = len(raw_str) - len(raw_str.rstrip())

        start = raw_start + leading
        end = raw_end - trailing
        p_text = text[start:end]

        paragraphs.append(Paragraph(text=p_text, start=start, end=end))

    return paragraphs


def _split_sentences_fallback(text: str) -> List[Tuple[int, int]]:
    """
    Fallback regex-based sentence boundary detector if NLTK punkt is unavailable.
    Returns (start, end) tuples.
    """
    spans: List[Tuple[int, int]] = []
    # Match sentences ending with .!? followed by space or newline and a capital letter/quote/end
    pattern = re.compile(r".+?(?:[.!?]+(?=[\s\n]+[A-Z0-9\"'“]|$)|$)", re.DOTALL)
    for match in pattern.finditer(text):
        chunk = match.group(0)
        stripped = chunk.strip()
        if not stripped:
            continue
        leading = len(chunk) - len(chunk.lstrip())
        trailing = len(chunk) - len(chunk.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        if start < end:
            spans.append((start, end))
    return spans


def split_sentences(text: str) -> List[Sentence]:
    """
    Splits text into sentences using NLTK's PunktSentenceTokenizer while tracking exact character offsets.
    Guarantees text[sentence.start:sentence.end] == sentence.text.
    """
    sentences: List[Sentence] = []
    if not text or not text.strip():
        return sentences

    _ensure_nltk_punkt()

    raw_spans: List[Tuple[int, int]] = []
    try:
        import nltk.tokenize
        tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        raw_spans = list(tokenizer.span_tokenize(text))
    except Exception:
        raw_spans = []

    if not raw_spans:
        raw_spans = _split_sentences_fallback(text)

    for r_start, r_end in raw_spans:
        span_str = text[r_start:r_end]
        stripped = span_str.strip()
        if not stripped:
            continue

        leading = len(span_str) - len(span_str.lstrip())
        trailing = len(span_str) - len(span_str.rstrip())

        start = r_start + leading
        end = r_end - trailing
        s_text = text[start:end]

        sentences.append(Sentence(text=s_text, start=start, end=end))

    return sentences
