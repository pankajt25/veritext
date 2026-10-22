"""Sentence and paragraph splitting with exact character offset tracking."""

import re
from typing import List, Tuple
from veritext.ingest.models import Paragraph, Sentence


# Section heading terms frequently appearing in academic and structured documents
SECTION_KEYWORDS = {
    "abstract",
    "introduction",
    "methodology",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "results and discussion",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "works cited",
    "background",
    "overview",
    "summary",
    "literature review",
    "appendix",
    "appendices",
}


def is_heading_text(text: str) -> bool:
    """
    Determines whether a snippet of text constitutes a heading or title rather than a body sentence.
    Headings:
    - Markdown headers (e.g. # Title, ## Subtitle)
    - Short phrases (<= 15 words and <= 120 chars) without terminal punctuation (. ! ?)
      and containing no internal newlines.
    """
    t = text.strip()
    if not t:
        return False
    if re.match(r"^#{1,6}\s+", t):
        return True
    if "\n" in t:
        return False
    # If it ends with terminal punctuation (. ! ?), it is a sentence
    if re.search(r'[.!?]["\'”’]?$', t):
        return False
    words = t.split()
    if len(words) <= 15 and len(t) <= 120:
        return True
    return False


def _is_heading_line(line: str, next_line: str) -> bool:
    """
    Checks if a line within a multi-line paragraph block is a heading line.
    """
    l = line.strip()
    if not l:
        return False
    if re.match(r"^#{1,6}\s+", l):
        return True
    nl = next_line.strip()
    if not nl or (not nl[0].isupper() and nl[0] not in "\"\'“"):
        return False
    if re.search(r"[,;:-]$", l):
        return False
    if re.search(r'[.!?]["\'”’]?$', l):
        return False

    low = l.lower().strip(":")
    if low in SECTION_KEYWORDS:
        return True

    words = [w for w in l.split() if w.lower() not in {"and", "or", "of", "in", "on", "the", "a", "an", "for", "with", "to", "by", "&"}]
    if len(l.split()) <= 8 and words and all(w[0].isupper() for w in words if w):
        return True

    return False


_PUNKT_INITIALIZED = False
_PUNKT_TOKENIZER = None


def _get_punkt_tokenizer():
    """Lazily initializes and caches NLTK PunktSentenceTokenizer once."""
    global _PUNKT_INITIALIZED, _PUNKT_TOKENIZER
    if not _PUNKT_INITIALIZED:
        _PUNKT_INITIALIZED = True
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
            import nltk.tokenize
            _PUNKT_TOKENIZER = nltk.tokenize.PunktSentenceTokenizer()
        except Exception:
            _PUNKT_TOKENIZER = None
    return _PUNKT_TOKENIZER


def _ensure_nltk_punkt():
    """Ensures NLTK punkt tokenizer models are available."""
    _get_punkt_tokenizer()


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
    Headings are identified and flagged with is_heading=True.
    """
    sentences: List[Sentence] = []
    if not text or not text.strip():
        return sentences

    tokenizer = _get_punkt_tokenizer()

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        paragraphs = [Paragraph(text=text, start=0, end=len(text))]

    for p in paragraphs:
        p_text = p.text
        p_offset = p.start

        # 1. Check if entire paragraph is a standalone heading/title
        if is_heading_text(p_text):
            p_lead = len(p_text) - len(p_text.lstrip())
            p_trail = len(p_text) - len(p_text.rstrip())
            s_start = p_offset + p_lead
            s_end = p_offset + len(p_text) - p_trail
            s_text = text[s_start:s_end]
            sentences.append(Sentence(text=s_text, start=s_start, end=s_end, is_heading=True))
            continue

        # 2. Extract leading heading line(s) if present within paragraph block
        while "\n" in p_text:
            lines = p_text.split("\n", 1)
            if _is_heading_line(lines[0], lines[1]):
                h_str = lines[0].strip()
                if h_str:
                    h_lead = len(lines[0]) - len(lines[0].lstrip())
                    h_trail = len(lines[0]) - len(lines[0].rstrip())
                    h_start = p_offset + h_lead
                    h_end = p_offset + len(lines[0]) - h_trail
                    sentences.append(Sentence(text=text[h_start:h_end], start=h_start, end=h_end, is_heading=True))
                p_offset = p_offset + len(lines[0]) + 1
                p_text = lines[1]
            else:
                break

        # 3. Tokenize standard sentences within the paragraph body
        if not p_text.strip():
            continue

        p_lead = len(p_text) - len(p_text.lstrip())
        p_strip = p_text.strip()
        current_offset = p_offset + p_lead

        if tokenizer:
            try:
                spans = list(tokenizer.span_tokenize(p_strip))
            except Exception:
                spans = _split_sentences_fallback(p_strip)
        else:
            spans = _split_sentences_fallback(p_strip)

        if not spans:
            spans = _split_sentences_fallback(p_strip)

        for s_start, s_end in spans:
            chunk = p_strip[s_start:s_end]
            s_lead = len(chunk) - len(chunk.lstrip())
            s_trail = len(chunk) - len(chunk.rstrip())
            final_start = current_offset + s_start + s_lead
            final_end = current_offset + s_end - s_trail
            final_text = text[final_start:final_end]

            if not final_text:
                continue

            is_head = is_heading_text(final_text)
            sentences.append(Sentence(text=final_text, start=final_start, end=final_end, is_heading=is_head))

    return sentences
