"""Fast document profiling before analysis engine execution.

Profiles word, character, sentence, and paragraph counts, estimates pages and tokens,
detects language via stopword heuristic, calculates reference section words,
and classifies document size class (tiny, short, medium, long, very long).
Runs purely offline with zero neural models and zero network calls.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Union

from veritext.ingest.models import Document
from veritext.ingest.pipeline import load_document

# Common stopwords for fast heuristic language identification
STOPWORDS: Dict[str, Set[str]] = {
    "en": {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
    },
    "es": {
        "de", "la", "que", "el", "en", "y", "a", "los", "se", "del",
        "las", "un", "por", "con", "no", "una", "su", "para", "es", "al",
        "lo", "como", "mas", "o", "pero", "sus", "le", "ha", "me", "si",
    },
    "fr": {
        "de", "la", "le", "et", "les", "des", "en", "un", "du", "une",
        "que", "est", "pour", "qui", "dans", "a", "par", "sur", "pas", "plus",
        "au", "avec", "ce", "son", "ne", "il", "sa", "se", "ont", "ses",
    },
    "de": {
        "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
        "des", "auf", "fur", "ist", "im", "dem", "nicht", "ein", "eine", "als",
        "auch", "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach",
    },
}

REFERENCES_HEADER_RE = re.compile(
    r"(?m)^(?:\s*(?:#{1,6}\s+|(?:\d+\.?\s+)?)(?:References|Bibliography|Works Cited|Literature Cited)\s*:?\s*$)",
    re.IGNORECASE,
)


@dataclass
class DocumentProfile:
    """Statistical summary of document metrics and structural sizing."""

    word_count: int
    character_count: int
    sentence_count: int
    paragraph_count: int
    page_estimate: int
    estimated_tokens: int
    reference_words: int
    language_guess: str
    size_class: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentProfile":
        return cls(
            word_count=int(data.get("word_count", 0)),
            character_count=int(data.get("character_count", 0)),
            sentence_count=int(data.get("sentence_count", 0)),
            paragraph_count=int(data.get("paragraph_count", 0)),
            page_estimate=int(data.get("page_estimate", 1)),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            reference_words=int(data.get("reference_words", 0)),
            language_guess=str(data.get("language_guess", "en")),
            size_class=str(data.get("size_class", "short")),
        )


def guess_language(text: str) -> str:
    """Fast stopword-ratio heuristic to determine natural language (default 'en')."""
    words = [re.sub(r"[^\w]", "", w).lower() for w in text.split()[:2000]]
    words = [w for w in words if w]
    if not words:
        return "en"

    total = len(words)
    best_lang = "en"
    best_ratio = 0.0

    for lang, sw_set in STOPWORDS.items():
        matched = sum(1 for w in words if w in sw_set)
        ratio = matched / total
        if ratio > best_ratio:
            best_ratio = ratio
            best_lang = lang

    # Default to English if no clear match
    return best_lang if best_ratio >= 0.03 else "en"


def classify_size(word_count: int) -> str:
    """Classifies document size category:
    - tiny: under 150 words
    - short: under 1,500 words
    - medium: under 8,000 words
    - long: under 30,000 words
    - very long: 30,000 words and above
    """
    if word_count < 150:
        return "tiny"
    if word_count < 1500:
        return "short"
    if word_count < 8000:
        return "medium"
    if word_count < 30000:
        return "long"
    return "very long"


def profile_document(document: Union[Document, str, Path]) -> DocumentProfile:
    """Profiles document structure immediately after ingestion, before any engine runs.

    Args:
        document: Document instance, file path, or raw text string.

    Returns:
        DocumentProfile containing size metrics, page estimates, language, and size class.
    """
    if isinstance(document, (str, Path)):
        doc_path = Path(document)
        if doc_path.exists() and doc_path.is_file():
            doc = load_document(doc_path)
        else:
            from veritext.ingest.models import Paragraph
            raw = str(document)
            clean = raw.strip()
            paras = [Paragraph(text=p, start=0, end=len(p)) for p in clean.split("\n\n") if p.strip()]
            sents = []
            doc = Document(
                raw_text=raw,
                clean_text=clean,
                sentences=sents,
                paragraphs=paras,
                removed_sections=[],
                file_type=".txt",
            )
    else:
        doc = document

    text = doc.clean_text or doc.raw_text or ""
    words = text.split()
    word_count = len(words)
    character_count = len(text)

    # Paragraph and sentence counts
    paragraph_count = len(doc.paragraphs) if doc.paragraphs else max(1, len([p for p in text.split("\n\n") if p.strip()])) if text.strip() else 0
    sentence_count = len(doc.sentences) if doc.sentences else max(1, len(re.split(r"[.!?]+\s+", text.strip()))) if text.strip() else 0

    # Page estimate: 500 words per page
    page_estimate = max(1, math.ceil(word_count / 500.0)) if word_count > 0 else 0

    # Estimated tokens: approx 1.33 tokens per word
    estimated_tokens = int(math.ceil(word_count * 1.33))

    # Reference section words
    reference_words = 0
    if doc.removed_sections:
        for section in doc.removed_sections:
            if getattr(section, "type", "") in ("references", "bibliography"):
                reference_words += len(section.content.split())
    elif doc.raw_text:
        match = REFERENCES_HEADER_RE.search(doc.raw_text)
        if match:
            ref_slice = doc.raw_text[match.start():]
            reference_words = len(ref_slice.split())

    language_guess = guess_language(text)
    size_class = classify_size(word_count)

    return DocumentProfile(
        word_count=word_count,
        character_count=character_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        page_estimate=page_estimate,
        estimated_tokens=estimated_tokens,
        reference_words=reference_words,
        language_guess=language_guess,
        size_class=size_class,
    )
