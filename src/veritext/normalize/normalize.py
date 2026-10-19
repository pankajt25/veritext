"""Text canonicalisation, homoglyph folding, zero-width stripping, and offset mapping."""

import re
import unicodedata
from typing import List, Optional, Tuple

from veritext.normalize.models import Anomaly, CanonicalText

# Comprehensive confusables mapping: Cyrillic and Greek lookalikes of Latin letters
HOMOGLYPH_MAP = {
    # Cyrillic lowercase
    "\u0430": "a",  # Cyrillic small letter a
    "\u0441": "c",  # Cyrillic small letter es
    "\u0435": "e",  # Cyrillic small letter ie
    "\u0456": "i",  # Cyrillic small letter byelorussian-ukrainian i
    "\u0458": "j",  # Cyrillic small letter je
    "\u043e": "o",  # Cyrillic small letter o
    "\u0440": "p",  # Cyrillic small letter er
    "\u0455": "s",  # Cyrillic small letter dze
    "\u0443": "y",  # Cyrillic small letter u
    "\u0445": "x",  # Cyrillic small letter ha
    "\u0432": "b",  # Cyrillic small letter ve
    "\u043c": "m",  # Cyrillic small letter em
    "\u043d": "h",  # Cyrillic small letter en
    "\u0442": "t",  # Cyrillic small letter te
    "\u045d": "i",  # Cyrillic small letter i with grave
    # Cyrillic uppercase
    "\u0410": "A",  # Cyrillic capital letter a
    "\u0412": "B",  # Cyrillic capital letter ve
    "\u0421": "C",  # Cyrillic capital letter es
    "\u0415": "E",  # Cyrillic capital letter ie
    "\u041d": "H",  # Cyrillic capital letter en
    "\u0406": "I",  # Cyrillic capital letter byelorussian-ukrainian i
    "\u0408": "J",  # Cyrillic capital letter je
    "\u041a": "K",  # Cyrillic capital letter ka
    "\u041c": "M",  # Cyrillic capital letter em
    "\u041e": "O",  # Cyrillic capital letter o
    "\u0420": "P",  # Cyrillic capital letter er
    "\u0422": "T",  # Cyrillic capital letter te
    "\u0425": "X",  # Cyrillic capital letter ha
    "\u0423": "Y",  # Cyrillic capital letter u
    # Greek lowercase
    "\u03b1": "a",  # Greek small letter alpha
    "\u03b2": "b",  # Greek small letter beta
    "\u03b3": "y",  # Greek small letter gamma
    "\u03b5": "e",  # Greek small letter epsilon
    "\u03b7": "n",  # Greek small letter eta
    "\u03b9": "i",  # Greek small letter iota
    "\u03ba": "k",  # Greek small letter kappa
    "\u03bd": "v",  # Greek small letter nu
    "\u03bf": "o",  # Greek small letter omicron
    "\u03c1": "p",  # Greek small letter rho
    "\u03c4": "t",  # Greek small letter tau
    "\u03c5": "u",  # Greek small letter upsilon
    "\u03c7": "x",  # Greek small letter chi
    # Greek uppercase
    "\u0391": "A",  # Greek capital letter alpha
    "\u0392": "B",  # Greek capital letter beta
    "\u0395": "E",  # Greek capital letter epsilon
    "\u0396": "Z",  # Greek capital letter zeta
    "\u0397": "H",  # Greek capital letter eta
    "\u0399": "I",  # Greek capital letter iota
    "\u039a": "K",  # Greek capital letter kappa
    "\u039c": "M",  # Greek capital letter mu
    "\u039d": "N",  # Greek capital letter nu
    "\u039f": "O",  # Greek capital letter omicron
    "\u03a1": "P",  # Greek capital letter rho
    "\u03a4": "T",  # Greek capital letter tau
    "\u03a5": "Y",  # Greek capital letter upsilon
    "\u03a7": "X",  # Greek capital letter chi
}

# Zero-width, soft hyphens, and bidirectional formatting characters
ZERO_WIDTH_CHARS = {
    "\u200b": "Zero-width space",
    "\u200c": "Zero-width non-joiner",
    "\u200d": "Zero-width joiner",
    "\u200e": "Left-to-right mark",
    "\u200f": "Right-to-left mark",
    "\u2060": "Word joiner",
    "\ufeff": "Zero-width no-break space / BOM",
    "\u00ad": "Soft hyphen",
}

# Bidirectional embedding and isolate control characters
BIDI_CONTROL_CHARS = {
    "\u202a": "Left-to-right embedding",
    "\u202b": "Right-to-left embedding",
    "\u202c": "Pop directional formatting",
    "\u202d": "Left-to-right override",
    "\u202e": "Right-to-left override",
    "\u2066": "Left-to-right isolate",
    "\u2067": "Right-to-left isolate",
    "\u2068": "First strong isolate",
    "\u2069": "Pop directional isolate",
}

# Standard punctuation canonicalisation map
PUNCTUATION_CANONICAL = {
    "\u201c": '"',  # Left double quotation mark
    "\u201d": '"',  # Right double quotation mark
    "\u201e": '"',  # Double low-9 quotation mark
    "\u201f": '"',  # Double high-reversed-9 quotation mark
    "\u2018": "'",  # Left single quotation mark
    "\u2019": "'",  # Right single quotation mark
    "\u201a": "'",  # Single low-9 quotation mark
    "\u201b": "'",  # Single high-reversed-9 quotation mark
    "\u2014": "-",  # Em dash
    "\u2013": "-",  # En dash
    "\u2026": "...",  # Horizontal ellipsis
    "\u00ab": '"',  # Left-pointing double angle quotation mark
    "\u00bb": '"',  # Right-pointing double angle quotation mark
}

_lemmatizer = None


def _get_lemmatizer():
    """Lazily initializes and returns the NLTK WordNet lemmatizer."""
    global _lemmatizer
    if _lemmatizer is None:
        try:
            import nltk
            try:
                nltk.data.find("corpora/wordnet")
            except (LookupError, OSError):
                nltk.download("wordnet", quiet=True)
            try:
                nltk.data.find("corpora/omw-1.4")
            except (LookupError, OSError):
                nltk.download("omw-1.4", quiet=True)
            from nltk.stem import WordNetLemmatizer
            _lemmatizer = WordNetLemmatizer()
        except Exception:
            _lemmatizer = False
    return _lemmatizer if _lemmatizer is not False else None


def normalize_text(text: str) -> CanonicalText:
    """
    Performs full text canonicalisation:
    - Enforces valid UTF-8
    - Applies Unicode NFKC normalization
    - Detects and strips zero-width, bidi, and control characters (recording anomalies)
    - Folds homoglyphs (Cyrillic/Greek lookalikes) to Latin (recording anomalies)
    - Canonicalises punctuation and collapses whitespace
    - Lowercases and lemmatizes words
    - Constructs an exact token offset_map referencing character spans in original text.

    Returns:
        CanonicalText object.
    """
    if not text:
        return CanonicalText(
            normalized_text="",
            tokens=[],
            offset_map=[],
            anomalies=[],
            original_text="",
        )

    # 1. Force valid UTF-8 string
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    else:
        text = text.encode("utf-8", errors="replace").decode("utf-8")

    anomalies: List[Anomaly] = []

    # Step A: Scan character by character through original text,
    # mapping each emitted character to its original character position.
    emitted_chars: List[str] = []
    emitted_indices: List[int] = []

    for orig_idx, ch in enumerate(text):
        # 1. Check for zero-width characters
        if ch in ZERO_WIDTH_CHARS:
            desc = ZERO_WIDTH_CHARS[ch]
            anomalies.append(
                Anomaly(
                    type="zero_width",
                    character=ch,
                    position=orig_idx,
                    description=desc,
                )
            )
            continue

        # 2. Check for bidi/control characters
        if ch in BIDI_CONTROL_CHARS:
            desc = BIDI_CONTROL_CHARS[ch]
            anomalies.append(
                Anomaly(
                    type="control_character",
                    character=ch,
                    position=orig_idx,
                    description=desc,
                )
            )
            continue

        # Check for other non-printable control characters (except newline, tab)
        if ord(ch) < 32 and ch not in ("\n", "\t", "\r"):
            anomalies.append(
                Anomaly(
                    type="control_character",
                    character=ch,
                    position=orig_idx,
                    description=f"Control character {ord(ch)}",
                )
            )
            continue

        # 3. Check for homoglyphs
        if ch in HOMOGLYPH_MAP:
            repl = HOMOGLYPH_MAP[ch]
            anomalies.append(
                Anomaly(
                    type="homoglyph",
                    character=ch,
                    position=orig_idx,
                    replacement=repl,
                    description=f"Homoglyph U+{ord(ch):04X} -> '{repl}'",
                )
            )
            emitted_chars.append(repl)
            emitted_indices.append(orig_idx)
            continue

        # 4. Canonicalise punctuation
        if ch in PUNCTUATION_CANONICAL:
            repl = PUNCTUATION_CANONICAL[ch]
            for r_ch in repl:
                emitted_chars.append(r_ch)
                emitted_indices.append(orig_idx)
            continue

        # Normal character
        emitted_chars.append(ch)
        emitted_indices.append(orig_idx)

    # Step B: Apply Unicode NFKC normalization while preserving index alignment
    intermediate_str = "".join(emitted_chars)
    nfkc_chars: List[str] = []
    nfkc_indices: List[int] = []

    for i, ch in enumerate(intermediate_str):
        orig_pos = emitted_indices[i]
        norm = unicodedata.normalize("NFKC", ch)
        for n_ch in norm:
            nfkc_chars.append(n_ch)
            nfkc_indices.append(orig_pos)

    cleaned_str = "".join(nfkc_chars)

    # Step C: Extract word tokens and compute exact original text spans
    lemmatizer = _get_lemmatizer()
    tokens: List[str] = []
    offset_map: List[Tuple[int, int]] = []

    # Find contiguous word characters (letters and digits)
    word_pattern = re.compile(r"\b[a-zA-Z0-9]+\b")
    for match in word_pattern.finditer(cleaned_str):
        c_start, c_end = match.start(), match.end()
        raw_word = match.group(0).lower()

        # Map character bounds back to original text indices
        orig_start = nfkc_indices[c_start]
        orig_end = nfkc_indices[c_end - 1] + 1

        # Lemmatize
        if lemmatizer:
            try:
                lemma = lemmatizer.lemmatize(raw_word)
            except Exception:
                lemma = raw_word
        else:
            lemma = raw_word

        tokens.append(lemma)
        offset_map.append((orig_start, orig_end))

    normalized_text = " ".join(tokens)

    return CanonicalText(
        normalized_text=normalized_text,
        tokens=tokens,
        offset_map=offset_map,
        anomalies=anomalies,
        original_text=text,
    )
