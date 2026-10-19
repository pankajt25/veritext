"""Stylometric feature extraction requiring no language models."""

from collections import Counter
import math
import re
from typing import Dict, List, Sequence

# Common formulaic discourse transitions and AI hallmark phrases
DISCOURSE_MARKERS = [
    "however",
    "moreover",
    "furthermore",
    "in conclusion",
    "it is important to note",
    "it should be noted",
    "it is worth noting",
    "on the other hand",
    "consequently",
    "specifically",
    "additionally",
    "nevertheless",
    "in summary",
    "to summarize",
    "as a result",
    "in contrast",
    "delves into",
    "plays a crucial role",
    "plays a vital role",
    "a testament to",
    "in essence",
    "ultimately",
    "further details",
]

PUNCTUATION_CHARS = set(".,!?:;-\"\'()[]{}—…")


def extract_word_tokens(text: str) -> List[str]:
    """Extracts lowercase alphabetic word tokens."""
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def extract_sentences(text: str) -> List[str]:
    """Extracts sentences using standard boundary regex."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def extract_paragraphs(text: str) -> List[str]:
    """Extracts non-empty paragraphs split by double newline."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def compute_stylometric_features(
    text: str,
    sentences: Sequence[str] = None,
    paragraphs: Sequence[str] = None,
) -> Dict[str, float]:
    """
    Computes a comprehensive suite of lightweight stylometric features:
    - sentence_length_mean, sentence_length_variance
    - type_token_ratio, hapax_ratio, average_word_length
    - punctuation_diversity, repeated_ngram_rate
    - discourse_marker_density, paragraph_length_variance
    """
    words = extract_word_tokens(text)
    total_words = len(words)

    if total_words == 0:
        return {
            "total_words": 0.0,
            "sentence_length_mean": 0.0,
            "sentence_length_variance": 0.0,
            "type_token_ratio": 0.0,
            "hapax_ratio": 0.0,
            "average_word_length": 0.0,
            "punctuation_diversity": 0.0,
            "repeated_ngram_rate": 0.0,
            "discourse_marker_density": 0.0,
            "paragraph_length_variance": 0.0,
        }

    s_list = list(sentences) if sentences is not None else extract_sentences(text)
    p_list = list(paragraphs) if paragraphs is not None else extract_paragraphs(text)

    # 1. Sentence Length Statistics (Words per sentence)
    sent_lengths = [len(extract_word_tokens(s)) for s in s_list if s.strip()]
    if sent_lengths:
        sent_mean = sum(sent_lengths) / len(sent_lengths)
        sent_var = (
            sum((l - sent_mean) ** 2 for l in sent_lengths) / len(sent_lengths)
            if len(sent_lengths) > 1
            else 0.0
        )
    else:
        sent_mean = float(total_words)
        sent_var = 0.0

    # 2. Vocabulary Richness (TTR and Hapax Legomena)
    word_counts = Counter(words)
    unique_words = len(word_counts)
    ttr = unique_words / total_words

    hapax_count = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_ratio = hapax_count / total_words

    # 3. Average Word Length
    avg_word_len = sum(len(w) for w in words) / total_words

    # 4. Punctuation Diversity (Shannon entropy of punctuation characters)
    punct_in_text = [ch for ch in text if ch in PUNCTUATION_CHARS]
    if punct_in_text:
        p_counts = Counter(punct_in_text)
        n_punct = len(punct_in_text)
        punct_entropy = -sum(
            (c / n_punct) * math.log2(c / n_punct) for c in p_counts.values()
        )
    else:
        punct_entropy = 0.0

    # 5. Repeated n-gram rate (repeated word bigrams)
    if total_words >= 2:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(total_words - 1)]
        bg_counts = Counter(bigrams)
        repeated_bigrams = sum(c - 1 for c in bg_counts.values() if c > 1)
        repeated_ngram_rate = repeated_bigrams / len(bigrams)
    else:
        repeated_ngram_rate = 0.0

    # 6. Discourse & Formulaic Transition Marker Density (occurrences per 100 words)
    text_lower = text.lower()
    marker_hits = 0
    for marker in DISCOURSE_MARKERS:
        # Match as word phrase boundary
        pattern = r"\b" + re.escape(marker) + r"\b"
        marker_hits += len(re.findall(pattern, text_lower))
    discourse_marker_density = (marker_hits / total_words) * 100.0

    # 7. Paragraph Length Uniformity (Variance of words per paragraph)
    para_lengths = [len(extract_word_tokens(p)) for p in p_list if p.strip()]
    if len(para_lengths) > 1:
        p_mean = sum(para_lengths) / len(para_lengths)
        para_var = sum((l - p_mean) ** 2 for l in para_lengths) / len(para_lengths)
    else:
        para_var = 0.0

    return {
        "total_words": float(total_words),
        "sentence_length_mean": sent_mean,
        "sentence_length_variance": sent_var,
        "type_token_ratio": ttr,
        "hapax_ratio": hapax_ratio,
        "average_word_length": avg_word_len,
        "punctuation_diversity": punct_entropy,
        "repeated_ngram_rate": repeated_ngram_rate,
        "discourse_marker_density": discourse_marker_density,
        "paragraph_length_variance": para_var,
    }
