"""Readability calculations: Flesch Reading Ease and Gunning Fog index."""

import re
from typing import List, Tuple

from veritext.style.models import StyleMetric


def count_syllables_in_word(word: str) -> int:
    """Heuristic count of syllables in an English word."""
    w = word.lower().strip()
    if not w:
        return 0
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    if len(w) <= 3:
        return 1

    # Remove silent trailing e / es / ed
    if w.endswith("e") and not w.endswith("le") and not w.endswith("ee"):
        w = w[:-1]
    elif w.endswith("ed") and not w.endswith("ted") and not w.endswith("ded"):
        w = w[:-2]

    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    in_vowel_group = False
    for ch in w:
        if ch in vowels:
            if not in_vowel_group:
                count += 1
                in_vowel_group = True
        else:
            in_vowel_group = False

    return max(1, count)


def is_complex_word(word: str) -> bool:
    """Checks if a word has 3 or more syllables for Gunning Fog calculation."""
    clean = re.sub(r"[^a-zA-Z]", "", word)
    if len(clean) < 6:
        return False
    # Exclude proper nouns (capitalized) if used in mid-sentence, but check syllables
    syllables = count_syllables_in_word(clean)
    return syllables >= 3


def split_sentences_and_words(text: str) -> Tuple[List[str], List[str]]:
    """Splits text into sentences and tokenized words."""
    if not text:
        return [], []
    # Split sentences roughly by . ! ? followed by space or newline
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    words = re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", text)
    return sentences, words


def get_flesch_label(score: float) -> Tuple[str, str]:
    """Returns (ease_label, grade_level) for a Flesch Reading Ease score."""
    if score >= 90.0:
        return "Very Easy", "5th grade"
    elif score >= 80.0:
        return "Easy", "6th grade"
    elif score >= 70.0:
        return "Fairly Easy", "7th grade"
    elif score >= 60.0:
        return "Standard", "8th-9th grade"
    elif score >= 50.0:
        return "Fairly Difficult", "10th-12th grade"
    elif score >= 30.0:
        return "Difficult", "College"
    else:
        return "Very Difficult", "College Graduate"


def calculate_style_metrics(text: str) -> StyleMetric:
    """Calculates Flesch Reading Ease and Gunning Fog for input text."""
    sentences, words = split_sentences_and_words(text)
    total_words = len(words)
    total_sentences = max(1, len(sentences))

    if total_words == 0:
        return StyleMetric(
            flesch_reading_ease=100.0,
            gunning_fog=0.0,
            total_words=0,
            total_sentences=0,
            total_syllables=0,
            complex_words=0,
            reading_ease_label="Standard",
            reading_level="N/A",
        )

    total_syllables = sum(count_syllables_in_word(w) for w in words)
    complex_words = sum(1 for w in words if is_complex_word(w))

    # Flesch Reading Ease: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    asl = total_words / total_sentences
    asw = total_syllables / total_words
    flesch = 206.835 - (1.015 * asl) - (84.6 * asw)
    flesch = max(0.0, min(100.0, flesch))

    # Gunning Fog: 0.4 * ((words / sentences) + 100 * (complex_words / words))
    pct_complex = (complex_words / total_words) * 100.0
    gunning_fog = 0.4 * (asl + pct_complex)
    gunning_fog = max(0.0, min(30.0, gunning_fog))

    ease_label, reading_level = get_flesch_label(flesch)

    return StyleMetric(
        flesch_reading_ease=flesch,
        gunning_fog=gunning_fog,
        total_words=total_words,
        total_sentences=len(sentences),
        total_syllables=total_syllables,
        complex_words=complex_words,
        reading_ease_label=ease_label,
        reading_level=reading_level,
    )
