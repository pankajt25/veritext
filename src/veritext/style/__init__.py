"""Grammar and style feedback module."""

from veritext.style.linter import check_languagetool_support, lint_style
from veritext.style.metrics import (
    calculate_style_metrics,
    count_syllables_in_word,
    is_complex_word,
)
from veritext.style.models import StyleMetric, StyleResult
from veritext.style.rules import (
    check_filler_phrases,
    check_passive_voice,
    check_repeated_words,
    check_sentence_length,
    check_weak_transitions,
)

__all__ = [
    "StyleMetric",
    "StyleResult",
    "lint_style",
    "check_languagetool_support",
    "calculate_style_metrics",
    "count_syllables_in_word",
    "is_complex_word",
    "check_passive_voice",
    "check_repeated_words",
    "check_sentence_length",
    "check_filler_phrases",
    "check_weak_transitions",
]
