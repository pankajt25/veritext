"""Tests for grammar and style feedback."""

from unittest.mock import patch
import pytest

from veritext.style import (
    calculate_style_metrics,
    check_filler_phrases,
    check_languagetool_support,
    check_passive_voice,
    check_repeated_words,
    check_sentence_length,
    check_weak_transitions,
    count_syllables_in_word,
    lint_style,
)


def test_passive_voice_detection():
    text = "The experiment was conducted by researchers, and the findings were written carefully."
    findings = check_passive_voice(text)
    assert len(findings) >= 2
    # Verify offsets
    for f in findings:
        assert f.severity == "info"
        assert f.start >= 0
        assert f.end > f.start
        assert text[f.start:f.end].lower().startswith("was") or text[f.start:f.end].lower().startswith("were")


def test_repeated_words():
    text = "This paper shows that the the results are significant in in the dataset."
    findings = check_repeated_words(text)
    assert len(findings) == 2
    words = [f.details["word"].lower() for f in findings]
    assert "the" in words
    assert "in" in words
    for f in findings:
        assert f.severity == "info"


def test_sentence_length():
    short_sentence = "This is a short statement."
    long_sentence = " ".join(["word"] * 40) + "."
    text = f"{short_sentence} {long_sentence}"
    findings = check_sentence_length(text, max_words=35)
    assert len(findings) == 1
    assert findings[0].details["word_count"] == 40
    assert findings[0].severity == "info"


def test_filler_phrases():
    text = "In order to test the system, due to the fact that time was limited, we ran tests."
    findings = check_filler_phrases(text)
    assert len(findings) == 2
    phrases = [f.details["phrase"] for f in findings]
    assert "in order to" in phrases
    assert "due to the fact that" in phrases


def test_weak_transitions():
    text = "Basically, we started testing. Obviously, the output was consistent."
    findings = check_weak_transitions(text)
    assert len(findings) == 2
    for f in findings:
        assert f.severity == "info"


def test_readability_metrics():
    simple_text = "The cat sat on the mat. The dog saw the cat. It was a nice day."
    metrics = calculate_style_metrics(simple_text)
    assert metrics.flesch_reading_ease > 75.0
    assert metrics.gunning_fog < 8.0
    assert metrics.total_sentences == 3
    assert metrics.total_words > 0

    complex_text = (
        "The epistemological ramifications of post-structuralist ontology necessitate "
        "a rigorous multi-faceted deconstruction of institutionalized pedagogical paradigms."
    )
    complex_metrics = calculate_style_metrics(complex_text)
    assert complex_metrics.flesch_reading_ease < metrics.flesch_reading_ease
    assert complex_metrics.gunning_fog > metrics.gunning_fog


def test_syllable_counter():
    assert count_syllables_in_word("cat") == 1
    assert count_syllables_in_word("simple") == 2
    assert count_syllables_in_word("complex") == 2
    assert count_syllables_in_word("university") == 5


def test_languagetool_unavailable_reported_cleanly():
    # Test when Java is not found
    with patch("shutil.which", return_value=None):
        status = check_languagetool_support()
        assert status["available"] is False
        assert "Java" in status["reason"] or "neither" in status["reason"]

    # Test full linter with disabled LanguageTool
    res = lint_style("This is a test sentence.", enable_languagetool=False)
    assert res.engine_status["builtin_rules"] is True
    assert "languagetool" in res.engine_status
    assert res.metrics is not None
