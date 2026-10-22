"""Tests for cross-language matching, language heuristic, and opt-in model management."""

from unittest.mock import MagicMock, patch
import pytest

from veritext.similarity.crosslang import (
    CROSSLANG_MODEL_NAME,
    DOWNLOAD_INSTRUCTION,
    compute_crosslang_embeddings,
    detect_language_heuristic,
    get_crosslang_status,
    is_crosslang_model_available,
)


def test_language_heuristic_english():
    text = (
        "In this paper we present a novel framework for distributed fault-tolerant computing. "
        "The algorithms maintain consistency guarantees across asynchronous networks."
    )
    res = detect_language_heuristic(text)
    assert res["language"] == "en"
    assert res["is_likely_non_english"] is False


def test_language_heuristic_spanish():
    text = (
        "En este artículo presentamos un marco computacional para el análisis del lenguaje natural. "
        "Los métodos tradicionales de aprendizaje automático proporcionan una precisión adecuada para las tareas."
    )
    res = detect_language_heuristic(text)
    assert res["language"] == "es"
    assert res["is_likely_non_english"] is True


def test_language_heuristic_french():
    text = (
        "Dans ce travail de recherche, nous analysons les structures syntaxiques de la langue française. "
        "Les résultats démontrent une corrélation significative entre les variables étudiées par l'équipe."
    )
    res = detect_language_heuristic(text)
    assert res["language"] == "fr"
    assert res["is_likely_non_english"] is True


def test_language_heuristic_german():
    text = (
        "In dieser Arbeit untersuchen wir die Grundlagen der quantenmechanischen Berechnungen. "
        "Die experimentellen Ergebnisse bestätigen die theoretischen Vorhersagen für das Gesamtsystem."
    )
    res = detect_language_heuristic(text)
    assert res["language"] == "de"
    assert res["is_likely_non_english"] is True


def test_language_heuristic_cyrillic():
    text = "В данной статье рассматриваются основные принципы распределенных вычислений."
    res = detect_language_heuristic(text)
    assert res["script"] == "cyrillic"
    assert res["is_likely_non_english"] is True


def test_crosslang_disabled_notice(monkeypatch, tmp_path):
    # Ensure model dir points to empty directory
    monkeypatch.setattr("veritext.similarity.crosslang.CROSSLANG_MODEL_DIR", tmp_path / "nonexistent_model")

    assert is_crosslang_model_available() is False
    status = get_crosslang_status()
    assert status["available"] is False
    assert DOWNLOAD_INSTRUCTION in status["message"]
    assert "python -m veritext.similarity --download-crosslang-model" in status["message"]


def test_crosslang_batch_size_guard(monkeypatch):
    """Ensures embedding computation limits batch size to <= 16 for 8 GB RAM safety."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    monkeypatch.setattr("veritext.similarity.crosslang.get_crosslang_model", lambda: mock_model)

    sentences = ["First sentence in French.", "Second sentence in Spanish."]
    compute_crosslang_embeddings(sentences, batch_size=64)  # request oversize batch

    args, kwargs = mock_model.encode.call_args
    assert kwargs.get("batch_size") <= 16
    assert kwargs.get("batch_size") == 16
