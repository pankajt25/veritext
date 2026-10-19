"""Unit tests for the 5 free search clients using mock fixtures."""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from veritext.sources.clients.arxiv import parse_arxiv_xml, search as search_arxiv
from veritext.sources.clients.crossref import search as search_crossref
from veritext.sources.clients.openalex import (
    reconstruct_abstract,
    search as search_openalex,
)
from veritext.sources.clients.semantic_scholar import (
    search as search_semantic_scholar,
)
from veritext.sources.clients.wikipedia import search as search_wikipedia
from veritext.sources.http import CachedResponse, PoliteHttpClient


@pytest.fixture
def fixtures_dir():
    return Path(__file__).resolve().parent / "fixtures"


def test_crossref_client_parsing(fixtures_dir):
    fixture_path = fixtures_dir / "crossref.json"
    content = fixture_path.read_text(encoding="utf-8")

    mock_http = MagicMock(spec=PoliteHttpClient)
    mock_http.get.return_value = CachedResponse(
        status_code=200, text=content, headers={}, url="https://api.crossref.org"
    )

    results = search_crossref("neural text", limit=3, client=mock_http)
    assert len(results) == 1
    cand = results[0]
    assert cand.origin == "crossref"
    assert "Adversarial Robustness" in cand.title
    assert "Alex Johnson" in cand.authors
    assert "Maria Garcia" in cand.authors
    assert cand.year == 2020
    assert "defensive token masking" in cand.text
    assert "<jats:p>" not in cand.text  # JATS tags stripped


def test_openalex_abstract_reconstruction():
    inverted = {
        "Deep": [0],
        "learning": [1],
        "models": [2],
        "require": [3],
        "rigorous": [4],
        "verification.": [5],
    }
    reconstructed = reconstruct_abstract(inverted)
    assert reconstructed == "Deep learning models require rigorous verification."


def test_openalex_client_parsing(fixtures_dir):
    fixture_path = fixtures_dir / "openalex.json"
    content = fixture_path.read_text(encoding="utf-8")

    mock_http = MagicMock(spec=PoliteHttpClient)
    mock_http.get.return_value = CachedResponse(
        status_code=200, text=content, headers={}, url="https://api.openalex.org"
    )

    results = search_openalex("authorship verification", limit=3, client=mock_http)
    assert len(results) == 1
    cand = results[0]
    assert cand.origin == "openalex"
    assert "Neural Paraphrase Identification" in cand.title
    assert "Elena Rostova" in cand.authors
    assert cand.year == 2021
    assert "stylistic variance in neural paraphrasing" in cand.text


def test_arxiv_xml_parsing(fixtures_dir):
    fixture_path = fixtures_dir / "arxiv.xml"
    xml_content = fixture_path.read_text(encoding="utf-8")

    candidates = parse_arxiv_xml(xml_content)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.origin == "arxiv"
    assert "Learning Transferable Visual Models" in cand.title
    assert "Alec Radford" in cand.authors
    assert cand.year == 2021
    assert "predetermined object categories" in cand.text
    assert "arxiv_2103.00020v1" in cand.id


def test_semantic_scholar_client_parsing(fixtures_dir):
    fixture_path = fixtures_dir / "semantic_scholar.json"
    content = fixture_path.read_text(encoding="utf-8")

    mock_http = MagicMock(spec=PoliteHttpClient)
    mock_http.get.return_value = CachedResponse(
        status_code=200,
        text=content,
        headers={},
        url="https://api.semanticscholar.org",
    )

    results = search_semantic_scholar("survey plagiarism", limit=3, client=mock_http)
    assert len(results) == 1
    cand = results[0]
    assert cand.origin == "semantic_scholar"
    assert "A Survey on Automated Text Plagiarism" in cand.title
    assert "Sarah Connor" in cand.authors
    assert cand.year == 2022
    assert "contextualized embeddings" in cand.text


def test_semantic_scholar_429_graceful_handling():
    mock_http = MagicMock(spec=PoliteHttpClient)
    mock_http.get.return_value = CachedResponse(
        status_code=429,
        text='{"message": "Too Many Requests"}',
        headers={},
        url="https://api.semanticscholar.org",
    )

    # Must return empty list gracefully and not raise an exception
    results = search_semantic_scholar("rate limited query", limit=3, client=mock_http)
    assert results == []


def test_wikipedia_client_parsing(fixtures_dir):
    fixture_path = fixtures_dir / "wikipedia.json"
    content = fixture_path.read_text(encoding="utf-8")

    mock_http = MagicMock(spec=PoliteHttpClient)
    mock_http.get.return_value = CachedResponse(
        status_code=200, text=content, headers={}, url="https://en.wikipedia.org"
    )

    results = search_wikipedia("Plagiarism", limit=3, client=mock_http)
    assert len(results) == 1
    cand = results[0]
    assert cand.origin == "wikipedia"
    assert cand.title == "Plagiarism"
    assert "Wikipedia Contributors" in cand.authors
    assert "representation of another author's language" in cand.text


def test_client_handles_network_failure():
    mock_http = MagicMock(spec=PoliteHttpClient)
    mock_http.get.side_effect = ConnectionError("Network unreachable")

    # None of these should crash
    assert search_crossref("query", client=mock_http) == []
    assert search_openalex("query", client=mock_http) == []
    assert search_arxiv("query", client=mock_http) == []
    assert search_semantic_scholar("query", client=mock_http) == []
    assert search_wikipedia("query", client=mock_http) == []
