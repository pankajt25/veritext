"""Unit tests for unified candidate retrieval and ranking."""

from unittest.mock import MagicMock, patch
import pytest

from veritext.ingest.models import Document, Paragraph, Sentence
from veritext.sources.models import SourceCandidate
from veritext.sources.retrieval import find_candidates, rank_candidates_by_overlap
from veritext.sources.store import ReferenceStore


@pytest.fixture
def sample_document():
    text = (
        "Adversarial perturbations against neural sequence models can degrade "
        "plagiarism detection accuracy. By introducing subtle token substitutions, "
        "adversaries evade detection systems without altering semantic intent."
    )
    return Document(
        raw_text=text,
        clean_text=text,
        sentences=[Sentence(text=text, start=0, end=len(text))],
        paragraphs=[Paragraph(text=text, start=0, end=len(text))],
        removed_sections=[],
        file_type="txt",
    )


def test_rank_candidates_by_overlap():
    doc_text = "Adversarial machine learning and neural networks in text classification."

    c1 = SourceCandidate(
        id="c1",
        title="Adversarial Text Attacks",
        authors=["Author A"],
        url="https://example.com/1",
        origin="test",
        year=2021,
        text="Adversarial machine learning techniques targeting text classification.",
    )
    c2 = SourceCandidate(
        id="c2",
        title="Cellular Biology and Genetics",
        authors=["Author B"],
        url="https://example.com/2",
        origin="test",
        year=2019,
        text="Photosynthesis and mitochondrial DNA sequencing in plants.",
    )

    ranked = rank_candidates_by_overlap(doc_text, [c2, c1])
    assert ranked[0].id == "c1"
    assert ranked[0].score > ranked[1].score


def test_find_candidates_offline_mode(tmp_path, sample_document):
    db_file = tmp_path / "offline_store.sqlite"
    store = ReferenceStore(db_file)

    # Add a relevant local reference
    local_cand = SourceCandidate(
        id="loc_1",
        title="Adversarial Robustness in Natural Language",
        authors=["Local Scholar"],
        url="file:///local/paper.txt",
        origin="local_library",
        year=2023,
        text="Neural sequence models face significant vulnerability from adversarial token perturbations.",
    )
    store.add_source(local_cand)

    # In offline mode, use_online=False
    results = find_candidates(
        sample_document,
        use_online=False,
        use_local=True,
        max_sources=5,
        store=store,
    )

    assert len(results) >= 1
    assert results[0].origin == "local_library"
    assert "Adversarial Robustness" in results[0].title


def test_find_candidates_with_mocked_clients(tmp_path, sample_document):
    db_file = tmp_path / "mocked_store.sqlite"
    store = ReferenceStore(db_file)

    mock_crossref_cand = SourceCandidate(
        id="cr_1",
        title="Adversarial Perturbations",
        authors=["Author C"],
        url="https://doi.org/10.1234/5678",
        origin="crossref",
        year=2022,
        text="Evaluation of token substitutions in detection systems.",
    )

    with patch(
        "veritext.sources.retrieval.search_crossref",
        return_value=[mock_crossref_cand],
    ), patch(
        "veritext.sources.retrieval.search_openalex", return_value=[]
    ), patch(
        "veritext.sources.retrieval.search_arxiv", return_value=[]
    ), patch(
        "veritext.sources.retrieval.search_semantic_scholar", return_value=[]
    ), patch(
        "veritext.sources.retrieval.search_wikipedia", return_value=[]
    ):
        results = find_candidates(
            sample_document,
            use_online=True,
            use_local=False,
            max_sources=5,
            store=store,
        )

        assert len(results) == 1
        assert results[0].origin == "crossref"
        assert results[0].id == "cr_1"


def test_find_candidates_resilience_to_client_failure(tmp_path, sample_document):
    db_file = tmp_path / "resilience_store.sqlite"
    store = ReferenceStore(db_file)

    mock_wiki_cand = SourceCandidate(
        id="wk_1",
        title="Plagiarism and Attribution",
        authors=["Wikipedia Contributors"],
        url="https://en.wikipedia.org/wiki/Plagiarism",
        origin="wikipedia",
        year=None,
        text="Plagiarism is the representation of thoughts or language as original.",
    )

    # Crossref raises an unexpected exception; Wikipedia succeeds
    with patch(
        "veritext.sources.retrieval.search_crossref",
        side_effect=RuntimeError("API crashed!"),
    ), patch(
        "veritext.sources.retrieval.search_openalex", return_value=[]
    ), patch(
        "veritext.sources.retrieval.search_arxiv", return_value=[]
    ), patch(
        "veritext.sources.retrieval.search_semantic_scholar", return_value=[]
    ), patch(
        "veritext.sources.retrieval.search_wikipedia", return_value=[mock_wiki_cand]
    ):
        # find_candidates must NOT raise an exception
        results = find_candidates(
            sample_document,
            use_online=True,
            use_local=False,
            max_sources=5,
            store=store,
        )

        assert len(results) == 1
        assert results[0].origin == "wikipedia"
