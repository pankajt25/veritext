"""Unit tests for SQLite ReferenceStore and local library indexing."""

from pathlib import Path
import pytest

from veritext.sources.library import index_default_references, index_directory, index_file
from veritext.sources.models import SourceCandidate
from veritext.sources.store import ReferenceStore, compute_content_hash


def test_store_add_and_deduplicate(tmp_path):
    db_file = tmp_path / "test_store.sqlite"
    store = ReferenceStore(db_file)

    cand = SourceCandidate(
        id="test_001",
        title="Attention Is All You Need",
        authors=["A. Vaswani", "N. Shazeer"],
        url="https://arxiv.org/abs/1706.03762",
        origin="arxiv",
        year=2017,
        text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    )

    # First insertion must succeed
    assert store.add_source(cand) is True

    # Duplicate insertion must be rejected
    assert store.add_source(cand) is False

    stats = store.stats()
    assert stats["sources_count"] == 1
    assert stats["passages_count"] >= 1


def test_store_fts_search(tmp_path):
    db_file = tmp_path / "test_store.sqlite"
    store = ReferenceStore(db_file)

    cand1 = SourceCandidate(
        id="c1",
        title="Deep Reinforcement Learning in Robotics",
        authors=["Author One"],
        url="https://example.com/c1",
        origin="local_library",
        year=2020,
        text="Robotics algorithms utilize policy gradients and reinforcement techniques for trajectory optimization.",
    )
    cand2 = SourceCandidate(
        id="c2",
        title="Natural Language Understanding",
        authors=["Author Two"],
        url="https://example.com/c2",
        origin="local_library",
        year=2021,
        text="Syntactic parsing and semantics in computational linguistics for document understanding.",
    )

    store.add_source(cand1)
    store.add_source(cand2)

    matches = store.search_fts("robotics reinforcement", limit=5)
    assert len(matches) >= 1
    assert matches[0].title == "Deep Reinforcement Learning in Robotics"

    matches_nlp = store.search_fts("linguistics semantics", limit=5)
    assert len(matches_nlp) >= 1
    assert matches_nlp[0].title == "Natural Language Understanding"


def test_index_directory_and_references(tmp_path):
    db_file = tmp_path / "test_library.sqlite"
    store = ReferenceStore(db_file)

    # Index default references created in data/references/
    count = index_default_references(store=store)
    assert count >= 3

    stats = store.stats()
    assert stats["sources_count"] >= 3
    assert stats["passages_count"] >= 3

    # FTS search across the indexed references
    turing_matches = store.search_fts("imitation game interrogator", limit=5)
    assert len(turing_matches) >= 1
    assert any("Turing" in m.title or "Computing Machinery" in m.title for m in turing_matches)

    shannon_matches = store.search_fts("entropy communication channel", limit=5)
    assert len(shannon_matches) >= 1
    assert any("Shannon" in m.title or "Communication" in m.title for m in shannon_matches)
