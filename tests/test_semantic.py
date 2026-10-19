"""Tests for semantic vector search, sentence windowing, embedding caching, and paraphrase matching."""

import numpy as np
import pytest
from veritext.ingest.models import Document, Paragraph, Sentence
from veritext.similarity.models import MatchSpan
from veritext.similarity.semantic import (
    EmbeddingCache,
    build_sentence_windows,
    encode_texts_cached,
    find_semantic_matches,
    is_span_covered,
)
from veritext.sources.models import SourceCandidate


class DummyModel:
    """Mock embedding model that generates synthetic normalized vectors."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(
        self,
        sentences,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        vecs = []
        for s in sentences:
            # Deterministic pseudo-vector based on string hash
            rng = np.random.RandomState(abs(hash(s)) % (2**31))
            vec = rng.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            vecs.append(vec / (norm if norm != 0 else 1.0))
        return np.array(vecs, dtype=np.float32)


def test_sentence_windows_and_overlap_filtering():
    doc = Document(
        raw_text="Sentence one. Sentence two. Sentence three.",
        clean_text="Sentence one. Sentence two. Sentence three.",
        file_type=".txt",
        sentences=[
            Sentence("Sentence one.", 0, 13),
            Sentence("Sentence two.", 14, 27),
            Sentence("Sentence three.", 28, 43),
        ],
        paragraphs=[Paragraph("Sentence one. Sentence two. Sentence three.", 0, 43)],
        removed_sections=[],
    )

    windows = build_sentence_windows(doc, window_size=2)
    # Should contain individual sentences and pairs:
    # 3 single + 2 pairs = 5 windows
    assert len(windows) == 5
    assert windows[0] == ("Sentence one.", 0, 13)
    assert windows[1] == ("Sentence one. Sentence two.", 0, 27)

    # Test span covered
    exact_span = MatchSpan(
        type="exact",
        score=1.0,
        source_id="s1",
        source_title="T",
        source_url="",
        doc_start=0,
        doc_end=20,
        matched_text="Sentence one. Sente",
    )
    # Window (0, 13) is 100% covered by exact_span [0, 20]
    assert is_span_covered(0, 13, [exact_span]) is True
    # Window (28, 43) does not overlap
    assert is_span_covered(28, 43, [exact_span]) is False


def test_embedding_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "test_emb.sqlite")
    dummy_model = DummyModel()

    texts = ["Attention mechanisms in NLP", "Convolutional computer vision"]
    vecs = encode_texts_cached(texts, cache=cache, model=dummy_model)
    assert vecs.shape == (2, 384)

    # Second call should fetch directly from SQLite cache
    cached_map, missing = cache.get_many(texts)
    assert len(missing) == 0
    assert len(cached_map) == 2


def test_mock_semantic_vector_search(tmp_path):
    cache = EmbeddingCache(tmp_path / "mock_sim.sqlite")

    # Document
    doc = Document(
        raw_text="The attention mechanism routes information dynamically across sequences.",
        clean_text="The attention mechanism routes information dynamically across sequences.",
        file_type=".txt",
        sentences=[
            Sentence(
                "The attention mechanism routes information dynamically across sequences.",
                0,
                72,
            )
        ],
        paragraphs=[
            Paragraph(
                "The attention mechanism routes information dynamically across sequences.",
                0,
                72,
            )
        ],
        removed_sections=[],
    )

    # Candidate source with identical semantics
    cand = SourceCandidate(
        id="src_sem_1",
        title="Attention Basics",
        authors=["Researcher"],
        url="https://example.org/attention",
        origin="web",
        year=2021,
        text="The attention mechanism routes information dynamically across sequences.",
    )

    # Using dummy model where identical text generates identical vector (cosine sim = 1.0)
    dummy_model = DummyModel()
    spans = find_semantic_matches(
        document=doc,
        candidates=[cand],
        threshold=0.80,
        cache=cache,
        model=dummy_model,
    )

    assert len(spans) >= 1
    assert spans[0].source_id == "src_sem_1"
    assert spans[0].score >= 0.80


@pytest.mark.slow
def test_real_paraphrase_detection():
    """
    Slow integration test verifying paraphrase detection using all-MiniLM-L6-v2.
    Marked with @pytest.mark.slow so it does not slow down default test execution.
    """
    from veritext.similarity.semantic import get_embedding_model

    model = get_embedding_model()
    assert model is not None

    sentence_orig = "The transformer architecture relies upon multi-head self-attention mechanisms to dispense with recurrence entirely."
    sentence_para = "Rather than using recurrent connections, modern transformer models utilize multi-headed self-attention structures to process text."
    sentence_unrelated = "Ancient Roman harbor docks utilized volcanic ash cement to resist wave erosion in salt water."

    embs = model.encode(
        [sentence_orig, sentence_para, sentence_unrelated],
        normalize_embeddings=True,
    )

    sim_paraphrase = float(np.dot(embs[0], embs[1]))
    sim_unrelated = float(np.dot(embs[0], embs[2]))

    # Paraphrased sentences should exhibit substantial semantic similarity
    assert sim_paraphrase > 0.65
    # Unrelated sentences should exhibit significantly lower similarity
    assert sim_unrelated < 0.35
    assert sim_paraphrase > sim_unrelated + 0.30
