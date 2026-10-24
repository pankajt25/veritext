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


def test_semantic_produces_paraphrase_spans_exact_does_not_mocked(tmp_path):
    """Regression test proving that semantic matching finds paraphrase spans where exact winnowing finds none.

    Uses mock embeddings (no network / fast) on heavily reworded sentences sharing zero 5-token k-grams.
    """
    from veritext.fingerprint.index import FingerprintIndex
    from veritext.fingerprint.winnow import winnow
    from veritext.ingest.pipeline import split_paragraphs, split_sentences
    from veritext.normalize.normalize import normalize_text
    from veritext.similarity.exact import find_exact_matches
    from veritext.similarity.scorer import compute_similarity_score

    # Setup reference and candidate
    ref_text = (
        "Consensus protocols in distributed systems guarantee that a cluster of independent nodes "
        "agrees upon a unified replicated state machine despite network partitions and arbitrary failures."
    )
    # Rephrased document sharing zero 5-token k-grams with reference
    query_text = (
        "Agreement algorithms across decentralized networks ensure that multiple compute servers "
        "converge on an identical transition history notwithstanding link disruptions and machine crashes."
    )

    # 1. Exact Winnowing
    canon_ref = normalize_text(ref_text)
    canon_q = normalize_text(query_text)

    # Verify zero 5-gram overlap between query and reference
    q_tokens = canon_q.tokens
    ref_tokens = canon_ref.tokens
    q_5grams = set(tuple(q_tokens[i:i+5]) for i in range(len(q_tokens) - 4))
    ref_5grams = set(tuple(ref_tokens[i:i+5]) for i in range(len(ref_tokens) - 4))
    assert len(q_5grams & ref_5grams) == 0, "Query and reference must share 0 five-grams"

    db_path = tmp_path / "test_exact.sqlite"
    index = FingerprintIndex(db_path=db_path)
    cand = SourceCandidate(
        id="ref_consensus",
        title="Distributed Consensus",
        authors=["Lamport et al."],
        url="https://example.org/consensus",
        origin="web",
        year=2020,
        text=ref_text,
    )
    index.index_source_candidate(cand)
    sources_map = {"ref_consensus": cand}

    fps_q = winnow(q_tokens, k=5, w=4)
    exact_spans = find_exact_matches(
        doc_clean_text=query_text,
        canonical=canon_q,
        doc_fingerprints=fps_q,
        index=index,
        sources_map=sources_map,
        min_words=5,
    )
    # Exact winnowing must produce ZERO matches
    assert len(exact_spans) == 0
    score_exact = compute_similarity_score(query_text, exact_spans, sources_map).similarity_percentage
    assert score_exact == 0.0

    # 2. Semantic Matching with Mock Model
    doc = Document(
        raw_text=query_text,
        clean_text=query_text,
        sentences=split_sentences(query_text),
        paragraphs=split_paragraphs(query_text),
        removed_sections=[],
        file_type="txt",
    )

    class ControlledParaphraseModel:
        def encode(self, sentences, **kwargs):
            # Map both semantic concepts to identical embedding vector
            vecs = []
            for s in sentences:
                if "Agreement algorithms" in s or "Consensus protocols" in s:
                    v = np.ones(384, dtype=np.float32)
                else:
                    v = np.zeros(384, dtype=np.float32)
                norm = np.linalg.norm(v)
                vecs.append(v / (norm if norm != 0 else 1.0))
            return np.array(vecs, dtype=np.float32)

    cache = EmbeddingCache(tmp_path / "test_sem_cache.sqlite")
    sem_spans = find_semantic_matches(
        document=doc,
        candidates=[cand],
        exact_spans=exact_spans,
        threshold=0.75,
        cache=cache,
        model=ControlledParaphraseModel(),
    )

    # Semantic matching MUST produce paraphrase matches
    assert len(sem_spans) >= 1
    assert sem_spans[0].type in ("near_copy", "paraphrase", "possible paraphrase (review)")
    assert sem_spans[0].score >= 0.75

    # By default in v1.0.3, headline similarity score ignores semantic matches
    score_headline_default = compute_similarity_score(query_text, sem_spans, sources_map).similarity_percentage
    assert score_headline_default == 0.0

    # With opt-in flag include_semantic_in_score=True, overall similarity jumps from 0.0% to > 50%
    score_semantic = compute_similarity_score(
        query_text, sem_spans, sources_map, include_single_window_semantic=True, include_semantic_in_score=True
    ).similarity_percentage
    assert score_semantic > 50.0


def test_minimum_consecutive_windows_and_single_window_exclusion(tmp_path):
    """Fast test with mocked vectors verifying the consecutive windows rule and single-window exclusion."""
    from veritext.similarity.scorer import compute_similarity_score

    cache = EmbeddingCache(tmp_path / "consec_test.sqlite")

    v1 = np.ones(384, dtype=np.float32)
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.zeros(384, dtype=np.float32)
    v2[0] = 1.0

    class MockControlledModel:
        def encode(self, sentences, **kwargs):
            vecs = []
            for s in sentences:
                if "Alpha sentence" in s:
                    vecs.append(v1)
                elif s.strip() == s3.strip():
                    vecs.append(v2)
                else:
                    rng = np.random.RandomState(abs(hash(s)) % (2**31))
                    rnd = rng.randn(384).astype(np.float32)
                    vecs.append(rnd / np.linalg.norm(rnd))
            return np.array(vecs, dtype=np.float32)

    # 4-sentence document:
    # Sentence 0 & 1 match Cand A (consecutive run = 2)
    # Sentence 2 is unrelated
    # Sentence 3 matches Cand B (isolated single window, run = 1)
    s0 = "Alpha sentence zero describes first consecutive mechanism."
    s1 = "Alpha sentence one describes second consecutive mechanism."
    s2 = "Beta sentence two discusses unrelated historical navigation methods."
    s3 = "Gamma sentence three explains an isolated cryptographic definition."

    raw = f"{s0} {s1} {s2} {s3}"
    c0_s, c0_e = 0, len(s0)
    c1_s, c1_e = c0_e + 1, c0_e + 1 + len(s1)
    c2_s, c2_e = c1_e + 1, c1_e + 1 + len(s2)
    c3_s, c3_e = c2_e + 1, c2_e + 1 + len(s3)

    doc = Document(
        raw_text=raw,
        clean_text=raw,
        file_type=".txt",
        sentences=[
            Sentence(s0, c0_s, c0_e),
            Sentence(s1, c1_s, c1_e),
            Sentence(s2, c2_s, c2_e),
            Sentence(s3, c3_s, c3_e),
        ],
        paragraphs=[Paragraph(raw, 0, len(raw))],
        removed_sections=[],
    )

    cand_a = SourceCandidate(
        id="cand_a",
        title="Candidate A",
        authors=["Author A"],
        url="local://cand_a",
        text=f"{s0} {s1}",
        origin="local_library",
        year=2024,
    )
    cand_b = SourceCandidate(
        id="cand_b",
        title="Candidate B",
        authors=["Author B"],
        url="local://cand_b",
        text=s3,
        origin="local_library",
        year=2024,
    )

    # Run semantic matching with min_consecutive_windows=2
    spans = find_semantic_matches(
        document=doc,
        candidates=[cand_a, cand_b],
        threshold=0.75,
        cache=cache,
        model=MockControlledModel(),
        min_consecutive_windows=2,
        include_single_window=False,
    )

    # All semantic spans must display as 'possible paraphrase (review)'
    for sp in spans:
        assert sp.type == "possible paraphrase (review)"

    spans_a = [s for s in spans if s.source_id == "cand_a"]
    spans_b = [s for s in spans if s.source_id == "cand_b"]

    assert len(spans_a) >= 1
    assert len(spans_b) >= 1

    # cand_a matches are consecutive (run >= 2) -> is_single_window == False
    for sp in spans_a:
        assert sp.metadata["is_single_window"] is False
        assert sp.metadata["consecutive_windows"] >= 2

    # cand_b match is isolated (run == 1) -> is_single_window == True
    for sp in spans_b:
        assert sp.metadata["is_single_window"] is True
        assert sp.metadata["consecutive_windows"] == 1

    # Default score calculation in v1.0.3: semantic matches are NOT added to headline percentage
    sources_map = {"cand_a": cand_a, "cand_b": cand_b}
    res_default = compute_similarity_score(raw, spans, sources_map)
    assert res_default.similarity_percentage == 0.0
    assert res_default.metadata["paraphrase_leads_count"] == len(spans)
    assert res_default.metadata["score_label"] == "exact & near-copy matches only"

    # Opt-in score calculation: single-window match from cand_b is EXCLUDED from score when include_single_window_semantic=False
    res_optin = compute_similarity_score(
        raw, spans, sources_map, include_semantic_in_score=True, include_single_window_semantic=False
    )
    assert res_optin.metadata["score_label"] == "score includes unverified semantic leads"
    cand_b_match = next((s for s in res_optin.sources if s.source_id == "cand_b"), None)
    assert cand_b_match is not None
    assert cand_b_match.percentage == 0.0
    # Headline percentage in opt-in mode reflects cand_a
    cand_a_match = next(s for s in res_optin.sources if s.source_id == "cand_a")
    assert cand_a_match.percentage > 0.0
    assert res_optin.similarity_percentage == cand_a_match.percentage

    # When user also enables include_single_window_semantic=True in opt-in mode, cand_b is included
    res_with_single = compute_similarity_score(
        raw, spans, sources_map, include_semantic_in_score=True, include_single_window_semantic=True
    )
    cand_b_match_incl = next(s for s in res_with_single.sources if s.source_id == "cand_b")
    assert cand_b_match_incl.percentage > 0.0
    assert res_with_single.similarity_percentage > res_optin.similarity_percentage


@pytest.mark.slow
def test_real_paraphrase_fixtures_minilm_detection():
    """Slow integration test verifying semantic detection on real paraphrase fixtures against references.

    Evaluates para_01, para_02, and para_03 against ref_01, ref_02, and ref_03 using all-MiniLM-L6-v2.
    Proves that exact winnowing finds 0 matches on these fixtures while semantic search detects all three.
    """
    from pathlib import Path
    from veritext.eval.dataset import load_document
    from veritext.fingerprint.index import FingerprintIndex
    from veritext.fingerprint.winnow import winnow
    from veritext.ingest.pipeline import split_paragraphs, split_sentences
    from veritext.normalize.normalize import normalize_text
    from veritext.similarity.exact import find_exact_matches
    from veritext.similarity.scorer import compute_similarity_score
    from veritext.similarity.semantic import find_semantic_matches, get_embedding_model

    model = get_embedding_model()
    assert model is not None, "MiniLM model must be available locally"

    base_dir = Path(__file__).resolve().parent.parent / "data" / "eval"
    docs_dir = base_dir / "documents"
    refs_dir = base_dir / "references"

    fixtures = [("para_01.txt", "ref_01.txt"), ("para_02.txt", "ref_02.txt"), ("para_03.txt", "ref_03.txt")]

    for para_fn, ref_fn in fixtures:
        para_txt = load_document(para_fn, docs_dir)
        ref_txt = load_document(ref_fn, refs_dir)

        cand = SourceCandidate(
            id=ref_fn,
            title=ref_fn,
            authors=["Benchmark Reference"],
            url=f"local://{ref_fn}",
            origin="eval",
            year=2024,
            text=ref_txt,
        )
        sources_map = {ref_fn: cand}

        # Exact winnowing
        canon = normalize_text(para_txt)
        fps = winnow(canon.tokens, k=5, w=4)
        idx_mock = FingerprintIndex()
        idx_mock.index_source_candidate(cand)
        exact_spans = find_exact_matches(para_txt, canon, fps, idx_mock, sources_map, min_words=6)
        sim_exact = compute_similarity_score(para_txt, exact_spans, sources_map).similarity_percentage

        # Semantic matching
        doc_obj = Document(
            raw_text=para_txt,
            clean_text=para_txt,
            sentences=split_sentences(para_txt),
            paragraphs=split_paragraphs(para_txt),
            removed_sections=[],
            file_type="txt",
            path=para_fn,
        )
        sem_spans = find_semantic_matches(
            document=doc_obj,
            candidates=[cand],
            exact_spans=exact_spans,
            threshold=0.70,
            model=model,
        )
        res_default = compute_similarity_score(para_txt, exact_spans + sem_spans, sources_map)
        sim_sem = compute_similarity_score(
            para_txt, exact_spans + sem_spans, sources_map, include_semantic_in_score=True
        ).similarity_percentage

        # Exact winnowing finds minimal or zero spans
        assert sim_exact < 10.0, f"Expected exact score < 10% on paraphrase, got {sim_exact}%"
        # Headline default ignores semantic spans
        assert res_default.similarity_percentage < 10.0
        # Semantic search finds multiple high-confidence paraphrase spans
        assert len(sem_spans) >= 2, f"Expected >= 2 semantic spans on {para_fn}, got {len(sem_spans)}"
        assert res_default.metadata["paraphrase_leads_count"] >= 2
        assert sim_sem >= 50.0, f"Expected semantic similarity >= 50% on {para_fn}, got {sim_sem}%"


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
