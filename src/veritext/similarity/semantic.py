"""Semantic vector search and paraphrase detection using SentenceTransformers and FAISS."""

import hashlib
import logging
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from veritext.ingest.models import Document, Sentence
from veritext.similarity.models import MatchSpan
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.75
EMBEDDING_DIM = 384

_model = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """Lazily loads the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading SentenceTransformer: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


class EmbeddingCache:
    """On-disk SQLite cache for sentence embeddings to prevent redundant model calls."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            db_path = base_dir / "data" / "cache" / "embeddings_cache.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def get_many(
        self, texts: List[str]
    ) -> Tuple[Dict[int, np.ndarray], List[int]]:
        """
        Retrieves cached embeddings for a list of texts.
        Returns:
            (cached_map: {index: vector}, missing_indices: [index, ...])
        """
        cached: Dict[int, np.ndarray] = {}
        missing: List[int] = []

        hashes = [
            hashlib.sha256(t.strip().encode("utf-8")).hexdigest()
            for t in texts
        ]

        with self._get_connection() as conn:
            for idx, h in enumerate(hashes):
                row = conn.execute(
                    "SELECT vector_blob, dim FROM embeddings WHERE text_hash = ?",
                    (h,),
                ).fetchone()
                if row:
                    blob, dim = row
                    vec = np.frombuffer(blob, dtype=np.float32)
                    cached[idx] = vec
                else:
                    missing.append(idx)

        return cached, missing

    def set_many(self, texts: List[str], vectors: np.ndarray):
        """Saves a batch of computed vectors to the database cache."""
        now = time.time()
        records = []
        for text, vec in zip(texts, vectors):
            h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            blob = vec.astype(np.float32).tobytes()
            records.append((h, blob, len(vec), now))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings (text_hash, vector_blob, dim, created_at)
                VALUES (?, ?, ?, ?)
                """,
                records,
            )


def encode_texts_cached(
    texts: List[str],
    cache: Optional[EmbeddingCache] = None,
    model=None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Computes normalized embeddings for texts using EmbeddingCache and SentenceTransformer.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    emb_cache = cache or EmbeddingCache()
    cached_map, missing_indices = emb_cache.get_many(texts)

    vectors = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for idx, vec in cached_map.items():
        vectors[idx] = vec

    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        encoder = model or get_embedding_model()
        computed = encoder.encode(
            missing_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for idx, vec in zip(missing_indices, computed):
            vectors[idx] = vec
        emb_cache.set_many(missing_texts, computed)

    # Normalize vectors to unit length for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return vectors.astype(np.float32)


def is_span_covered(
    start: int, end: int, exact_spans: List[MatchSpan], threshold_ratio: float = 0.5
) -> bool:
    """Checks if a character interval overlaps substantially with an existing exact match span."""
    span_len = max(1, end - start)
    for s in exact_spans:
        overlap_start = max(start, s.doc_start)
        overlap_end = min(end, s.doc_end)
        if overlap_end > overlap_start:
            overlap_len = overlap_end - overlap_start
            if (overlap_len / span_len) >= threshold_ratio:
                return True
    return False


def build_sentence_windows(
    document: Document, window_size: int = 2
) -> List[Tuple[str, int, int]]:
    """
    Constructs sliding sentence windows (text, doc_char_start, doc_char_end) from document.
    """
    sentences = document.sentences
    if not sentences:
        # Fallback to paragraphs if sentences are empty
        return [
            (p.text, p.start, p.end)
            for p in document.paragraphs
            if len(p.text.strip()) > 30
        ]

    windows: List[Tuple[str, int, int]] = []
    n = len(sentences)

    for i in range(n):
        s_curr = sentences[i]
        # Single sentence window
        windows.append((s_curr.text, s_curr.start, s_curr.end))

        # Two-sentence combined window
        if window_size >= 2 and i + 1 < n:
            s_next = sentences[i + 1]
            win_text = f"{s_curr.text} {s_next.text}"
            win_start = s_curr.start
            win_end = s_next.end
            windows.append((win_text, win_start, win_end))

    return windows


def find_semantic_matches(
    document: Document,
    candidates: List[SourceCandidate],
    exact_spans: Optional[List[MatchSpan]] = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    cache: Optional[EmbeddingCache] = None,
    model=None,
    top_k: int = 3,
) -> List[MatchSpan]:
    """
    Performs semantic vector search using FAISS nearest-neighbor matching
    to detect paraphrased and near-copy passages, skipping spans covered by exact matches.
    """
    covered_spans = exact_spans or []

    # 1. Prepare document sentence windows not covered by exact matches
    all_windows = build_sentence_windows(document, window_size=2)
    doc_windows = [
        (w_text, w_start, w_end)
        for w_text, w_start, w_end in all_windows
        if not is_span_covered(w_start, w_end, covered_spans)
        and len(w_text.split()) >= 6
    ]

    if not doc_windows or not candidates:
        return []

    # 2. Prepare candidate source passages (sentence-level and multi-sentence windows)
    candidate_passages: List[Dict[str, Any]] = []
    for cand in candidates:
        cand_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", cand.text)
            if len(s.strip().split()) >= 5
        ]
        cand_windows = list(cand_sentences)
        for i in range(len(cand_sentences) - 1):
            cand_windows.append(f"{cand_sentences[i]} {cand_sentences[i+1]}")
        for p in cand.text.split("\n\n"):
            p_strip = p.strip()
            if len(p_strip.split()) >= 6 and p_strip not in cand_windows:
                cand_windows.append(p_strip)

        if not cand_windows and cand.text.strip():
            cand_windows = [cand.text.strip()]

        for p_idx, p_str in enumerate(cand_windows):
            candidate_passages.append(
                {
                    "source_id": cand.id,
                    "source_title": cand.title,
                    "source_url": cand.url,
                    "text": p_str,
                    "passage_index": p_idx,
                }
            )

    if not candidate_passages:
        return []

    # 3. Compute Embeddings with Caching
    doc_texts = [w[0] for w in doc_windows]
    cand_texts = [cp["text"] for cp in candidate_passages]

    doc_vectors = encode_texts_cached(doc_texts, cache=cache, model=model)
    cand_vectors = encode_texts_cached(cand_texts, cache=cache, model=model)

    # 4. Perform Vector Nearest-Neighbor Search using normalized cosine matrix multiplication
    match_pairs: List[Tuple[int, int, float]] = []
    # Cosine similarity matrix: (num_doc_windows, num_cand_passages)
    sim_matrix = np.dot(doc_vectors, cand_vectors.T)
    k = min(top_k, len(candidate_passages))
    for d_idx in range(len(doc_windows)):
        top_indices = np.argsort(sim_matrix[d_idx])[::-1][:k]
        for c_idx in top_indices:
            sim = float(sim_matrix[d_idx, c_idx])
            if sim >= threshold:
                match_pairs.append((d_idx, int(c_idx), sim))

    # 5. Formulate MatchSpans
    semantic_spans: List[MatchSpan] = []
    seen_window_starts = set()

    for d_idx, c_idx, score in match_pairs:
        w_text, w_start, w_end = doc_windows[d_idx]
        if w_start in seen_window_starts:
            continue
        seen_window_starts.add(w_start)

        target_source = candidate_passages[c_idx]
        match_type = "near_copy" if score >= 0.94 else "paraphrase"

        semantic_spans.append(
            MatchSpan(
                type=match_type,
                score=score,
                source_id=target_source["source_id"],
                source_title=target_source["source_title"],
                source_url=target_source["source_url"],
                doc_start=w_start,
                doc_end=w_end,
                matched_text=w_text,
                source_text=target_source["text"][:300],
                metadata={
                    "similarity_score": round(score, 4),
                    "search_type": "vector_cosine",
                },
            )
        )

    semantic_spans.sort(key=lambda s: s.doc_start)
    return semantic_spans
