"""Plagiarism detection and text similarity engine module."""

from veritext.similarity.engine import check_document
from veritext.similarity.exact import find_exact_matches
from veritext.similarity.models import MatchSpan, SimilarityResult, SourceMatch
from veritext.similarity.scorer import (
    calculate_covered_characters,
    compute_similarity_score,
    merge_character_intervals,
)
from veritext.similarity.semantic import (
    EmbeddingCache,
    encode_texts_cached,
    find_semantic_matches,
)

__all__ = [
    "check_document",
    "find_exact_matches",
    "find_semantic_matches",
    "compute_similarity_score",
    "merge_character_intervals",
    "calculate_covered_characters",
    "MatchSpan",
    "SourceMatch",
    "SimilarityResult",
    "EmbeddingCache",
    "encode_texts_cached",
]
