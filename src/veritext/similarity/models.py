"""Data models for match spans, source matches, and overall similarity results."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional


@dataclass
class MatchSpan:
    """Represents a matched span between the input document and a source."""

    type: str  # 'exact', 'near_copy', 'paraphrase'
    score: float
    source_id: str
    source_title: str
    source_url: str
    doc_start: int
    doc_end: int
    matched_text: str
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    source_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "score": round(self.score, 4),
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "doc_start": self.doc_start,
            "doc_end": self.doc_end,
            "matched_text": self.matched_text,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "source_text": self.source_text,
            "length_chars": self.doc_end - self.doc_start,
            "metadata": self.metadata,
        }


@dataclass
class SourceMatch:
    """Summary of matches corresponding to a single candidate source."""

    source_id: str
    source_title: str
    source_url: str
    origin: str
    matched_chars: int
    percentage: float
    match_count: int
    spans: List[MatchSpan] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "origin": self.origin,
            "matched_chars": self.matched_chars,
            "percentage": round(self.percentage, 2),
            "match_count": self.match_count,
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclass
class SimilarityResult:
    """Complete plagiarism detection analysis result."""

    document_path: Optional[str]
    total_clean_chars: int
    matched_chars: int
    similarity_percentage: float
    spans: List[MatchSpan] = field(default_factory=list)
    sources: List[SourceMatch] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_path": self.document_path,
            "total_clean_chars": self.total_clean_chars,
            "matched_chars": self.matched_chars,
            "similarity_percentage": round(self.similarity_percentage, 2),
            "span_count": len(self.spans),
            "source_count": len(self.sources),
            "anomaly_count": len(self.anomalies),
            "sources": [s.to_dict() for s in self.sources],
            "spans": [span.to_dict() for span in self.spans],
            "anomalies": self.anomalies,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
