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
    lead_count: int = 0
    strong_lead_count: int = 0
    weak_lead_count: int = 0
    lead_percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "origin": self.origin,
            "matched_chars": self.matched_chars,
            "percentage": round(self.percentage, 2),
            "match_count": self.match_count,
            "lead_count": self.lead_count,
            "strong_lead_count": self.strong_lead_count,
            "weak_lead_count": self.weak_lead_count,
            "lead_percentage": round(self.lead_percentage, 2),
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclass
class SimilarityResult:
    """Complete plagiarism detection analysis result."""

    total_clean_chars: int
    matched_chars: int
    similarity_percentage: float
    document_path: Optional[str] = None
    spans: List[MatchSpan] = field(default_factory=list)
    sources: List[SourceMatch] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        meta = dict(self.metadata)
        if "all_original_spans" in meta:
            meta["all_original_spans"] = [
                s.to_dict() if hasattr(s, "to_dict") else s
                for s in meta["all_original_spans"]
            ]
        return {
            "document_path": self.document_path,
            "total_clean_chars": self.total_clean_chars,
            "matched_chars": self.matched_chars,
            "similarity_percentage": round(self.similarity_percentage, 2),
            "score_label": meta.get("score_label", "exact & near-copy matches only"),
            "paraphrase_leads_count": meta.get("paraphrase_leads_count", 0),
            "paraphrase_leads_percentage": meta.get("paraphrase_leads_percentage", 0.0),
            "strong_leads_count": meta.get("strong_leads_count", 0),
            "weak_leads_count": meta.get("weak_leads_count", 0),
            "explanatory_line": meta.get("explanatory_line", ""),
            "interpretation_guide": meta.get("interpretation_guide", ""),
            "span_count": len(self.spans),
            "source_count": len(self.sources),
            "anomaly_count": len(self.anomalies),
            "sources": [s.to_dict() for s in self.sources],
            "spans": [span.to_dict() for span in self.spans],
            "anomalies": self.anomalies,
            "metadata": meta,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
