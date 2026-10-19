"""Data models for candidate source retrieval and search queries."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SourceCandidate:
    """Represents a candidate source document or passage found via search or local indexing."""

    id: str
    title: str
    authors: List[str]
    url: str
    origin: str  # e.g., 'crossref', 'openalex', 'arxiv', 'semantic_scholar', 'wikipedia', 'local_library'
    year: Optional[int]
    text: str
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "origin": self.origin,
            "year": self.year,
            "text": self.text,
            "retrieved_at": self.retrieved_at,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceCandidate":
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            url=data.get("url", ""),
            origin=data.get("origin", "unknown"),
            year=data.get("year"),
            text=data.get("text", ""),
            retrieved_at=data.get(
                "retrieved_at", datetime.now(timezone.utc).isoformat()
            ),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Query:
    """Represents a search query generated from document paragraphs."""

    text: str
    origin_paragraph_index: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "origin_paragraph_index": self.origin_paragraph_index,
            "keywords": self.keywords,
            "weight": self.weight,
        }
