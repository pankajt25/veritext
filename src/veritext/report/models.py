"""Data models for unified Veritext reports and highlight spans."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional

from veritext.ai_detect.models import AIResult, SentenceAIResult
from veritext.checks.models import IntegrityReport
from veritext.similarity.models import SimilarityResult

DEFAULT_DISCLAIMER = (
    "Disclaimer: Veritext provides statistical indicators and heuristic signals for self-review "
    "and pedagogical evaluation. These scores do not constitute definitive proof of academic "
    "misconduct or generative AI authorship. Automated checks can produce false positives and "
    "false negatives; human review and academic judgment are always required."
)


@dataclass
class HighlightSpan:
    """Represents a highlighted segment within the original document text.

    Attributes:
        id: Unique identifier for this highlight span (e.g. 'span_1').
        start: Character start offset in the ORIGINAL document text.
        end: Character end offset in the ORIGINAL document text.
        category: Highlight category ('exact match', 'paraphrase', 'likely AI',
                  'hidden content', 'citation issue', 'self-overlap').
        severity: Severity level ('info', 'warning', 'high').
        message: Descriptive explanation of the finding or match.
        source: Source title, author, or URL if applicable.
        ignored: Whether the user has chosen to ignore/dismiss this finding.
        details: Additional contextual data or engine diagnostics.
    """

    id: str
    start: int
    end: int
    category: str
    severity: str
    message: str
    source: Optional[str] = None
    ignored: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
            "ignored": self.ignored,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HighlightSpan":
        return cls(
            id=data.get("id", ""),
            start=data.get("start", 0),
            end=data.get("end", 0),
            category=data.get("category", "info"),
            severity=data.get("severity", "info"),
            message=data.get("message", ""),
            source=data.get("source"),
            ignored=data.get("ignored", False),
            details=data.get("details", {}),
        )


@dataclass
class FullReport:
    """Unified report combining plagiarism, AI text detection, and integrity checks."""

    document_name: str
    document_path: Optional[str]
    original_text: str
    clean_text: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    settings: Dict[str, Any] = field(default_factory=dict)
    similarity: Optional[SimilarityResult] = None
    ai_detect: Optional[AIResult] = None
    integrity: Optional[IntegrityReport] = None
    highlights: List[HighlightSpan] = field(default_factory=list)
    summary: str = ""
    disclaimer: str = DEFAULT_DISCLAIMER
    profile: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the report to a JSON-compatible dictionary."""
        return {
            "document_name": self.document_name,
            "document_path": self.document_path,
            "original_text": self.original_text,
            "clean_text": self.clean_text,
            "timestamp": self.timestamp,
            "settings": self.settings,
            "similarity": self.similarity.to_dict() if self.similarity else None,
            "ai_detect": self.ai_detect.to_dict() if self.ai_detect else None,
            "integrity": self.integrity.to_dict() if self.integrity else None,
            "highlights": [h.to_dict() for h in self.highlights],
            "profile": self.profile,
            "plan": self.plan,
            "summary": self.summary,
            "disclaimer": self.disclaimer,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes the report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FullReport":
        """Deserializes a report dictionary back to a FullReport instance."""
        sim_data = data.get("similarity")
        ai_data = data.get("ai_detect")
        integ_data = data.get("integrity")

        # Reconstruct SimilarityResult if present
        similarity_obj = None
        if sim_data:
            from veritext.similarity.models import MatchSpan, SourceMatch
            spans = [
                MatchSpan(
                    type=s.get("type", "exact"),
                    score=s.get("score", 1.0),
                    source_id=s.get("source_id", ""),
                    source_title=s.get("source_title", ""),
                    source_url=s.get("source_url", ""),
                    doc_start=s.get("doc_start", 0),
                    doc_end=s.get("doc_end", 0),
                    matched_text=s.get("matched_text", ""),
                    source_start=s.get("source_start"),
                    source_end=s.get("source_end"),
                    source_text=s.get("source_text", ""),
                    metadata=s.get("metadata", {}),
                )
                for s in sim_data.get("spans", [])
            ]
            sources = [
                SourceMatch(
                    source_id=sm.get("source_id", ""),
                    source_title=sm.get("source_title", ""),
                    source_url=sm.get("source_url", ""),
                    origin=sm.get("origin", "local"),
                    matched_chars=sm.get("matched_chars", 0),
                    percentage=sm.get("percentage", 0.0),
                    match_count=sm.get("match_count", 0),
                    spans=[
                        MatchSpan(
                            type=s.get("type", "exact"),
                            score=s.get("score", 1.0),
                            source_id=s.get("source_id", ""),
                            source_title=s.get("source_title", ""),
                            source_url=s.get("source_url", ""),
                            doc_start=s.get("doc_start", 0),
                            doc_end=s.get("doc_end", 0),
                            matched_text=s.get("matched_text", ""),
                            source_start=s.get("source_start"),
                            source_end=s.get("source_end"),
                            source_text=s.get("source_text", ""),
                            metadata=s.get("metadata", {}),
                        )
                        for s in sm.get("spans", [])
                    ],
                )
                for sm in sim_data.get("sources", [])
            ]
            similarity_obj = SimilarityResult(
                document_path=sim_data.get("document_path"),
                total_clean_chars=sim_data.get("total_clean_chars", 0),
                matched_chars=sim_data.get("matched_chars", 0),
                similarity_percentage=sim_data.get("similarity_percentage", 0.0),
                spans=spans,
                sources=sources,
                anomalies=sim_data.get("anomalies", []),
                metadata=sim_data.get("metadata", {}),
            )

        # Reconstruct AIResult if present
        ai_obj = None
        if ai_data:
            from veritext.ai_detect.models import AIResult, SentenceAIResult
            sentences = [
                SentenceAIResult(
                    text=s.get("text", ""),
                    doc_start=s.get("doc_start", s.get("start", 0)),
                    doc_end=s.get("doc_end", s.get("end", 0)),
                    label=s.get("label", "likely human"),
                    score=s.get("score", 0.0),
                    perplexity=s.get("perplexity"),
                    explanation=s.get("explanation", ""),
                )
                for s in ai_data.get("sentences", [])
            ]
            ai_obj = AIResult(
                status=ai_data.get("status", "success"),
                word_count=ai_data.get("word_count", 0),
                overall_probability=ai_data.get("overall_probability", ai_data.get("ai_probability")),
                confidence_band=ai_data.get("confidence_band", ai_data.get("confidence", "low")),
                reduced_confidence=ai_data.get("reduced_confidence", False),
                mode=ai_data.get("mode", "full"),
                sentences=sentences,
                features=ai_data.get("features", {}),
                explanations=ai_data.get("explanations", ai_data.get("drivers", [])),
                disclaimer=ai_data.get("disclaimer", DEFAULT_DISCLAIMER),
                metadata=ai_data.get("metadata", {}),
            )

        # Reconstruct IntegrityReport if present
        integrity_obj = None
        if integ_data:
            from veritext.checks.models import CheckResult, Finding
            results = [
                CheckResult(
                    check_name=cr.get("check_name", ""),
                    findings=[
                        Finding(
                            check=f.get("check", ""),
                            severity=f.get("severity", "info"),
                            message=f.get("message", ""),
                            start=f.get("start", 0),
                            end=f.get("end", 0),
                            details=f.get("details", {}),
                        )
                        for f in cr.get("findings", [])
                    ],
                    summary=cr.get("summary", ""),
                    passed=cr.get("passed", True),
                    metadata=cr.get("metadata", {}),
                )
                for cr in integ_data.get("results", [])
            ]
            integrity_obj = IntegrityReport(
                document_path=integ_data.get("document_path"),
                results=results,
                summary=integ_data.get("summary", ""),
                passed=integ_data.get("passed", True),
                metadata=integ_data.get("metadata", {}),
            )

        highlights = [
            HighlightSpan.from_dict(h) for h in data.get("highlights", [])
        ]

        return cls(
            document_name=data.get("document_name", "Untitled"),
            document_path=data.get("document_path"),
            original_text=data.get("original_text", ""),
            clean_text=data.get("clean_text", ""),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            settings=data.get("settings", {}),
            similarity=similarity_obj,
            ai_detect=ai_obj,
            integrity=integrity_obj,
            highlights=highlights,
            profile=data.get("profile"),
            plan=data.get("plan"),
            summary=data.get("summary", ""),
            disclaimer=data.get("disclaimer", DEFAULT_DISCLAIMER),
            metadata=data.get("metadata", {}),
        )
