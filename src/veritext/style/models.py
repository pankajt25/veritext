"""Data models for grammar and style feedback."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from veritext.checks.models import Finding


@dataclass
class StyleMetric:
    """Readability scores and structural text metrics."""

    flesch_reading_ease: float
    gunning_fog: float
    total_words: int
    total_sentences: int
    total_syllables: int
    complex_words: int
    reading_ease_label: str
    reading_level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flesch_reading_ease": round(self.flesch_reading_ease, 2),
            "gunning_fog": round(self.gunning_fog, 2),
            "total_words": self.total_words,
            "total_sentences": self.total_sentences,
            "total_syllables": self.total_syllables,
            "complex_words": self.complex_words,
            "reading_ease_label": self.reading_ease_label,
            "reading_level": self.reading_level,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StyleMetric":
        return cls(
            flesch_reading_ease=float(d.get("flesch_reading_ease", 0.0)),
            gunning_fog=float(d.get("gunning_fog", 0.0)),
            total_words=int(d.get("total_words", 0)),
            total_sentences=int(d.get("total_sentences", 0)),
            total_syllables=int(d.get("total_syllables", 0)),
            complex_words=int(d.get("complex_words", 0)),
            reading_ease_label=d.get("reading_ease_label", "Standard"),
            reading_level=d.get("reading_level", "College"),
        )


@dataclass
class StyleResult:
    """Consolidated grammar and style feedback results.

    Note: All style findings have severity 'info' and never alter similarity or AI scores.
    """

    findings: List[Finding] = field(default_factory=list)
    metrics: Optional[StyleMetric] = None
    engine_status: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "engine_status": self.engine_status,
            "finding_count": len(self.findings),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StyleResult":
        findings = [Finding(**f) for f in d.get("findings", [])]
        metrics_dict = d.get("metrics")
        metrics = StyleMetric.from_dict(metrics_dict) if metrics_dict else None
        return cls(
            findings=findings,
            metrics=metrics,
            engine_status=d.get("engine_status", {}),
        )
