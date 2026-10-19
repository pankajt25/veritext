"""Data models for AI-written text detection and confidence reporting."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional

DISCLAIMER_TEXT = (
    "Disclaimer: AI-text detection is probabilistic and may misidentify human or formulaic text. "
    "Scores should be used as an indicator for self-review, not as definitive proof of authorship."
)


@dataclass
class SentenceAIResult:
    """Per-sentence classification result with exact character offsets from the ingested document."""

    text: str
    doc_start: int
    doc_end: int
    label: str  # 'likely human', 'uncertain', 'likely AI'
    score: float  # 0.0 to 1.0
    perplexity: Optional[float] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "doc_start": self.doc_start,
            "doc_end": self.doc_end,
            "label": self.label,
            "score": round(self.score, 4),
            "perplexity": round(self.perplexity, 2) if self.perplexity is not None else None,
            "explanation": self.explanation,
        }


@dataclass
class AIResult:
    """Document-level AI detection result object with plain-language explanations and disclaimer."""

    status: str  # 'success', 'insufficient_text', 'error'
    word_count: int
    overall_probability: Optional[float]  # 0.0 to 100.0 or None if insufficient text
    confidence_band: str  # 'low', 'medium', 'high', 'none'
    reduced_confidence: bool  # True if operating on stylometry alone
    mode: str  # 'full', 'stylometry_only'
    sentences: List[SentenceAIResult] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER_TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "word_count": self.word_count,
            "overall_probability": (
                round(self.overall_probability, 1)
                if self.overall_probability is not None
                else None
            ),
            "confidence_band": self.confidence_band,
            "reduced_confidence": self.reduced_confidence,
            "mode": self.mode,
            "sentence_count": len(self.sentences),
            "sentences": [s.to_dict() for s in self.sentences],
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "explanations": self.explanations,
            "disclaimer": self.disclaimer,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
