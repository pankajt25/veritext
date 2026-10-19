"""Data models for text normalization and anomaly tracking."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Anomaly:
    """Represents a suspicious character anomaly (homoglyph, zero-width, or control character)."""

    type: str  # 'homoglyph', 'zero_width', 'control_character'
    character: str
    position: int
    replacement: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "character": self.character,
            "position": self.position,
            "replacement": self.replacement,
            "description": self.description,
            "codepoint": f"U+{ord(self.character):04X}",
        }


@dataclass
class CanonicalText:
    """
    Represents the output of text canonicalisation.
    Contains normalized/lemmatized text, tokens, exact offset mapping to original text,
    and a list of detected adversarial/formatting anomalies.
    """

    normalized_text: str
    tokens: List[str]
    offset_map: List[Tuple[int, int]]
    anomalies: List[Anomaly] = field(default_factory=list)
    original_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized_text": self.normalized_text,
            "tokens": self.tokens,
            "offset_map": self.offset_map,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "token_count": len(self.tokens),
            "anomaly_count": len(self.anomalies),
        }
