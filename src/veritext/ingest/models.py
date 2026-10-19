"""Data models for document ingestion and text representations."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class Sentence:
    """Represents a single sentence with character offsets in the clean text."""
    text: str
    start: int
    end: int

    def __getitem__(self, index: int) -> Any:
        return (self.text, self.start, self.end)[index]

    def __iter__(self) -> Iterator[Any]:
        return iter((self.text, self.start, self.end))

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass
class Paragraph:
    """Represents a paragraph with character offsets in the clean text."""
    text: str
    start: int
    end: int

    def __str__(self) -> str:
        return self.text

    def __getitem__(self, index: int) -> Any:
        return (self.text, self.start, self.end)[index]

    def __iter__(self) -> Iterator[Any]:
        return iter((self.text, self.start, self.end))

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass
class RemovedSection:
    """Records content that was filtered or removed from the raw document."""
    type: str  # e.g. "references", "quote", "latex_math", "code_block", "citation"
    content: str
    start: Optional[int] = None
    end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class Document:
    """Represents an ingested and cleaned document."""
    raw_text: str
    clean_text: str
    sentences: List[Sentence]
    paragraphs: List[Paragraph]
    removed_sections: List[RemovedSection]
    file_type: str
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "file_type": self.file_type,
            "raw_text": self.raw_text,
            "clean_text": self.clean_text,
            "sentences": [s.to_dict() for s in self.sentences],
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "removed_sections": [r.to_dict() for r in self.removed_sections],
            "metadata": self.metadata,
        }
