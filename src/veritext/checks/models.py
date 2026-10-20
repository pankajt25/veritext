"""Data models for document integrity checks and findings."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional


@dataclass
class Finding:
    """Represents a specific suspicious pattern or anomaly found during integrity checks.

    Attributes:
        check: Name of the check (e.g. 'hidden_content', 'citations', 'self_plagiarism', 'code_similarity').
        severity: Severity level ('info', 'warning', 'high').
        message: Human-readable explanation of the finding.
        start: Character offset in the original document text.
        end: Character offset in the original document text.
        details: Extra context, counts, affected words, or metadata.
    """

    check: str
    severity: str
    message: str
    start: int
    end: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes finding to a standard dictionary."""
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "start": self.start,
            "end": self.end,
            "details": self.details,
        }


@dataclass
class CheckResult:
    """Outcome and aggregated findings of a single integrity check.

    Attributes:
        check_name: Identifier for the check module.
        findings: List of individual Finding items.
        summary: Concise plain-language summary of findings.
        passed: Boolean indicating whether the check passed without major issues.
        metadata: Additional diagnostic statistics or execution details.
    """

    check_name: str
    findings: List[Finding] = field(default_factory=list)
    summary: str = ""
    passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes check result to a dictionary."""
        return {
            "check_name": self.check_name,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "passed": self.passed,
            "metadata": self.metadata,
        }


@dataclass
class IntegrityReport:
    """Consolidated integrity report covering all checks run on a document.

    Attributes:
        document_path: Path of the inspected document, if known.
        results: List of CheckResult objects.
        summary: High-level plain-language summary.
        passed: True if all individual checks passed (no warning or high severity findings).
        metadata: Additional session details or runtime diagnostics.
    """

    document_path: Optional[str]
    results: List[CheckResult] = field(default_factory=list)
    summary: str = ""
    passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes integrity report to a dictionary."""
        return {
            "document_path": self.document_path,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "passed": self.passed,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes integrity report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
