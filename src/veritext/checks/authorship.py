"""Authorship and revision timeline check from local document metadata.

Extracts creation timestamps, modification history, revision counts, editing durations,
and software producers from DOCX, PDF, and text document metadata.

Strict Guarantees:
- Neutral 'info' wording only; never accusatory.
- Explicitly disclaimed: not keystroke tracking and not proof of authorship.
- Completely local and offline; nothing leaves the user's machine.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from veritext.checks.models import CheckResult, Finding
from veritext.ingest.models import Document

logger = logging.getLogger(__name__)

AUTHORSHIP_DISCLAIMER = (
    "Informational Notice: Document metadata timeline is derived from container file properties. "
    "This check does not perform keystroke tracking, typing biometrics, or remote monitoring, "
    "and does not constitute proof of authorship or academic integrity status."
)

# Forbidden accusatory terms to strictly enforce neutral, constructive phrasing
FORBIDDEN_ACCUSATORY_WORDS = [
    "cheat",
    "cheater",
    "cheating",
    "fake",
    "fraud",
    "fraudulent",
    "dishonest",
    "illicit",
    "suspicious",
    "suspect",
    "guilty",
    "stolen",
    "plagiarist",
    "falsified",
]


@dataclass
class TimelineEvent:
    """Represents a single milestone event in a document's lifecycle."""

    timestamp: Optional[str]
    event_type: str  # e.g., 'created', 'modified', 'printed', 'revision'
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "description": self.description,
            "details": self.details,
        }


def parse_pdf_date(date_str: Optional[str]) -> Optional[str]:
    """Parses PDF timestamp format 'D:YYYYMMDDHHmmSS...' into ISO 8601 string."""
    if not date_str or not isinstance(date_str, str):
        return None
    clean = date_str.strip()
    if clean.startswith("D:"):
        clean = clean[2:]
    match = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", clean)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        hour = match.group(4) or "00"
        minute = match.group(5) or "00"
        second = match.group(6) or "00"
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"
    return str(date_str)


def extract_metadata_timeline(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extracts metadata properties and builds a chronological revision timeline."""
    path = Path(file_path)
    if not path.is_file():
        return {
            "events": [],
            "properties": {},
            "summary": "Document file not available on local disk for metadata extraction.",
        }

    ext = path.suffix.lower()
    properties: Dict[str, Any] = {
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "format": ext.lstrip("."),
    }
    events: List[TimelineEvent] = []

    # 1. DOCX metadata
    if ext == ".docx":
        try:
            import docx

            doc = docx.Document(str(path))
            cp = doc.core_properties
            if cp.author:
                properties["author"] = cp.author
            if cp.last_modified_by:
                properties["last_modified_by"] = cp.last_modified_by
            if cp.revision:
                properties["revision_count"] = cp.revision
            if cp.created:
                c_iso = cp.created.isoformat()
                properties["created"] = c_iso
                events.append(TimelineEvent(c_iso, "created", f"Document created by {cp.author or 'author'}"))
            if cp.modified:
                m_iso = cp.modified.isoformat()
                properties["modified"] = m_iso
                events.append(TimelineEvent(m_iso, "modified", f"Document last saved by {cp.last_modified_by or 'editor'}"))
            if cp.last_printed:
                p_iso = cp.last_printed.isoformat()
                properties["last_printed"] = p_iso
                events.append(TimelineEvent(p_iso, "printed", "Document sent to print"))
        except Exception as e:
            logger.debug(f"Could not read DOCX properties from {path}: {e}")

    # 2. PDF metadata
    elif ext == ".pdf":
        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                meta = pdf.metadata or {}
                if meta.get("Author"):
                    properties["author"] = meta["Author"]
                if meta.get("Creator"):
                    properties["creator_software"] = meta["Creator"]
                if meta.get("Producer"):
                    properties["producer_software"] = meta["Producer"]
                if meta.get("CreationDate"):
                    c_iso = parse_pdf_date(meta["CreationDate"])
                    properties["created"] = c_iso
                    events.append(TimelineEvent(c_iso, "created", f"PDF created via {meta.get('Creator', 'software')}"))
                if meta.get("ModDate"):
                    m_iso = parse_pdf_date(meta["ModDate"])
                    properties["modified"] = m_iso
                    events.append(TimelineEvent(m_iso, "modified", "PDF modified / saved"))
        except Exception as e:
            logger.debug(f"Could not read PDF metadata from {path}: {e}")

    # Filesystem fallback if container metadata missing
    if not events:
        st = path.stat()
        m_time = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        properties["file_modified"] = m_time
        events.append(TimelineEvent(m_time, "modified", "File modified on filesystem"))

    # Sort events chronologically
    events.sort(key=lambda ev: ev.timestamp or "")

    return {
        "properties": properties,
        "events": [ev.to_dict() for ev in events],
        "disclaimer": AUTHORSHIP_DISCLAIMER,
    }


def check_authorship_timeline(document: Union[Document, str, Path]) -> CheckResult:
    """Executes the authorship timeline check with neutral info-level findings."""
    doc_path: Optional[Path] = None
    if isinstance(document, (str, Path)):
        doc_path = Path(document)
    elif isinstance(document, Document) and document.path:
        doc_path = Path(document.path)

    findings: List[Finding] = []
    meta_info: Dict[str, Any] = {}

    if doc_path and doc_path.is_file():
        meta_info = extract_metadata_timeline(doc_path)
        events = meta_info.get("events", [])
        props = meta_info.get("properties", {})

        # Formulate neutral timeline summary
        parts = []
        if "author" in props:
            parts.append(f"Recorded author: '{props['author']}'")
        if "created" in props:
            parts.append(f"Created: {props['created']}")
        if "modified" in props:
            parts.append(f"Modified: {props['modified']}")
        if "revision_count" in props:
            parts.append(f"Revisions: {props['revision_count']}")

        desc = (
            f"Document metadata timeline: {', '.join(parts)}. "
            f"Found {len(events)} lifecycle event(s)."
        ) if parts else f"Recorded {len(events)} filesystem event(s)."

        # Validate that message contains no forbidden terms
        for word in FORBIDDEN_ACCUSATORY_WORDS:
            assert word not in desc.lower(), f"Violated neutral wording policy with word '{word}'"

        findings.append(
            Finding(
                check="authorship_timeline",
                severity="info",  # Always informational
                message=desc,
                start=0,
                end=0,
                details={
                    "properties": props,
                    "events": events,
                    "disclaimer": AUTHORSHIP_DISCLAIMER,
                },
            )
        )
        summary = (
            f"Authorship metadata check recorded {len(events)} timeline milestone(s). "
            f"All findings informational; not keystroke tracking and not proof of authorship."
        )
    else:
        summary = (
            "No local file container available for metadata timeline extraction. "
            "Raw text input evaluated."
        )

    return CheckResult(
        check_name="authorship_timeline",
        findings=findings,
        summary=summary,
        passed=True,  # Always passes; informational only
        metadata={
            "timeline": meta_info,
            "disclaimer": AUTHORSHIP_DISCLAIMER,
            "severity_level": "info",
        },
    )
