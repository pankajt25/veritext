"""Local reference library indexing documents using Stage 1 ingestion into SQLite FTS5."""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Union

from veritext.ingest.pipeline import load_document
from veritext.sources.models import SourceCandidate
from veritext.sources.store import ReferenceStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".tex", ".latex", ".docx", ".pdf"}


def _extract_title_from_text(text: str, fallback_filename: str) -> str:
    """Derives a human-readable title from the first non-empty line or filename."""
    lines = [line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line) <= 120:
            return first_line
    return Path(fallback_filename).stem.replace("_", " ").title()


def index_file(
    file_path: Union[str, Path], store: Optional[ReferenceStore] = None
) -> bool:
    """
    Ingests and indexes a single document into the local SQLite reference library.
    Returns True if the document was newly added, False if duplicate or skipped.
    """
    path = Path(file_path).resolve()
    if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    ref_store = store or ReferenceStore()

    try:
        doc = load_document(path)
    except Exception as e:
        logger.warning(f"Failed to ingest local document '{path}': {e}")
        return False

    title = _extract_title_from_text(doc.clean_text, path.name)
    file_id = "local_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]

    candidate = SourceCandidate(
        id=file_id,
        title=title,
        authors=["Local Library"],
        url=f"file://{path}",
        origin="local_library",
        year=None,
        text=doc.clean_text,
        metadata={
            "file_type": doc.file_type,
            "file_path": str(path),
            "sentence_count": len(doc.sentences),
            "paragraph_count": len(doc.paragraphs),
        },
    )

    passages = [p.text for p in doc.paragraphs if len(p.text.strip()) > 20]
    return ref_store.add_source(candidate, passages=passages)


def index_directory(
    dir_path: Union[str, Path], store: Optional[ReferenceStore] = None
) -> int:
    """
    Indexes all supported documents inside dir_path into the local reference library.
    Returns the count of newly indexed files.
    """
    directory = Path(dir_path).resolve()
    if not directory.exists() or not directory.is_dir():
        return 0

    ref_store = store or ReferenceStore()
    newly_added = 0

    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if index_file(file_path, store=ref_store):
                newly_added += 1

    return newly_added


def index_default_references(store: Optional[ReferenceStore] = None) -> int:
    """Indexes all files in the standard data/references/ directory."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    ref_dir = base_dir / "data" / "references"
    return index_directory(ref_dir, store=store)
