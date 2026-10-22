"""Evaluation dataset loader and manifest query utilities."""

import csv
from pathlib import Path
from typing import Sequence

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "eval"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_DIR / "manifest.csv"
DEFAULT_DOCS_DIR = DEFAULT_DATA_DIR / "documents"
DEFAULT_REFS_DIR = DEFAULT_DATA_DIR / "references"


def load_manifest(manifest_path: Path | str | None = None) -> list[dict[str, str]]:
    """Loads manifest.csv as a list of dictionary records."""
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_document(filename: str, docs_dir: Path | str | None = None) -> str:
    """Loads content of a single document from docs_dir."""
    directory = Path(docs_dir) if docs_dir else DEFAULT_DOCS_DIR
    doc_path = directory / filename
    if not doc_path.exists():
        raise FileNotFoundError(f"Evaluation document not found at {doc_path}")
    return doc_path.read_text(encoding="utf-8")


def load_references(refs_dir: Path | str | None = None) -> dict[str, str]:
    """Loads all reference text passages from refs_dir."""
    directory = Path(refs_dir) if refs_dir else DEFAULT_REFS_DIR
    if not directory.exists():
        raise FileNotFoundError(f"References directory not found at {directory}")

    refs = {}
    for ref_file in sorted(directory.glob("ref_*.txt")):
        refs[ref_file.name] = ref_file.read_text(encoding="utf-8")
    return refs


def filter_manifest_by_group(
    manifest: Sequence[dict[str, str]],
    groups: str | Sequence[str],
) -> list[dict[str, str]]:
    """Filters manifest rows by group name or list of group names."""
    if isinstance(groups, str):
        groups = [groups]
    group_set = set(groups)
    return [row for row in manifest if row.get("group") in group_set]
