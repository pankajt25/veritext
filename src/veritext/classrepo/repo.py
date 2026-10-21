"""Local class repository manager for cross-submission and collusion matching.

Stores submissions per class in data/class_repository/<class_name>/index.sqlite.
Supports --fingerprints-only mode to preserve student privacy by storing only
mathematical hashes and offsets rather than full text.
"""

from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from veritext.fingerprint.winnow import winnow
from veritext.ingest.loaders import LOADER_MAP
from veritext.ingest.pipeline import load_document
from veritext.normalize.normalize import normalize_text
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path("data/class_repository")

OPT_IN_CONSENT_NOTICE = (
    "Privacy & Consent Note: Submissions in class repositories are processed locally offline. "
    "With --fingerprints-only, only mathematical winnowing hashes and offsets are stored; "
    "raw document text is not preserved on disk. Submissions receive anonymised IDs."
)


def generate_submission_id(class_name: str, filename: str) -> str:
    """Generates a stable, anonymised submission identifier."""
    raw = f"{class_name}:{filename}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"sub_{h}"


class ClassRepositoryManager:
    """Manages class repositories, submission indexing, and privacy settings."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir or DEFAULT_BASE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_class_dir(self, class_name: str) -> Path:
        # Sanitize class name to prevent directory traversal
        safe_name = "".join(c for c in class_name if c.isalnum() or c in ("-", "_")).strip()
        if not safe_name:
            raise ValueError(f"Invalid class name: '{class_name}'")
        return self.base_dir / safe_name

    def _get_db_conn(self, class_name: str) -> sqlite3.Connection:
        cdir = self._get_class_dir(class_name)
        cdir.mkdir(parents=True, exist_ok=True)
        db_path = cdir / "index.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        self._init_db(conn)
        return conn

    def _init_db(self, conn: sqlite3.Connection):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                fingerprints_only INTEGER NOT NULL,
                text TEXT,
                word_count INTEGER NOT NULL,
                fingerprint_count INTEGER NOT NULL,
                added_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fingerprints (
                hash INTEGER NOT NULL,
                submission_id TEXT NOT NULL,
                position INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_class_fp_hash ON fingerprints(hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_class_fp_sub ON fingerprints(submission_id)"
        )

    def add_submission(
        self,
        class_name: str,
        file_path: Union[str, Path],
        fingerprints_only: bool = False,
    ) -> str:
        """Adds a single submission to the named class repository.

        Args:
            class_name: Name of class repository.
            file_path: Path to the document.
            fingerprints_only: If True, stores winnowing hashes without saving prose.

        Returns:
            The anonymised submission ID.
        """
        p = Path(file_path).resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Submission file not found: {p}")

        ext = p.suffix.lower()
        if ext not in LOADER_MAP:
            raise ValueError(f"Unsupported file format '{ext}' for {p.name}")

        doc = load_document(p)
        clean_text = doc.clean_text or doc.raw_text or ""
        words = clean_text.split()
        word_count = len(words)

        canonical = normalize_text(clean_text)
        fps = winnow(canonical.tokens, k=5, w=4)

        sub_id = generate_submission_id(class_name, p.name)
        now_iso = datetime.now(timezone.utc).isoformat()

        # If fingerprints_only, discard text
        stored_text = None if fingerprints_only else clean_text

        with self._get_db_conn(class_name) as conn:
            # Delete any previous record with the same ID
            conn.execute("DELETE FROM submissions WHERE id = ?", (sub_id,))
            conn.execute("DELETE FROM fingerprints WHERE submission_id = ?", (sub_id,))

            conn.execute(
                """
                INSERT INTO submissions
                (id, filename, fingerprints_only, text, word_count, fingerprint_count, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sub_id,
                    p.name,
                    1 if fingerprints_only else 0,
                    stored_text,
                    word_count,
                    len(fps),
                    now_iso,
                ),
            )

            if fps:
                fp_records = [(h, sub_id, pos) for h, pos in fps]
                conn.executemany(
                    "INSERT INTO fingerprints (hash, submission_id, position) VALUES (?, ?, ?)",
                    fp_records,
                )

        return sub_id

    def add_target(
        self,
        class_name: str,
        target_path: Union[str, Path],
        fingerprints_only: bool = False,
    ) -> List[Tuple[str, str]]:
        """Adds a single file or an entire directory of submissions.

        Returns:
            List of (filename, submission_id) tuples.
        """
        p = Path(target_path)
        if not p.exists():
            raise FileNotFoundError(f"Target not found: {p}")

        results: List[Tuple[str, str]] = []
        if p.is_file():
            sub_id = self.add_submission(class_name, p, fingerprints_only=fingerprints_only)
            results.append((p.name, sub_id))
        elif p.is_dir():
            files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in LOADER_MAP])
            for f in files:
                try:
                    sub_id = self.add_submission(class_name, f, fingerprints_only=fingerprints_only)
                    results.append((f.name, sub_id))
                except Exception as e:
                    logger.warning(f"Could not add {f.name} to class {class_name}: {e}")

        return results

    def list_classes(self) -> List[Dict[str, Any]]:
        """Lists all classes with submission counts and privacy statistics."""
        classes = []
        if not self.base_dir.exists():
            return []

        for item in sorted(self.base_dir.iterdir()):
            if item.is_dir() and (item / "index.sqlite").exists():
                class_name = item.name
                with self._get_db_conn(class_name) as conn:
                    total_subs = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
                    fp_only = conn.execute(
                        "SELECT COUNT(*) FROM submissions WHERE fingerprints_only = 1"
                    ).fetchone()[0]
                    total_fps = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]

                classes.append({
                    "class_name": class_name,
                    "submission_count": total_subs,
                    "fingerprints_only_count": fp_only,
                    "total_fingerprints": total_fps,
                    "path": str(item),
                })
        return classes

    def list_submissions(self, class_name: str) -> List[Dict[str, Any]]:
        """Lists all submissions inside a given class repository."""
        cdir = self._get_class_dir(class_name)
        if not (cdir / "index.sqlite").exists():
            return []

        with self._get_db_conn(class_name) as conn:
            rows = conn.execute(
                """
                SELECT id, filename, fingerprints_only, word_count, fingerprint_count, added_at
                FROM submissions ORDER BY added_at DESC
                """
            ).fetchall()

            return [
                {
                    "id": r["id"],
                    "filename": r["filename"],
                    "fingerprints_only": bool(r["fingerprints_only"]),
                    "word_count": r["word_count"],
                    "fingerprint_count": r["fingerprint_count"],
                    "added_at": r["added_at"],
                }
                for r in rows
            ]

    def delete_class(self, class_name: str) -> bool:
        """Deletes a specific class repository."""
        cdir = self._get_class_dir(class_name)
        if not cdir.exists():
            return False

        import shutil
        shutil.rmtree(cdir)
        return True

    def purge_all(self) -> int:
        """Deletes all class repositories."""
        count = 0
        if self.base_dir.exists():
            for item in list(self.base_dir.iterdir()):
                if item.is_dir() and item.name not in (".git",):
                    import shutil
                    shutil.rmtree(item)
                    count += 1
        return count

    def get_candidates(
        self, class_name: str, show_names: bool = False
    ) -> Tuple[List[SourceCandidate], Dict[str, List[Tuple[int, int]]]]:
        """Retrieves submissions as SourceCandidates and their fingerprints.

        Returns:
            Tuple of (candidate_list, mapping_of_sub_id_to_fingerprints).
        """
        cdir = self._get_class_dir(class_name)
        if not (cdir / "index.sqlite").exists():
            return [], {}

        candidates: List[SourceCandidate] = []
        fps_map: Dict[str, List[Tuple[int, int]]] = {}

        with self._get_db_conn(class_name) as conn:
            sub_rows = conn.execute(
                "SELECT id, filename, fingerprints_only, text, word_count FROM submissions"
            ).fetchall()

            for sr in sub_rows:
                s_id = sr["id"]
                is_fp_only = bool(sr["fingerprints_only"])
                title = sr["filename"] if show_names else f"Class Submission {s_id}"
                body_text = sr["text"] or f"[Fingerprints-only submission: {s_id}]"

                cand = SourceCandidate(
                    id=f"class_{s_id}",
                    title=title,
                    authors=["Class Participant"],
                    url="",
                    origin="class submission",
                    year=None,
                    text=body_text,
                    metadata={
                        "submission_id": s_id,
                        "class_name": class_name,
                        "fingerprints_only": is_fp_only,
                        "show_names": show_names,
                        "real_filename": sr["filename"] if show_names else None,
                    },
                )
                candidates.append(cand)

                fp_rows = conn.execute(
                    "SELECT hash, position FROM fingerprints WHERE submission_id = ?",
                    (s_id,),
                ).fetchall()
                fps_map[cand.id] = [(r["hash"], r["position"]) for r in fp_rows]

        return candidates, fps_map
