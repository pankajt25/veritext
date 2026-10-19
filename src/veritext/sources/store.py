"""SQLite database store with FTS5 full-text search and de-duplication."""

import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Union

from veritext.sources.models import SourceCandidate


def compute_content_hash(
    title: str, url: str = "", text: str = ""
) -> str:
    """Computes a canonical SHA-256 hash for de-duplication."""
    norm_title = re.sub(r"\W+", "", title.lower())
    norm_url = re.sub(r"^https?://(www\.)?", "", url.lower().rstrip("/"))
    seed = f"{norm_title}|{norm_url}"
    if not norm_title and not norm_url:
        seed = re.sub(r"\W+", "", text.lower()[:200])
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


class ReferenceStore:
    """Manages the local reference library in SQLite with FTS5 indexing."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            db_path = base_dir / "data" / "veritext.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT,
                    url TEXT,
                    origin TEXT NOT NULL,
                    year INTEGER,
                    text TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS passages (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    passage_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE CASCADE
                )
                """
            )
            # Full-Text Search FTS5 virtual table
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
                    text,
                    passage_id UNINDEXED,
                    source_id UNINDEXED,
                    title UNINDEXED
                )
                """
            )

    def add_source(
        self, candidate: SourceCandidate, passages: Optional[List[str]] = None
    ) -> bool:
        """
        Adds a candidate source and its passages to the store.
        Returns True if newly inserted, False if a duplicate already exists.
        """
        chash = compute_content_hash(
            candidate.title, candidate.url, candidate.text
        )

        with self._get_connection() as conn:
            # Check for existing duplicate by hash or url
            existing = conn.execute(
                "SELECT id FROM sources WHERE content_hash = ?", (chash,)
            ).fetchone()
            if existing:
                return False

            authors_json = json.dumps(candidate.authors)
            meta_json = json.dumps(candidate.metadata)

            conn.execute(
                """
                INSERT INTO sources (id, title, authors_json, url, origin, year, text, content_hash, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.title,
                    authors_json,
                    candidate.url,
                    candidate.origin,
                    candidate.year,
                    candidate.text,
                    chash,
                    meta_json,
                    candidate.retrieved_at,
                ),
            )

            # Passages indexing
            passage_list = passages
            if not passage_list:
                # Segment text into paragraphs/chunks
                passage_list = [
                    p.strip()
                    for p in candidate.text.split("\n\n")
                    if len(p.strip()) > 30
                ]
                if not passage_list and candidate.text.strip():
                    passage_list = [candidate.text.strip()]

            for idx, p_text in enumerate(passage_list):
                p_id = f"{candidate.id}_p{idx}"
                token_count = len(p_text.split())
                conn.execute(
                    """
                    INSERT INTO passages (id, source_id, passage_index, text, token_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (p_id, candidate.id, idx, p_text, token_count),
                )
                conn.execute(
                    """
                    INSERT INTO passages_fts (text, passage_id, source_id, title)
                    VALUES (?, ?, ?, ?)
                    """,
                    (p_text, p_id, candidate.id, candidate.title),
                )

        return True

    def search_fts(
        self, query_text: str, limit: int = 10
    ) -> List[SourceCandidate]:
        """
        Queries the FTS5 full-text index using keywords.
        Returns matching SourceCandidate objects ranked by BM25 match quality.
        """
        if not query_text or not query_text.strip():
            return []

        # Sanitize query for FTS5: extract alphanumeric tokens only
        tokens = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query_text)
        if not tokens:
            return []

        # Join tokens with OR or standard match
        fts_query = " OR ".join(tokens)

        candidates: List[SourceCandidate] = []
        seen_source_ids = set()

        with self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT f.passage_id, f.source_id, f.text AS passage_text, f.rank,
                           s.title, s.authors_json, s.url, s.origin, s.year, s.text AS full_text,
                           s.metadata_json, s.created_at
                    FROM passages_fts f
                    JOIN sources s ON s.id = f.source_id
                    WHERE passages_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (fts_query, limit * 2),
                )
                rows = cursor.fetchall()

                for row in rows:
                    src_id = row["source_id"]
                    if src_id in seen_source_ids:
                        continue
                    seen_source_ids.add(src_id)

                    authors = (
                        json.loads(row["authors_json"])
                        if row["authors_json"]
                        else []
                    )
                    metadata = (
                        json.loads(row["metadata_json"])
                        if row["metadata_json"]
                        else {}
                    )

                    candidates.append(
                        SourceCandidate(
                            id=src_id,
                            title=row["title"],
                            authors=authors,
                            url=row["url"] or "",
                            origin=row["origin"] or "local_library",
                            year=row["year"],
                            text=row["passage_text"] or row["full_text"],
                            retrieved_at=row["created_at"],
                            score=abs(float(row["rank"])) if row["rank"] else 0.0,
                            metadata=metadata,
                        )
                    )
                    if len(candidates) >= limit:
                        break

            except sqlite3.OperationalError:
                # Fallback to simple LIKE query if FTS expression failed
                first_token = f"%{tokens[0]}%"
                cursor = conn.execute(
                    """
                    SELECT s.id, s.title, s.authors_json, s.url, s.origin, s.year, s.text, s.metadata_json, s.created_at
                    FROM sources s
                    WHERE s.text LIKE ? OR s.title LIKE ?
                    LIMIT ?
                    """,
                    (first_token, first_token, limit),
                )
                for row in cursor.fetchall():
                    if row["id"] in seen_source_ids:
                        continue
                    seen_source_ids.add(row["id"])
                    authors = (
                        json.loads(row["authors_json"])
                        if row["authors_json"]
                        else []
                    )
                    metadata = (
                        json.loads(row["metadata_json"])
                        if row["metadata_json"]
                        else {}
                    )
                    candidates.append(
                        SourceCandidate(
                            id=row["id"],
                            title=row["title"],
                            authors=authors,
                            url=row["url"] or "",
                            origin=row["origin"] or "local_library",
                            year=row["year"],
                            text=row["text"],
                            retrieved_at=row["created_at"],
                            metadata=metadata,
                        )
                    )

        return candidates

    def get_source(self, source_id: str) -> Optional[SourceCandidate]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE id = ?", (source_id,)
            ).fetchone()
            if not row:
                return None
            authors = (
                json.loads(row["authors_json"]) if row["authors_json"] else []
            )
            metadata = (
                json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            )
            return SourceCandidate(
                id=row["id"],
                title=row["title"],
                authors=authors,
                url=row["url"] or "",
                origin=row["origin"],
                year=row["year"],
                text=row["text"],
                retrieved_at=row["created_at"],
                metadata=metadata,
            )

    def stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[
                0
            ]
            passage_count = conn.execute(
                "SELECT COUNT(*) FROM passages"
            ).fetchone()[0]
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "sources_count": source_count,
            "passages_count": passage_count,
            "db_size_bytes": size_bytes,
        }

    def clear(self):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM passages_fts")
            conn.execute("DELETE FROM passages")
            conn.execute("DELETE FROM sources")
