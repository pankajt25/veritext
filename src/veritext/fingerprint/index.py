"""SQLite inverted index for document winnowing fingerprints."""

from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from veritext.fingerprint.winnow import winnow
from veritext.normalize.normalize import normalize_text
from veritext.sources.models import SourceCandidate
from veritext.sources.store import ReferenceStore


class FingerprintIndex:
    """Manages inverted index of winnowing fingerprints stored in SQLite."""

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
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fingerprints (
                    hash INTEGER NOT NULL,
                    source_id TEXT NOT NULL,
                    passage_id TEXT,
                    position INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fingerprints_hash ON fingerprints(hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fingerprints_source ON fingerprints(source_id)"
            )

    def add_fingerprints(
        self,
        source_id: str,
        passage_id: Optional[str],
        fingerprints: Sequence[Tuple[int, int]],
    ):
        """Batch inserts fingerprints into the SQLite inverted index."""
        if not fingerprints:
            return

        records = [
            (h, source_id, passage_id or "", pos) for h, pos in fingerprints
        ]
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO fingerprints (hash, source_id, passage_id, position)
                VALUES (?, ?, ?, ?)
                """,
                records,
            )

    def is_source_indexed(self, source_id: str) -> bool:
        """Checks if a source is already indexed in the fingerprints table."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM fingerprints WHERE source_id = ? LIMIT 1",
                (source_id,),
            ).fetchone()
            return row is not None

    def index_text(
        self,
        source_id: str,
        passage_id: Optional[str],
        text: str,
        k: int = 5,
        w: int = 4,
    ) -> List[Tuple[int, int]]:
        """
        Normalizes text, extracts winnowing fingerprints, and saves them to the index.
        """
        canonical = normalize_text(text)
        fps = winnow(canonical.tokens, k=k, w=w)
        self.add_fingerprints(source_id, passage_id, fps)
        return fps

    def index_source_candidate(
        self, candidate: SourceCandidate, k: int = 5, w: int = 4
    ) -> List[Tuple[int, int]]:
        """Indexes a SourceCandidate if not already present in the fingerprint index."""
        if self.is_source_indexed(candidate.id):
            return []
        return self.index_text(
            source_id=candidate.id,
            passage_id=None,
            text=f"{candidate.title}\n\n{candidate.text}",
            k=k,
            w=w,
        )

    def index_all_library_sources(
        self, store: Optional[ReferenceStore] = None, k: int = 5, w: int = 4
    ) -> int:
        """
        Iterates through all sources in the ReferenceStore and indexes their fingerprints.
        Returns the count of newly indexed sources.
        """
        ref_store = store or ReferenceStore(self.db_path)
        indexed_count = 0

        with ref_store._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, title, text FROM sources"
            ).fetchall()
            for r in rows:
                s_id = r["id"]
                if not self.is_source_indexed(s_id):
                    full_text = f"{r['title']}\n\n{r['text']}"
                    self.index_text(s_id, None, full_text, k=k, w=w)
                    indexed_count += 1

        return indexed_count

    def lookup_hashes(
        self, hashes: Sequence[int]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Finds matching indexed fingerprints for the given hash values.
        Returns mapping from hash value to list of match metadata dictionaries.
        """
        if not hashes:
            return {}

        hash_list = list(set(hashes))
        results: Dict[int, List[Dict[str, Any]]] = {h: [] for h in hash_list}

        # Query in chunks to prevent SQLite parameter limits
        chunk_size = 900
        with self._get_connection() as conn:
            for i in range(0, len(hash_list), chunk_size):
                chunk = hash_list[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"""
                    SELECT hash, source_id, passage_id, position
                    FROM fingerprints
                    WHERE hash IN ({placeholders})
                    """,
                    chunk,
                )
                for row in cursor.fetchall():
                    h = row["hash"]
                    if h in results:
                        results[h].append(
                            {
                                "source_id": row["source_id"],
                                "passage_id": row["passage_id"],
                                "position": row["position"],
                            }
                        )

        return results

    def stats(self) -> Dict[str, Any]:
        """Returns statistics on the fingerprint inverted index."""
        with self._get_connection() as conn:
            total_fps = conn.execute(
                "SELECT COUNT(*) FROM fingerprints"
            ).fetchone()[0]
            distinct_hashes = conn.execute(
                "SELECT COUNT(DISTINCT hash) FROM fingerprints"
            ).fetchone()[0]
            distinct_sources = conn.execute(
                "SELECT COUNT(DISTINCT source_id) FROM fingerprints"
            ).fetchone()[0]

        return {
            "total_fingerprints": total_fps,
            "distinct_hashes": distinct_hashes,
            "distinct_sources": distinct_sources,
        }

    def clear(self):
        """Clears all fingerprints from the index."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM fingerprints")
