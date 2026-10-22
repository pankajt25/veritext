"""SQLite-backed job queue and single-worker background processor for Veritext API."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Union
import urllib.request

from veritext.api.auth import sign_webhook_payload
from veritext.checks.runner import run_all_checks
from veritext.ingest.pipeline import load_document
from veritext.report.builder import build_report
from veritext.report.html_export import export_report_html
from veritext.report.pdf_export import export_report_pdf
from veritext.report.seal import compute_seal
from veritext.similarity.engine import check_document
from veritext.style.linter import lint_style

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "api_jobs.sqlite"
)

REPORTS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "reports"
)


class JobQueue:
    """SQLite-backed FIFO job queue for Veritext analysis tasks."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    file_path TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    options_json TEXT,
                    report_id TEXT,
                    error_message TEXT,
                    webhook_url TEXT,
                    webhook_secret TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, id)")
            conn.commit()

    def add_event(self, job_id: str, event_type: str, data: Dict[str, Any]) -> int:
        """Appends an event to the job event log."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO job_events (job_id, event_type, data_json, created_at) VALUES (?, ?, ?, ?)",
                (job_id, event_type, json.dumps(data), now),
            )
            conn.commit()
            return cur.lastrowid

    def get_events(self, job_id: str, after_id: int = 0) -> List[Dict[str, Any]]:
        """Fetches ordered events for a job strictly after after_id."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, event_type, data_json, created_at FROM job_events WHERE job_id = ? AND id > ? ORDER BY id ASC",
                (job_id, after_id),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "event_type": r["event_type"],
                    "data": json.loads(r["data_json"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def enqueue_job(
        self,
        job_id: str,
        file_path: str,
        original_filename: str,
        options: Dict[str, Any],
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Adds a new job in 'queued' state."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, created_at, file_path, original_filename,
                    options_json, webhook_url, webhook_secret
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    now,
                    file_path,
                    original_filename,
                    json.dumps(options),
                    webhook_url,
                    webhook_secret,
                ),
            )
            conn.commit()

        self.add_event(job_id, "status", {"job_id": job_id, "status": "queued", "progress": 0.0})

        return {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
        }

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves job state by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def fetch_next_pending(self) -> Optional[Dict[str, Any]]:
        """Atomically retrieves the oldest 'queued' job and marks it 'running'."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            job_id = row["job_id"]
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE job_id = ?",
                (now, job_id),
            )
            conn.commit()
            return dict(row)

    def mark_completed(self, job_id: str, report_id: str):
        """Marks a job as completed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'completed', finished_at = ?, report_id = ? WHERE job_id = ?",
                (now, report_id, job_id),
            )
            conn.commit()

    def mark_failed(self, job_id: str, error: str):
        """Marks a job as failed with error details."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', finished_at = ?, error_message = ? WHERE job_id = ?",
                (now, error, job_id),
            )
            conn.commit()

    def count_queued(self) -> int:
        """Returns number of jobs currently waiting in the queue."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM jobs WHERE status = 'queued'"
            ).fetchone()
            return row["cnt"] if row else 0


class SingleWorker:
    """Single background worker thread ensuring memory safety on 8 GB systems.

    Processes one document at a time without multi-process contention.
    """

    def __init__(self, queue: JobQueue, poll_interval: float = 1.0):
        self.queue = queue
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="VeritextWorker"
        )
        self._thread.start()
        logger.info("Veritext single background worker started.")

    def stop(self):
        """Signals worker thread to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("Veritext background worker stopped.")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            job = self.queue.fetch_next_pending()
            if not job:
                time.sleep(self.poll_interval)
                continue

            job_id = job["job_id"]
            file_path = job["file_path"]
            opts_raw = job.get("options_json") or "{}"
            try:
                options = json.loads(opts_raw)
            except Exception:
                options = {}

            logger.info(f"Worker processing job {job_id} ({file_path})")
            self.queue.add_event(
                job_id,
                "status",
                {"job_id": job_id, "status": "processing", "progress": 0.05, "message": "Job picked up by worker"},
            )
            try:
                report_id = f"rep_{job_id}"
                self._execute_job(job_id, report_id, file_path, options)
                self.queue.mark_completed(job_id, report_id)
                self.queue.add_event(
                    job_id,
                    "completed",
                    {
                        "job_id": job_id,
                        "status": "completed",
                        "report_id": report_id,
                        "progress": 1.0,
                        "message": "Analysis successfully completed",
                    },
                )
                logger.info(f"Job {job_id} successfully completed (report: {report_id})")

                # Webhook dispatch
                webhook_url = job.get("webhook_url")
                if webhook_url:
                    self._dispatch_webhook(
                        webhook_url,
                        job.get("webhook_secret"),
                        {
                            "event": "analysis_completed",
                            "job_id": job_id,
                            "report_id": report_id,
                            "status": "completed",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

            except Exception as e:
                logger.exception(f"Job {job_id} failed with error: {e}")
                self.queue.mark_failed(job_id, str(e))
                self.queue.add_event(
                    job_id,
                    "failed",
                    {
                        "job_id": job_id,
                        "status": "failed",
                        "error": str(e),
                        "progress": 1.0,
                        "message": "Analysis failed with an error",
                    },
                )
                webhook_url = job.get("webhook_url")
                if webhook_url:
                    self._dispatch_webhook(
                        webhook_url,
                        job.get("webhook_secret"),
                        {
                            "event": "analysis_failed",
                            "job_id": job_id,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

    def _execute_job(
        self,
        job_id: str,
        report_id: str,
        file_path: str,
        options: Dict[str, Any],
    ):
        """Executes full analysis pipeline and generates JSON, HTML, and PDF reports."""
        self.queue.add_event(
            job_id,
            "progress",
            {"step": "ingest", "progress": 0.15, "message": "Ingesting and loading document"},
        )
        doc = load_document(Path(file_path))

        self.queue.add_event(
            job_id,
            "progress",
            {"step": "similarity", "progress": 0.35, "message": "Executing plagiarism and similarity engine"},
        )
        use_offline = options.get("offline", True)
        sim_res = check_document(
            document=doc,
            use_online=not use_offline,
            use_local=True,
            options={"skip_semantic": options.get("skip_semantic", False)},
        )
        self.queue.add_event(
            job_id,
            "engine_done",
            {"engine": "similarity", "progress": 0.55, "message": "Similarity engine completed"},
        )

        self.queue.add_event(
            job_id,
            "progress",
            {"step": "integrity", "progress": 0.65, "message": "Executing document integrity checks"},
        )
        integ_res = run_all_checks(doc, similarity_result=sim_res)
        self.queue.add_event(
            job_id,
            "engine_done",
            {"engine": "integrity", "progress": 0.75, "message": "Integrity checks completed"},
        )

        self.queue.add_event(
            job_id,
            "progress",
            {"step": "style", "progress": 0.80, "message": "Running rule-based style linter"},
        )
        style_res = lint_style(doc.clean_text)
        self.queue.add_event(
            job_id,
            "engine_done",
            {"engine": "style", "progress": 0.85, "message": "Style linter completed"},
        )

        full_rep = build_report(
            document=doc,
            similarity=sim_res,
            ai=None,
            integrity=integ_res,
            settings=options,
        )
        full_rep.metadata["style"] = style_res.to_dict()

        # Apply cryptographic seal
        full_rep.seal = compute_seal(full_rep.to_dict())

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        # 1. JSON report
        json_path = REPORTS_DIR / f"{report_id}.json"
        json_path.write_text(full_rep.to_json(indent=2), encoding="utf-8")

        # 2. HTML report
        html_path = REPORTS_DIR / f"{report_id}.html"
        export_report_html(full_rep, output_path=html_path)

        # 3. PDF report
        pdf_path = REPORTS_DIR / f"{report_id}.pdf"
        try:
            export_report_pdf(full_rep, output_path=pdf_path)
        except Exception as e:
            logger.warning(f"PDF export warning for job {job_id}: {e}")

    def _dispatch_webhook(
        self,
        url: str,
        secret: Optional[str],
        payload: Dict[str, Any],
    ):
        """Sends signed POST request to configured webhook URL."""
        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            if secret:
                sig = sign_webhook_payload(secret, data_bytes)
                req.add_header("X-Veritext-Signature", sig)

            with urllib.request.urlopen(req, timeout=10.0) as resp:
                logger.info(f"Webhook delivered to {url} (status: {resp.status})")
        except Exception as e:
            logger.warning(f"Webhook notification failed for {url}: {e}")
