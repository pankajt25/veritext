"""FastAPI REST application for Veritext local engine."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import secrets
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from veritext.api.auth import check_rate_limit, hash_key, verify_api_key
from veritext.api.queue import JobQueue, SingleWorker
from veritext.review.models import ReviewSession
from veritext.review.session import load_review_session, save_review_session

logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "uploads"
)
REPORTS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "reports"
)
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB limit
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".tex", ".rtf", ".odt", ".html"}

queue = JobQueue()
worker = SingleWorker(queue)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Starts background worker on app launch and halts cleanly on exit."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    worker.start()
    yield
    worker.stop()


app = FastAPI(
    title="Veritext Academic Integrity API",
    description="Local REST API for multi-signal plagiarism, AI text detection, and integrity checks.",
    version="1.0.1",
    lifespan=lifespan,
)


def sanitize_filename(filename: str) -> str:
    """Removes path traversal components and normalizes filename across OS platforms."""
    normalized = filename.replace("\\", "/")
    base = os.path.basename(normalized)
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
    clean = clean.lstrip(".")
    return clean or "document.txt"


def get_current_user(x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Dependency verifying API key and enforcing rate limits."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required 'X-API-Key' header.",
        )
    user = verify_api_key(x_api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )
    khash = hash_key(x_api_key)
    if not check_rate_limit(khash):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded (maximum 60 requests per minute).",
        )
    return user


@app.get("/health")
def health_check():
    """Service health and background worker status."""
    return {
        "status": "ok",
        "version": "1.0.1",
        "worker_running": worker.is_alive(),
        "jobs_in_queue": queue.count_queued(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/submit", status_code=status.HTTP_202_ACCEPTED)
async def submit_document(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    offline: bool = Form(False),
    budget_minutes: float = Form(5.0),
    webhook_url: Optional[str] = Form(None),
    webhook_secret: Optional[str] = Form(None),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Submits a document file for asynchronous analysis."""
    safe_name = sanitize_filename(file.filename or "upload.txt")
    ext = os.path.splitext(safe_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}",
        )

    # Read and check size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum limit of {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    job_id = secrets.token_hex(16)
    dest_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    dest_path.write_bytes(content)

    options = {
        "mode": mode,
        "offline": offline,
        "budget_minutes": budget_minutes,
    }

    job_info = queue.enqueue_job(
        job_id=job_id,
        file_path=str(dest_path),
        original_filename=safe_name,
        options=options,
        webhook_url=webhook_url,
        webhook_secret=webhook_secret,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "original_filename": safe_name,
        "created_at": job_info["created_at"],
        "status_url": f"/jobs/{job_id}",
    }


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Checks the progress and completion status of a submitted job."""
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "", job_id)
    job = queue.get_job(clean_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    res = {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "original_filename": job["original_filename"],
    }
    if job["status"] == "completed":
        rep_id = job.get("report_id")
        res["report_id"] = rep_id
        res["reports"] = {
            "json": f"/reports/{rep_id}.json",
            "html": f"/reports/{rep_id}.html",
            "pdf": f"/reports/{rep_id}.pdf",
        }
    elif job["status"] == "failed":
        res["error"] = job.get("error_message")

    return res


@app.get("/jobs/{job_id}/events")
async def get_job_events(
    job_id: str,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Server-Sent Events (SSE) stream reporting real-time job progress and completion."""
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "", job_id)
    job = queue.get_job(clean_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    async def event_generator():
        last_id = 0
        terminal = False
        while not terminal:
            if await request.is_disconnected():
                break

            events = queue.get_events(clean_id, after_id=last_id)
            for ev in events:
                last_id = ev["id"]
                ev_type = ev["event_type"]
                payload = json.dumps(ev["data"])
                yield f"event: {ev_type}\ndata: {payload}\n\n"

                if ev_type in ("completed", "failed"):
                    terminal = True
                    break

            if terminal:
                break

            cur_job = queue.get_job(clean_id)
            if cur_job and cur_job["status"] in ("completed", "failed") and not events:
                if cur_job["status"] == "completed":
                    rep_id = cur_job.get("report_id")
                    yield f"event: completed\ndata: {json.dumps({'job_id': clean_id, 'status': 'completed', 'report_id': rep_id, 'progress': 1.0})}\n\n"
                else:
                    yield f"event: failed\ndata: {json.dumps({'job_id': clean_id, 'status': 'failed', 'error': cur_job.get('error_message')})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/reports/{report_id}.json")
def get_report_json(report_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Downloads full structured JSON analysis report."""
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "", report_id)
    p = REPORTS_DIR / f"{clean_id}.json"
    if not p.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report '{clean_id}.json' not found.")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error reading report: {e}")


@app.get("/reports/{report_id}.html", response_class=HTMLResponse)
def get_report_html(report_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Downloads standalone interactive HTML report."""
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "", report_id)
    p = REPORTS_DIR / f"{clean_id}.html"
    if not p.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report '{clean_id}.html' not found.")
    return HTMLResponse(content=p.read_text(encoding="utf-8"))


@app.get("/reports/{report_id}.pdf")
def get_report_pdf(report_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Downloads printable PDF report."""
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "", report_id)
    p = REPORTS_DIR / f"{clean_id}.pdf"
    if not p.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report '{clean_id}.pdf' not found.")
    return FileResponse(
        str(p),
        media_type="application/pdf",
        filename=f"{clean_id}.pdf",
    )


@app.get("/reviews/{report_id}")
def get_review_endpoint(report_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves existing review annotations and rubric scoring for a report."""
    session = load_review_session(report_id)
    if not session:
        return {"report_id": report_id, "reviews": {}, "rubric": None, "general_notes": ""}
    return session.to_dict()


@app.post("/reviews/{report_id}")
def save_review_endpoint(
    report_id: str,
    payload: Dict[str, Any],
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Saves or updates review annotations and rubric scores for a report."""
    payload["report_id"] = report_id
    session = ReviewSession.from_dict(payload)
    path = save_review_session(session)
    return {"status": "saved", "path": str(path.name), "review": session.to_dict()}
