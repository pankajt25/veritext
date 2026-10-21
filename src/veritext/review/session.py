"""Storage and persistence for review sessions and rubric templates."""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from veritext.review.models import ReviewSession, RubricCriterion

logger = logging.getLogger(__name__)

DEFAULT_REVIEWS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "reviews"
)

DEFAULT_RUBRIC_CONFIG = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "config"
    / "rubric_default.json"
)


def sanitize_report_id(report_id: str) -> str:
    """Sanitizes report ID to prevent path traversal or unsafe characters."""
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", report_id)
    return clean or "default_report"


def get_review_path(
    report_id: str, storage_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Returns the JSON file path for a review session."""
    base_dir = Path(storage_dir) if storage_dir else DEFAULT_REVIEWS_DIR
    clean_id = sanitize_report_id(report_id)
    return base_dir / f"{clean_id}.json"


def save_review_session(
    session: ReviewSession, storage_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Saves a review session to data/reviews/<report_id>.json."""
    path = get_review_path(session.report_id, storage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = session.to_dict()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved review session to {path}")
    return path


def load_review_session(
    report_id: str, storage_dir: Optional[Union[str, Path]] = None
) -> Optional[ReviewSession]:
    """Loads a review session from data/reviews/<report_id>.json if it exists."""
    path = get_review_path(report_id, storage_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ReviewSession.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to load review session from {path}: {e}")
        return None


def load_rubric_template(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Loads the rubric criteria template from JSON configuration."""
    path = Path(config_path) if config_path else DEFAULT_RUBRIC_CONFIG
    if not path.is_file():
        logger.warning(f"Rubric config not found at {path}, returning fallback rubric")
        return {
            "name": "Standard Academic Integrity & Writing Rubric",
            "criteria": [],
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to read rubric template from {path}: {e}")
        return {
            "name": "Standard Academic Integrity & Writing Rubric",
            "criteria": [],
        }


def get_default_criteria(
    config_path: Optional[Union[str, Path]] = None,
) -> List[RubricCriterion]:
    """Parses and returns list of RubricCriterion objects from template."""
    raw = load_rubric_template(config_path)
    criteria_list = []
    for item in raw.get("criteria", []):
        criteria_list.append(RubricCriterion.from_dict(item))
    return criteria_list
