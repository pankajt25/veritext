"""Data models for review workflow, annotations, and rubric assessment."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Set


class ReviewStatus:
    ACCEPTED = "accepted"            # Genuine integrity/attribution issue
    IGNORED = "ignored"              # False positive or sanctioned text
    NEEDS_DISCUSSION = "needs_discussion"  # Flagged for conversation with student

    ALL = {ACCEPTED, IGNORED, NEEDS_DISCUSSION}

    @classmethod
    def normalize(cls, val: str) -> str:
        v = (val or "").strip().lower().replace(" ", "_")
        if v in cls.ALL:
            return v
        if "accept" in v:
            return cls.ACCEPTED
        if "ignore" in v or "false_positive" in v:
            return cls.IGNORED
        if "discuss" in v:
            return cls.NEEDS_DISCUSSION
        return cls.NEEDS_DISCUSSION


@dataclass
class FindingReview:
    """Review decision and inline annotation for a finding or match span."""

    target_id: str
    status: str = ReviewStatus.NEEDS_DISCUSSION
    comment: str = ""
    author: str = "Instructor"
    target_type: str = "finding"  # 'finding' or 'span'
    start: int = 0
    end: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "status": self.status,
            "comment": self.comment,
            "author": self.author,
            "target_type": self.target_type,
            "start": self.start,
            "end": self.end,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FindingReview":
        return cls(
            target_id=d.get("target_id", ""),
            status=ReviewStatus.normalize(d.get("status", ReviewStatus.NEEDS_DISCUSSION)),
            comment=d.get("comment", ""),
            author=d.get("author", "Instructor"),
            target_type=d.get("target_type", "finding"),
            start=int(d.get("start", 0)),
            end=int(d.get("end", 0)),
            created_at=d.get("created_at") or datetime.now(timezone.utc).isoformat(),
            updated_at=d.get("updated_at") or datetime.now(timezone.utc).isoformat(),
        )


@dataclass
class RubricLevel:
    score: int
    label: str
    descriptor: str

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "label": self.label, "descriptor": self.descriptor}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RubricLevel":
        return cls(
            score=int(d.get("score", 0)),
            label=d.get("label", ""),
            descriptor=d.get("descriptor", ""),
        )


@dataclass
class RubricCriterion:
    id: str
    name: str
    description: str
    max_score: int = 4
    levels: List[RubricLevel] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "max_score": self.max_score,
            "levels": [lvl.to_dict() for lvl in self.levels],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RubricCriterion":
        levels = [RubricLevel.from_dict(lvl) for lvl in d.get("levels", [])]
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            max_score=int(d.get("max_score", 4)),
            levels=levels,
        )


@dataclass
class RubricScore:
    criterion_id: str
    score: int
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "score": self.score,
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RubricScore":
        return cls(
            criterion_id=d.get("criterion_id", ""),
            score=int(d.get("score", 0)),
            comment=d.get("comment", ""),
        )


@dataclass
class RubricAssessment:
    name: str = "Standard Academic Integrity & Writing Rubric"
    scores: Dict[str, RubricScore] = field(default_factory=dict)
    overall_comment: str = ""

    @property
    def total_score(self) -> int:
        return sum(s.score for s in self.scores.values())

    @property
    def max_possible(self) -> int:
        # Default 4 points per criterion evaluated, or minimum 4 * count
        return max(1, len(self.scores) * 4)

    @property
    def percentage(self) -> float:
        if self.max_possible == 0:
            return 0.0
        return round((self.total_score / self.max_possible) * 100.0, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_score": self.total_score,
            "max_possible": self.max_possible,
            "percentage": self.percentage,
            "overall_comment": self.overall_comment,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RubricAssessment":
        scores_raw = d.get("scores", {})
        scores = {k: RubricScore.from_dict(v) for k, v in scores_raw.items()}
        return cls(
            name=d.get("name", "Standard Academic Integrity & Writing Rubric"),
            scores=scores,
            overall_comment=d.get("overall_comment", ""),
        )


@dataclass
class ReviewSession:
    """Full review session associated with a report ID."""

    report_id: str
    document_name: str = ""
    author: str = "Instructor"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reviews: Dict[str, FindingReview] = field(default_factory=dict)
    rubric: Optional[RubricAssessment] = None
    general_notes: str = ""

    def set_review(
        self,
        target_id: str,
        status: str,
        comment: str = "",
        author: Optional[str] = None,
        target_type: str = "finding",
        start: int = 0,
        end: int = 0,
    ) -> FindingReview:
        norm_status = ReviewStatus.normalize(status)
        now_str = datetime.now(timezone.utc).isoformat()
        rev = FindingReview(
            target_id=target_id,
            status=norm_status,
            comment=comment,
            author=author or self.author,
            target_type=target_type,
            start=start,
            end=end,
            updated_at=now_str,
        )
        if target_id in self.reviews:
            rev.created_at = self.reviews[target_id].created_at
        self.reviews[target_id] = rev
        self.updated_at = now_str
        return rev

    def get_review(self, target_id: str) -> Optional[FindingReview]:
        return self.reviews.get(target_id)

    def is_ignored(self, target_id: str) -> bool:
        rev = self.reviews.get(target_id)
        return rev is not None and rev.status == ReviewStatus.IGNORED

    def is_accepted(self, target_id: str) -> bool:
        rev = self.reviews.get(target_id)
        return rev is not None and rev.status == ReviewStatus.ACCEPTED

    def get_ignored_target_ids(self) -> Set[str]:
        return {
            tid for tid, rev in self.reviews.items()
            if rev.status == ReviewStatus.IGNORED
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "document_name": self.document_name,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reviews": {k: v.to_dict() for k, v in self.reviews.items()},
            "rubric": self.rubric.to_dict() if self.rubric else None,
            "general_notes": self.general_notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReviewSession":
        reviews_raw = d.get("reviews", {})
        reviews = {k: FindingReview.from_dict(v) for k, v in reviews_raw.items()}
        rubric_raw = d.get("rubric")
        rubric = RubricAssessment.from_dict(rubric_raw) if rubric_raw else None
        return cls(
            report_id=d.get("report_id", ""),
            document_name=d.get("document_name", ""),
            author=d.get("author", "Instructor"),
            created_at=d.get("created_at") or datetime.now(timezone.utc).isoformat(),
            updated_at=d.get("updated_at") or datetime.now(timezone.utc).isoformat(),
            reviews=reviews,
            rubric=rubric,
            general_notes=d.get("general_notes", ""),
        )
