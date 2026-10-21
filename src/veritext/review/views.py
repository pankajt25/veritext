"""View transformations for Student View vs Instructor View.

Student View:
- Hides ranking, adversarial, and punitive language.
- Replaces harsh terminology with constructive developmental guidance
  (e.g., 'Add a citation here', 'Rewrite in your own words', 'Clarify terminology').
- Neither view makes accusations or moral judgments.

Instructor View:
- Presents complete diagnostic signals, engine scores, and technical details.
"""

import copy
import re
from typing import Any, Dict, List, Optional

from veritext.review.models import ReviewSession, ReviewStatus


# Terminology replacement map for Student View to maintain an encouraging,
# educational, and non-accusatory tone.
STUDENT_TERM_REPLACEMENTS = [
    (re.compile(r"\bplagiarism\b", re.IGNORECASE), "unattributed matching text"),
    (re.compile(r"\bplagiarised\b", re.IGNORECASE), "unattributed"),
    (re.compile(r"\bplagiarized\b", re.IGNORECASE), "unattributed"),
    (re.compile(r"\bcollusion\b", re.IGNORECASE), "collaborative text overlap"),
    (re.compile(r"\bcolluded\b", re.IGNORECASE), "shared content"),
    (re.compile(r"\bcheating\b", re.IGNORECASE), "source similarity"),
    (re.compile(r"\btortured\s+phrase(s)?\b", re.IGNORECASE), r"unconventional phrasing\1"),
    (re.compile(r"\bspinner\b", re.IGNORECASE), "automated synonym substitution"),
    (re.compile(r"\bspinbot\b", re.IGNORECASE), "automated paraphrasing"),
    (re.compile(r"\badversarial(\s+attack)?\b", re.IGNORECASE), "formatting anomaly"),
    (re.compile(r"\bfake\s+citation(s)?\b", re.IGNORECASE), r"unverified citation\1"),
    (re.compile(r"\bhallucinated\s+citation(s)?\b", re.IGNORECASE), r"unmatched citation\1"),
    (re.compile(r"\bsuspicious\b", re.IGNORECASE), "notable"),
]


def sanitize_text_for_student(text: str) -> str:
    """Replaces adversarial and punitive terminology with constructive phrasing."""
    if not text:
        return ""
    result = text
    for pattern, replacement in STUDENT_TERM_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


def get_student_guidance_for_finding(check_type: str, message: str) -> str:
    """Generates constructive, actionable student guidance based on finding category."""
    ctype = (check_type or "").lower()
    msg = (message or "").lower()

    if "citation" in ctype or "reference" in msg:
        return "Guidance: Verify this citation against the original source to ensure author, year, and title details are accurate."
    elif "spinner" in ctype or "tortured" in msg or "paraphras" in msg:
        return "Guidance: Clarify this sentence by using conventional domain terms and natural academic phrasing."
    elif "hidden" in ctype or "font" in msg or "color" in msg:
        return "Guidance: Review document typography and formatting to ensure text styles and spacing are clean and uniform."
    elif "collusion" in ctype or "class" in msg:
        return "Guidance: Significant text overlap with another submission was noted. Ensure your arguments and drafting are completely your own."
    elif "style" in ctype or "passive" in msg:
        return "Guidance: Consider revising into active voice or breaking into concise sentences to improve clarity."
    elif "repetition" in msg or "repeated" in msg:
        return "Guidance: Remove repeated adjacent words to keep your prose fluent."
    else:
        return "Guidance: Review this passage carefully. Ensure external concepts are cited and expressed in your own voice."


def get_student_guidance_for_span(match_type: str) -> str:
    """Generates constructive guidance for similarity match spans."""
    mtype = (match_type or "").lower()
    if mtype == "exact":
        return "Guidance: This text matches an existing source word-for-word. Place it in quotation marks and cite the author, or rewrite it in your own words."
    elif mtype == "near_copy":
        return "Guidance: This passage is very close to an external source. Paraphrase more thoroughly and add an in-text citation."
    elif mtype == "paraphrase":
        return "Guidance: Ensure you have credited the original author for these ideas with a proper citation."
    return "Guidance: Review this passage and ensure proper attribution is provided."


def format_finding_for_view(
    finding_dict: Dict[str, Any],
    view_mode: str = "instructor",
    review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Formats a single finding dictionary for either Instructor or Student view."""
    res = dict(finding_dict)
    is_student = (view_mode.lower() == "student")

    if is_student:
        original_msg = res.get("message", "")
        res["message"] = sanitize_text_for_student(original_msg)
        res["guidance"] = get_student_guidance_for_finding(
            res.get("check", ""), original_msg
        )
        # Severity in student view is framed as guidance priority rather than punitive risk
        raw_sev = (res.get("severity") or "info").lower()
        if raw_sev in ("high", "warning"):
            res["display_severity"] = "Action Recommended"
        else:
            res["display_severity"] = "Writing Suggestion"
    else:
        res["display_severity"] = res.get("severity", "info").capitalize()
        res["guidance"] = None

    if review:
        res["review_status"] = review.get("status", ReviewStatus.NEEDS_DISCUSSION)
        res["review_comment"] = review.get("comment", "")
        res["reviewer"] = review.get("author", "Instructor")
    else:
        res["review_status"] = None
        res["review_comment"] = None
        res["reviewer"] = None

    return res


def format_span_for_view(
    span_dict: Dict[str, Any],
    view_mode: str = "instructor",
    review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Formats a single match span dictionary for either Instructor or Student view."""
    res = dict(span_dict)
    is_student = (view_mode.lower() == "student")

    if is_student:
        res["guidance"] = get_student_guidance_for_span(res.get("type", ""))
        # In student view, refer to origin politely
        res["source_display"] = res.get("source_title") or "External Reference"
    else:
        res["guidance"] = None
        res["source_display"] = res.get("source_title") or res.get("source_id", "Unknown")

    if review:
        res["review_status"] = review.get("status", ReviewStatus.NEEDS_DISCUSSION)
        res["review_comment"] = review.get("comment", "")
        res["reviewer"] = review.get("author", "Instructor")
    else:
        res["review_status"] = None
        res["review_comment"] = None
        res["reviewer"] = None

    return res


def format_report_for_view(
    report_data: Dict[str, Any],
    view_mode: str = "instructor",
    session: Optional[ReviewSession] = None,
) -> Dict[str, Any]:
    """Generates an adjusted view of a complete report for Instructor or Student.

    - Neither view makes moral accusations.
    - Student view hides ranking metrics and provides developmental guidance.
    - Injected review states and rubric assessment are attached.
    """
    view_mode_norm = "student" if view_mode.lower() == "student" else "instructor"
    out = copy.deepcopy(report_data)
    out["view_mode"] = view_mode_norm

    reviews_map = session.reviews if session else {}
    ignored_keys = session.get_ignored_target_ids() if session else set()

    # Process similarity findings / spans
    sim_data = out.get("similarity", {})
    spans = sim_data.get("spans", [])
    formatted_spans = []
    for sp in spans:
        target_id = sp.get("metadata", {}).get("id") or f"span_{sp.get('doc_start')}_{sp.get('doc_end')}"
        rev = reviews_map.get(target_id)
        formatted_spans.append(
            format_span_for_view(
                sp, view_mode=view_mode_norm, review=rev.to_dict() if rev else None
            )
        )
    sim_data["spans"] = formatted_spans

    # Process integrity findings
    integrity_data = out.get("integrity", {})
    findings = integrity_data.get("findings", [])
    formatted_findings = []
    for f in findings:
        f_id = f.get("id") or f"{f.get('check', 'finding')}_{f.get('start', 0)}_{f.get('end', 0)}"
        rev = reviews_map.get(f_id)
        formatted_findings.append(
            format_finding_for_view(
                f, view_mode=view_mode_norm, review=rev.to_dict() if rev else None
            )
        )
    integrity_data["findings"] = formatted_findings

    # Process style findings if present
    style_data = out.get("style", {})
    if style_data and "findings" in style_data:
        formatted_style = []
        for sf in style_data["findings"]:
            sf_id = sf.get("id") or f"style_{sf.get('start', 0)}_{sf.get('end', 0)}"
            rev = reviews_map.get(sf_id)
            formatted_style.append(
                format_finding_for_view(
                    sf, view_mode=view_mode_norm, review=rev.to_dict() if rev else None
                )
            )
        style_data["findings"] = formatted_style

    # Attach Review Session & Rubric
    if session:
        out["review_session"] = session.to_dict()
        if session.rubric:
            out["rubric_assessment"] = session.rubric.to_dict()
    else:
        out["review_session"] = None
        out["rubric_assessment"] = None

    if view_mode_norm == "student":
        # Neutralize any header summaries in student view
        if "summary" in out:
            out["summary"] = sanitize_text_for_student(str(out["summary"]))
        # Neutralize risk ratings
        if "overall_risk" in out:
            r = str(out.get("overall_risk", "")).lower()
            if r in ("high", "critical"):
                out["overall_risk"] = "Review and Revision Recommended"
            elif r in ("moderate", "medium"):
                out["overall_risk"] = "Minor Revisions Suggested"
            else:
                out["overall_risk"] = "Clear"

    return out
