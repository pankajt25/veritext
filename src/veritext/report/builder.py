"""Report builder consolidating outputs from similarity, AI detection, and integrity checks."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from veritext.ai_detect.models import AIResult
from veritext.checks.models import IntegrityReport
from veritext.ingest.models import Document
from veritext.ingest.pipeline import load_document
from veritext.report.models import DEFAULT_DISCLAIMER, FullReport, HighlightSpan
from veritext.similarity.models import SimilarityResult


def build_report(
    document: Union[Document, str, Path],
    similarity: Optional[SimilarityResult] = None,
    ai: Optional[AIResult] = None,
    integrity: Optional[IntegrityReport] = None,
    settings: Optional[Dict[str, Any]] = None,
    profile: Optional[Any] = None,
    plan: Optional[Any] = None,
) -> FullReport:
    """Merges outputs from all analysis engines into a unified FullReport.

    Args:
        document: Document instance or file path/text string.
        similarity: Optional Stage 3 SimilarityResult.
        ai: Optional Stage 4 AIDetectionResult.
        integrity: Optional Stage 5 IntegrityReport.
        settings: Analysis configuration options used.
        profile: Optional DocumentProfile instance or dict.
        plan: Optional AnalysisPlan instance or dict.

    Returns:
        FullReport containing unified highlight spans and summary metrics.
    """
    doc_path: Optional[str] = None
    if isinstance(document, (str, Path)):
        p = Path(document)
        if p.exists() and p.is_file():
            doc = load_document(p)
            doc_path = str(p.resolve())
            doc_name = p.name
        else:
            doc = Document(
                raw_text=str(document),
                clean_text=str(document),
                sentences=[],
                paragraphs=[],
                removed_sections=[],
                file_type="txt",
                path=None,
            )
            doc_name = "Text Document"
    else:
        doc = document
        doc_path = str(doc.path) if doc.path else None
        doc_name = Path(doc.path).name if doc.path else "Document"

    orig_text = doc.raw_text or ""
    clean_text = doc.clean_text or ""
    span_items: List[HighlightSpan] = []
    span_counter = 1

    # 1. Similarity highlight spans
    if similarity and similarity.sources:
        for src in similarity.sources:
            for sp in src.spans:
                sp_type = getattr(sp, "type", "exact")
                cat = "exact match" if sp_type in ("exact", "near_copy") else "paraphrase"
                sev = "high" if sp_type == "exact" else "warning"

                m_text = getattr(sp, "matched_text", "")
                d_start = getattr(sp, "doc_start", 0)
                d_end = getattr(sp, "doc_end", 0)

                # Re-align start/end into original raw_text if possible
                if m_text and m_text in orig_text:
                    pos = orig_text.find(m_text, max(0, d_start - 100))
                    if pos == -1:
                        pos = orig_text.find(m_text)
                    if pos != -1:
                        d_start = pos
                        d_end = pos + len(m_text)

                span_items.append(
                    HighlightSpan(
                        id=f"span_{span_counter}",
                        start=d_start,
                        end=d_end,
                        category=cat,
                        severity=sev,
                        message=f"Matched with source '{src.source_title}' ({sp.score:.0%} match confidence)",
                        source=src.source_title,
                        details={
                            "engine": "similarity",
                            "source_id": src.source_id,
                            "source_url": src.source_url,
                            "score": sp.score,
                            "match_type": sp_type,
                        },
                    )
                )
                span_counter += 1

    # 2. AI Detection highlight spans (sentences flagged as likely AI)
    if ai and ai.sentences:
        for sent in ai.sentences:
            s_label_upper = sent.label.upper()
            if "AI" in s_label_upper or sent.score >= 0.65:
                sev = "high" if sent.score >= 0.85 else "warning"
                s_start = getattr(sent, "doc_start", getattr(sent, "start", 0))
                s_end = getattr(sent, "doc_end", getattr(sent, "end", 0))

                # Re-align sentence bounds to original text
                if sent.text and sent.text in orig_text:
                    pos = orig_text.find(sent.text, max(0, s_start - 50))
                    if pos == -1:
                        pos = orig_text.find(sent.text)
                    if pos != -1:
                        s_start = pos
                        s_end = pos + len(sent.text)

                span_items.append(
                    HighlightSpan(
                        id=f"span_{span_counter}",
                        start=s_start,
                        end=s_end,
                        category="likely AI",
                        severity=sev,
                        message=f"Likely AI-generated sentence (AI score: {sent.score:.1%}, {sent.label})",
                        source="Stylometric + LM Ensemble",
                        details={
                            "engine": "ai_detect",
                            "sentence_score": sent.score,
                            "label": sent.label,
                        },
                    )
                )
                span_counter += 1

    # 3. Integrity checks highlight spans
    if integrity and integrity.results:
        for cr in integrity.results:
            for f in cr.findings:
                if f.check == "hidden_content":
                    cat = "hidden content"
                elif f.check == "citations":
                    cat = "citation issue"
                elif f.check == "self_plagiarism":
                    cat = "self-overlap"
                else:
                    cat = f.check.replace("_", " ")

                src_name = f.details.get("source_title") or f.details.get("source") or cr.check_name
                span_items.append(
                    HighlightSpan(
                        id=f"span_{span_counter}",
                        start=f.start,
                        end=f.end,
                        category=cat,
                        severity=f.severity,
                        message=f.message,
                        source=src_name,
                        details={
                            "engine": "checks",
                            "check_name": f.check,
                            "extra": f.details,
                        },
                    )
                )
                span_counter += 1

    # Sort all highlight spans by start position
    span_items.sort(key=lambda s: (s.start, s.end))

    # Construct high-level summary
    summary_parts = []
    if similarity:
        sim_pct = similarity.similarity_percentage
        src_cnt = len(similarity.sources)
        summary_parts.append(f"Plagiarism: {sim_pct:.1f}% similarity across {src_cnt} source(s)")

    if ai:
        ai_prob = getattr(ai, "overall_probability", getattr(ai, "ai_probability", None))
        ai_pct_str = f"{ai_prob:.1f}%" if ai_prob is not None else "N/A"
        ai_conf = getattr(ai, "confidence_band", getattr(ai, "confidence", "none")).upper()
        summary_parts.append(f"AI Probability: {ai_pct_str} ({ai_conf} confidence)")

    if integrity:
        high_cnt = sum(sum(1 for f in r.findings if f.severity == "high") for r in integrity.results)
        warn_cnt = sum(sum(1 for f in r.findings if f.severity == "warning") for r in integrity.results)
        info_cnt = sum(sum(1 for f in r.findings if f.severity == "info") for r in integrity.results)
        summary_parts.append(f"Integrity: {high_cnt} high, {warn_cnt} warning, {info_cnt} info finding(s)")

    summary_str = " | ".join(summary_parts) if summary_parts else "No analyses executed."

    prof_dict = profile.to_dict() if hasattr(profile, "to_dict") else profile
    plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else plan

    return FullReport(
        document_name=doc_name,
        document_path=doc_path,
        original_text=orig_text,
        clean_text=clean_text,
        settings=settings or {},
        similarity=similarity,
        ai_detect=ai,
        integrity=integrity,
        highlights=span_items,
        profile=prof_dict,
        plan=plan_dict,
        summary=summary_str,
        disclaimer=DEFAULT_DISCLAIMER,
        metadata={
            "word_count": len(orig_text.split()),
            "character_count": len(orig_text),
            "total_highlights": len(span_items),
        },
    )
