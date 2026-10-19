"""Core AI-written text detector combining stylometry, distilgpt2 perplexity, and ensemble scoring."""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from veritext.ai_detect.ensemble import (
    calculate_ensemble_score,
    determine_confidence_band,
    label_sentence,
    load_weights_config,
)
from veritext.ai_detect.features_lm import (
    compute_lm_features,
    is_distilgpt2_available,
)
from veritext.ai_detect.features_style import compute_stylometric_features
from veritext.ai_detect.models import AIResult, SentenceAIResult
from veritext.ai_detect.roberta import classify_roberta, is_roberta_available
from veritext.ingest.models import Document, Sentence
from veritext.ingest.pipeline import load_document

logger = logging.getLogger(__name__)

DEFAULT_WORD_CAP = 5000
MIN_WORD_COUNT = 150


def detect_ai(
    document: Union[Document, str, Path],
    use_roberta: bool = False,
    max_words: int = DEFAULT_WORD_CAP,
    config_path: Optional[Path] = None,
    progress_callback=None,
) -> AIResult:
    """
    Performs AI text detection under Lite Mode constraints:
    - Below 150 words: returns 'insufficient text'.
    - If distilgpt2 is unavailable locally: operates on stylometry alone with reduced_confidence=True.
    - RoBERTa classifier is strictly opt-in via use_roberta=True.
    - Caps analysis at max_words (default 5000 words).
    - Returns JSON-serializable AIResult with exact sentence character offsets.
    """
    # 1. Normalize input to Document
    if isinstance(document, (str, Path)):
        doc_path = Path(document)
        if doc_path.exists() and doc_path.is_file():
            doc = load_document(doc_path)
        else:
            # Direct text string
            raw = str(document)
            from veritext.ingest.splitter import split_paragraphs, split_sentences
            clean = raw.strip()
            paras = split_paragraphs(clean)
            sents = split_sentences(clean)
            doc = Document(
                raw_text=raw,
                clean_text=clean,
                sentences=sents,
                paragraphs=paras,
                removed_sections=[],
                file_type=".txt",
            )
    else:
        doc = document

    clean_text = doc.clean_text.strip()
    words = clean_text.split()
    total_words = len(words)

    # 2. Check Insufficient Text Threshold (< 150 words)
    if total_words < MIN_WORD_COUNT:
        return AIResult(
            status="insufficient_text",
            word_count=total_words,
            overall_probability=None,
            confidence_band="none",
            reduced_confidence=False,
            mode="none",
            sentences=[],
            features={"total_words": float(total_words)},
            explanations=[
                f"Document contains {total_words} words. Statistical AI detection requires at least 150 words "
                "to yield statistically sound distributional features."
            ],
            metadata={"word_cap": max_words},
        )

    # 3. Cap text if exceeding max_words
    if total_words > max_words:
        capped_words = words[:max_words]
        analysis_text = " ".join(capped_words)
        analysis_word_count = max_words
    else:
        analysis_text = clean_text
        analysis_word_count = total_words

    if progress_callback:
        progress_callback("Extracting stylometric features...")

    # 4. Stylometric Features (always available, no neural model needed)
    sent_texts = [s.text for s in doc.sentences if s.text.strip()]
    para_texts = [p.text for p in doc.paragraphs if p.text.strip()]

    style_features = compute_stylometric_features(
        text=analysis_text,
        sentences=sent_texts,
        paragraphs=para_texts,
    )

    # 5. Language Model Features (distilgpt2 if available locally)
    has_lm = is_distilgpt2_available()
    lm_features: Dict[str, Any] = {}
    sentence_perplexities: List[float] = []

    if has_lm:
        if progress_callback:
            progress_callback("Evaluating perplexity and burstiness with distilgpt2...")
        try:
            lm_features = compute_lm_features(
                text=analysis_text,
                sentences=sent_texts,
            )
            sentence_perplexities = lm_features.get("sentence_perplexities", [])
        except Exception as e:
            logger.warning(f"Error computing LM features: {e}. Falling back to stylometry.")
            has_lm = False

    reduced_confidence = not has_lm
    mode = "full" if has_lm else "stylometry_only"

    # 6. Optional RoBERTa Classifier (strictly opt-in)
    roberta_prob: Optional[float] = None
    if use_roberta:
        if is_roberta_available():
            if progress_callback:
                progress_callback("Evaluating RoBERTa classifier...")
            roberta_prob = classify_roberta(analysis_text)
        else:
            logger.info("RoBERTa classifier requested but local weights are not installed.")

    # 7. Weighted Ensemble Combination
    combined_features: Dict[str, float] = {**style_features}
    for k, v in lm_features.items():
        if isinstance(v, (int, float)):
            combined_features[k] = float(v)

    weights_config = load_weights_config(config_path=config_path)
    overall_prob, explanations = calculate_ensemble_score(
        features=combined_features,
        has_lm=has_lm,
        roberta_prob=roberta_prob,
        weights_config=weights_config,
    )

    confidence_band = determine_confidence_band(
        word_count=analysis_word_count,
        probability=overall_prob,
        has_lm=has_lm,
    )

    if reduced_confidence:
        explanations.insert(
            0,
            "Language model unavailable locally: operating in stylometry-only fallback mode (reduced confidence)."
        )

    # 8. Per-Sentence Classification with Document Offsets
    sentence_results: List[SentenceAIResult] = []
    th = weights_config.get("thresholds", {"human_max": 35.0, "uncertain_max": 65.0})

    for idx, s in enumerate(doc.sentences):
        s_words = len(s.text.split())
        if s_words < 4:
            continue

        # Estimate sentence AI score from perplexity and local stylometry
        if idx < len(sentence_perplexities):
            s_ppl = sentence_perplexities[idx]
            # Lower perplexity correlates with higher AI likelihood
            s_score = 1.0 / (1.0 + math.exp((s_ppl - 30.0) / 15.0))
            s_expl = f"Perplexity: {s_ppl:.1f}"
        else:
            s_ppl = None
            # Fallback based on sentence length and discourse markers
            has_marker = any(m in s.text.lower() for m in ["however", "furthermore", "moreover", "in conclusion"])
            s_score = 0.60 if has_marker else (overall_prob / 100.0)
            s_expl = "Stylometric transition marker" if has_marker else "Document baseline"

        s_label = label_sentence(s_score, thresholds=th)

        sentence_results.append(
            SentenceAIResult(
                text=s.text,
                doc_start=s.start,
                doc_end=s.end,
                label=s_label,
                score=s_score,
                perplexity=s_ppl,
                explanation=s_expl,
            )
        )

    # Sort sentence results by score descending (most AI-like first)
    sentence_results.sort(key=lambda item: item.score, reverse=True)

    return AIResult(
        status="success",
        word_count=total_words,
        overall_probability=overall_prob,
        confidence_band=confidence_band,
        reduced_confidence=reduced_confidence,
        mode=mode,
        sentences=sentence_results,
        features=combined_features,
        explanations=explanations,
        metadata={
            "word_cap_applied": total_words > max_words,
            "words_analyzed": analysis_word_count,
            "used_roberta": use_roberta and roberta_prob is not None,
            "roberta_probability": round(roberta_prob, 4) if roberta_prob is not None else None,
        },
    )
