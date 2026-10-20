"""Weighted logistic ensemble combining stylometric and language model signals."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from veritext.ai_detect.features_style import DISCOURSE_MARKERS

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "config"
    / "ai_weights.json"
)

# Default conservative weights (trained to prioritize low false-positive rate on human text)
DEFAULT_WEIGHTS = {
    "version": "1.0",
    "description": "Default conservative logistic regression weights for AI text detection",
    "thresholds": {
        "human_max": 35.0,
        "uncertain_max": 65.0,
    },
    "full": {
        "intercept": 0.5,
        "coefficients": {
            "perplexity": -0.04,          # Lower perplexity -> higher AI probability
            "burstiness": -0.06,          # Lower burstiness -> higher AI probability
            "mean_log_rank": -0.35,       # Lower rank (higher predictability) -> higher AI
            "sentence_length_variance": -0.008,  # Uniform sentence lengths -> higher AI
            "discourse_marker_density": 0.45,    # More formulaic transitions -> higher AI
            "repeated_ngram_rate": 8.0,          # Repetitive n-grams -> higher AI
            "type_token_ratio": -2.0,            # Lower vocabulary richness -> higher AI
        },
    },
    "stylometry_only": {
        "intercept": -0.8,
        "coefficients": {
            "sentence_length_variance": -0.012,
            "discourse_marker_density": 0.60,
            "repeated_ngram_rate": 10.0,
            "type_token_ratio": -2.5,
            "paragraph_length_variance": -0.002,
        },
    },
}


def load_weights_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads weights config from JSON file, falling back to defaults if missing."""
    target_path = config_path or DEFAULT_CONFIG_PATH
    if target_path.exists():
        try:
            return json.loads(target_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_WEIGHTS


def save_weights_config(weights: Dict[str, Any], config_path: Optional[Path] = None):
    """Saves weights config dictionary to JSON file."""
    target_path = config_path or DEFAULT_CONFIG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")


def calculate_ensemble_score(
    features: Dict[str, float],
    has_lm: bool = True,
    roberta_prob: Optional[float] = None,
    weights_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[str]]:
    """
    Computes overall AI probability (0.0 to 100.0) and generates plain-language driver explanations.
    """
    config = weights_config or load_weights_config()
    mode_key = "full" if has_lm else "stylometry_only"
    mode_weights = config.get(mode_key, DEFAULT_WEIGHTS[mode_key])

    intercept = mode_weights.get("intercept", 0.0)
    coeffs = mode_weights.get("coefficients", {})

    z = intercept
    explanations: List[str] = []

    for feat_name, weight in coeffs.items():
        val = features.get(feat_name)
        if val is not None:
            contrib = weight * val
            z += contrib

            # Record human-readable drivers for notable features
            if feat_name == "discourse_marker_density" and val >= 1.2:
                explanations.append(
                    f"Elevated frequency of formulaic transition phrases ({val:.1f} per 100 words)"
                )
            elif feat_name == "sentence_length_variance" and val < 40.0:
                explanations.append(
                    f"Unusually uniform sentence length cadence (variance {val:.1f})"
                )
            elif feat_name == "sentence_length_variance" and val > 120.0:
                explanations.append(
                    f"Highly varied, natural sentence rhythm (variance {val:.1f})"
                )
            elif feat_name == "burstiness" and val < 15.0:
                explanations.append(
                    f"Low perplexity variation across sentences (burstiness {val:.1f})"
                )
            elif feat_name == "burstiness" and val >= 30.0:
                explanations.append(
                    f"Dynamic, bursty predictability variation across sentences ({val:.1f})"
                )
            elif feat_name == "perplexity" and val < 25.0:
                explanations.append(
                    f"Very low language model perplexity ({val:.1f}), indicating high predictability"
                )
            elif feat_name == "type_token_ratio" and val > 0.65:
                explanations.append(
                    f"Rich, diverse vocabulary usage (Type-Token Ratio {val:.2f})"
                )

    # Blend RoBERTa classifier probability if available
    if roberta_prob is not None:
        # Logistic adjustment incorporating RoBERTa probability
        roberta_logit = math.log(max(1e-5, min(0.99999, roberta_prob)) / (1 - max(1e-5, min(0.99999, roberta_prob))))
        z = 0.6 * z + 0.4 * roberta_logit
        explanations.append(f"RoBERTa classifier output: {roberta_prob * 100:.1f}% AI probability")

    # Standard logistic sigmoid
    try:
        prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
    except OverflowError:
        prob = 1.0 if z > 0 else 0.0

    percentage = prob * 100.0

    if not explanations:
        if percentage >= 50.0:
            explanations.append("Balanced combination of predictable syntax and formulaic phrasing")
        else:
            explanations.append("Natural stylistic variation, rich vocabulary, and irregular sentence lengths")

    return percentage, explanations


def determine_confidence_band(
    word_count: int,
    probability: float,
    has_lm: bool,
) -> str:
    """
    Determines confidence band: 'low', 'medium', or 'high'.
    """
    if word_count < 150:
        return "none"

    if not has_lm:
        # Stylometry alone has capped confidence
        if word_count >= 500 and (probability >= 80.0 or probability <= 20.0):
            return "medium"
        return "low"

    # Full mode
    if word_count < 300:
        return "low" if (40.0 <= probability <= 60.0) else "medium"

    # > 300 words with language model
    if probability >= 75.0 or probability <= 25.0:
        return "high"
    elif probability >= 65.0 or probability <= 35.0:
        return "medium"
    return "low"


def label_sentence(
    score: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Classifies a sentence score into 'likely human', 'uncertain', or 'likely AI'."""
    th = thresholds or DEFAULT_WEIGHTS["thresholds"]
    h_max = th.get("human_max", 35.0)
    u_max = th.get("uncertain_max", 65.0)

    if score * 100.0 < h_max:
        return "likely human"
    elif score * 100.0 <= u_max:
        return "uncertain"
    return "likely AI"


def score_sentence(
    sentence_text: str,
    overall_probability: float,
    sentence_perplexity: Optional[float] = None,
    mean_perplexity: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[float, str, str]:
    """
    Computes a calibrated AI probability score (0.0 to 1.0), classification label,
    and explanation for an individual sentence. Calibrated on the exact same scale
    as the document-level ensemble to maintain consistency between document-level
    classifications and sentence-level rankings.
    """
    th = thresholds or DEFAULT_WEIGHTS["thresholds"]
    h_max = th.get("human_max", 35.0)
    u_max = th.get("uncertain_max", 65.0)

    # Document base probability in (0.01, 0.99)
    p_base = max(0.01, min(0.99, overall_probability / 100.0))
    z = math.log(p_base / (1.0 - p_base))

    # 1. Stylometric transition phrase adjustment
    has_marker = any(m in sentence_text.lower() for m in DISCOURSE_MARKERS)
    if has_marker:
        z += 0.55
    else:
        z -= 0.10

    # 2. Sentence length cadence adjustment
    words = sentence_text.split()
    w_count = len(words)
    if 14 <= w_count <= 28:
        z += 0.15
    elif w_count > 38 or w_count < 8:
        z -= 0.15

    # 3. Relative perplexity adjustment (when distilgpt2 perplexity is available)
    if sentence_perplexity is not None and sentence_perplexity > 0 and mean_perplexity and mean_perplexity > 0:
        delta_loss = math.log(sentence_perplexity) - math.log(mean_perplexity)
        # Clamped adjustment: lower perplexity than average increases AI probability
        ppl_adj = -0.45 * max(-2.0, min(2.0, delta_loss))
        z += ppl_adj

        if has_marker:
            explanation = f"Perplexity: {sentence_perplexity:.1f}, transition marker"
        else:
            explanation = f"Perplexity: {sentence_perplexity:.1f}"
    else:
        if has_marker:
            explanation = "Stylometric transition marker"
        else:
            explanation = "Document baseline stylometry"

    # Convert logit back to probability
    raw_score = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
    score = round(raw_score, 4)

    # Consistency guarantee:
    # If the document is classified as "likely AI" (overall_probability >= u_max),
    # ensure top AI hallmark sentences reflect this classification and are not labeled "likely human".
    if overall_probability >= u_max:
        min_ai_bound = u_max / 100.0
        if has_marker and score < min_ai_bound:
            score = round(min_ai_bound + 0.05, 4)
        elif score < (h_max / 100.0):
            score = round(h_max / 100.0 + 0.05, 4)

    label = label_sentence(score, thresholds=th)
    return score, label, explanation
