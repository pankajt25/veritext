"""Veritext AI-written text detection engine (perplexity, burstiness, stylometry, ensemble)."""

from veritext.ai_detect.detector import detect_ai
from veritext.ai_detect.ensemble import calculate_ensemble_score, load_weights_config, score_sentence
from veritext.ai_detect.features_lm import compute_lm_features, is_distilgpt2_available
from veritext.ai_detect.features_style import compute_stylometric_features
from veritext.ai_detect.models import AIResult, SentenceAIResult

__all__ = [
    "detect_ai",
    "AIResult",
    "SentenceAIResult",
    "compute_stylometric_features",
    "compute_lm_features",
    "calculate_ensemble_score",
    "load_weights_config",
    "score_sentence",
    "is_distilgpt2_available",
]
