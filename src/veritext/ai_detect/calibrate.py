"""Calibration script fitting ensemble weights on labeled datasets using scikit-learn."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from veritext.ai_detect.ensemble import DEFAULT_WEIGHTS, save_weights_config
from veritext.ai_detect.features_lm import compute_lm_features, is_distilgpt2_available
from veritext.ai_detect.features_style import compute_stylometric_features

logger = logging.getLogger(__name__)

STYLE_FEATURE_NAMES = [
    "sentence_length_variance",
    "discourse_marker_density",
    "repeated_ngram_rate",
    "type_token_ratio",
    "paragraph_length_variance",
]

FULL_FEATURE_NAMES = [
    "perplexity",
    "burstiness",
    "mean_log_rank",
    "sentence_length_variance",
    "discourse_marker_density",
    "repeated_ngram_rate",
    "type_token_ratio",
]


def load_labeled_csv(csv_path: Path) -> Tuple[List[str], List[int]]:
    """Loads text and binary labels (0=human, 1=ai) from CSV."""
    texts: List[str] = []
    labels: List[int] = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            raw_label = str(row.get("label", "")).strip().lower()

            if not text:
                continue

            if raw_label in ("1", "ai", "ai-generated", "true"):
                lbl = 1
            elif raw_label in ("0", "human", "human-written", "false"):
                lbl = 0
            else:
                continue

            texts.append(text)
            labels.append(lbl)

    return texts, labels


def calibrate_ensemble(
    csv_path: Path,
    out_config_path: Optional[Path] = None,
    use_lm: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Fits logistic regression models on the dataset and exports calibrated weights JSON.
    """
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    texts, y_list = load_labeled_csv(path)
    if len(texts) < 6:
        raise ValueError(f"Dataset too small for calibration: {len(texts)} samples found (need >= 6)")

    y = np.array(y_list, dtype=np.int32)
    has_lm = is_distilgpt2_available() if use_lm is None else use_lm

    # 1. Extract Stylometric Features
    X_style_rows = []
    for t in texts:
        feats = compute_stylometric_features(t)
        X_style_rows.append([feats.get(fn, 0.0) for fn in STYLE_FEATURE_NAMES])
    X_style = np.array(X_style_rows, dtype=np.float32)

    # Fit Stylometry-Only Model
    clf_style = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_style.fit(X_style, y)
    preds_style_prob = clf_style.predict_proba(X_style)[:, 1]

    # Evaluate Stylometry Metrics (at conservative 0.50 cutoff)
    preds_style = (preds_style_prob >= 0.50).astype(int)
    cm_style = confusion_matrix(y, preds_style).tolist()
    tn_s, fp_s = cm_style[0][0], cm_style[0][1]
    fpr_style = fp_s / max(1, tn_s + fp_s)

    style_weights = {
        "intercept": float(clf_style.intercept_[0]),
        "coefficients": {
            fn: float(clf_style.coef_[0][idx])
            for idx, fn in enumerate(STYLE_FEATURE_NAMES)
        },
    }

    # 2. Extract Language Model Features (if available)
    full_weights = DEFAULT_WEIGHTS["full"]
    metrics_full = None

    if has_lm:
        try:
            X_full_rows = []
            for t in texts:
                s_feats = compute_stylometric_features(t)
                lm_feats = compute_lm_features(t)
                combined = {**s_feats, **lm_feats}
                X_full_rows.append([combined.get(fn, 0.0) for fn in FULL_FEATURE_NAMES])
            X_full = np.array(X_full_rows, dtype=np.float32)

            clf_full = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            clf_full.fit(X_full, y)
            preds_full_prob = clf_full.predict_proba(X_full)[:, 1]
            preds_full = (preds_full_prob >= 0.50).astype(int)

            cm_full = confusion_matrix(y, preds_full).tolist()
            tn_f, fp_f = cm_full[0][0], cm_full[0][1]
            fpr_full = fp_f / max(1, tn_f + fp_f)

            full_weights = {
                "intercept": float(clf_full.intercept_[0]),
                "coefficients": {
                    fn: float(clf_full.coef_[0][idx])
                    for idx, fn in enumerate(FULL_FEATURE_NAMES)
                },
            }

            metrics_full = {
                "accuracy": float(accuracy_score(y, preds_full)),
                "precision": float(precision_score(y, preds_full, zero_division=0)),
                "recall": float(recall_score(y, preds_full, zero_division=0)),
                "f1": float(f1_score(y, preds_full, zero_division=0)),
                "false_positive_rate": float(fpr_full),
                "confusion_matrix": cm_full,
            }
        except Exception as e:
            logger.warning(f"Could not fit full LM model during calibration: {e}")

    # Build Output Config
    calibrated_config = {
        "version": "1.1-calibrated",
        "description": f"Calibrated on {path.name} with {len(texts)} samples",
        "thresholds": {
            "human_max": 35.0,
            "uncertain_max": 65.0,
        },
        "full": full_weights,
        "stylometry_only": style_weights,
    }

    save_weights_config(calibrated_config, config_path=out_config_path)

    report = {
        "samples_count": len(texts),
        "class_distribution": {"human": int((y == 0).sum()), "ai": int((y == 1).sum())},
        "stylometry_model": {
            "accuracy": float(accuracy_score(y, preds_style)),
            "precision": float(precision_score(y, preds_style, zero_division=0)),
            "recall": float(recall_score(y, preds_style, zero_division=0)),
            "f1": float(f1_score(y, preds_style, zero_division=0)),
            "false_positive_rate": float(fpr_style),
            "confusion_matrix": cm_style,
        },
        "full_model": metrics_full,
        "saved_to": str(out_config_path or DEFAULT_WEIGHTS),
    }

    return report
