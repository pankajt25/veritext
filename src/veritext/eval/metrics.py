"""Pure-arithmetic metrics calculation for evaluation benchmarks.

Zero heavy dependencies (pure Python standard library).
"""

from typing import Sequence


def binary_classification_metrics(
    y_true: Sequence[bool | int],
    y_pred: Sequence[bool | int],
) -> dict[str, float | int]:
    """Computes binary classification metrics: accuracy, precision, recall, F1, and FPR.

    Args:
        y_true: Ground truth binary labels (True/1 = positive, False/0 = negative).
        y_pred: Predicted binary labels (True/1 = positive, False/0 = negative).

    Returns:
        dict containing tp, fp, tn, fn, total, accuracy, precision, recall, f1, fpr.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if bool(yt) and bool(yp))
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not bool(yt) and bool(yp))
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not bool(yt) and not bool(yp))
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if bool(yt) and not bool(yp))
    total = len(y_true)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


def mean(values: Sequence[float]) -> float:
    """Computes arithmetic mean of a sequence of numbers."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
