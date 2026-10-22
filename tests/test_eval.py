"""Unit and integration tests for the evaluation and benchmarking suite."""

import csv
from pathlib import Path
import pytest

from veritext.eval.dataset import (
    filter_manifest_by_group,
    load_document,
    load_manifest,
    load_references,
)
from veritext.eval.metrics import binary_classification_metrics, mean
from veritext.eval.runner import BenchmarkRunner


def test_binary_classification_metrics_perfect():
    """Verifies precision, recall, f1, and fpr on perfect classifications."""
    y_true = [True, True, False, False]
    y_pred = [True, True, False, False]

    m = binary_classification_metrics(y_true, y_pred)
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["tn"] == 2
    assert m["fn"] == 0
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["fpr"] == 0.0


def test_binary_classification_metrics_mixed_math():
    """Verifies precision, recall, f1, and fpr mathematics on known mixed inputs."""
    # 4 positives, 4 negatives.
    # Predictions: 3 true positives, 1 false negative, 1 false positive, 3 true negatives.
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0]

    m = binary_classification_metrics(y_true, y_pred)
    assert m["tp"] == 3
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 3
    assert m["total"] == 8
    assert m["accuracy"] == 0.75  # (3 + 3) / 8
    assert m["precision"] == 0.75  # 3 / (3 + 1)
    assert m["recall"] == 0.75  # 3 / (3 + 1)
    assert m["f1"] == 0.75
    assert m["fpr"] == 0.25  # 1 / (1 + 3)


def test_binary_classification_metrics_edge_cases():
    """Verifies zero-division resilience and length mismatch validation."""
    # Zero positives predicted
    m = binary_classification_metrics([0, 0, 0], [0, 0, 0])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["fpr"] == 0.0
    assert m["accuracy"] == 1.0

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        binary_classification_metrics([1, 0], [1])


def test_mean_calculation():
    """Verifies arithmetic mean helper."""
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([0.15, 0.25]) == 0.2
    assert mean([]) == 0.0


def test_load_manifest_and_filter():
    """Verifies evaluation manifest loading and cohort filtering without any models."""
    manifest = load_manifest()
    assert len(manifest) >= 90

    # Verify manifest columns
    first = manifest[0]
    for col in [
        "file",
        "group",
        "expected_plagiarism",
        "expected_ai",
        "expected_spinner",
        "expected_collusion_group",
    ]:
        assert col in first

    # Test filtering
    copies = filter_manifest_by_group(manifest, "a_copies")
    assert len(copies) == 15
    for c in copies:
        assert c["group"] == "a_copies"

    # Multi-group filtering
    subset = filter_manifest_by_group(manifest, ["c_original", "e_non_native"])
    assert len(subset) == 25


def test_load_references_structure():
    """Verifies loading of evaluation reference passages."""
    refs = load_references()
    assert len(refs) >= 15
    assert "ref_01.txt" in refs
    assert len(refs["ref_01.txt"].split()) >= 80


@pytest.mark.slow
def test_benchmark_runner_subset_e2e(tmp_path):
    """End-to-end integration test running benchmark on a fast subset of documents."""
    # Create a mini manifest with 8 documents across key groups
    full_manifest = load_manifest()
    subset_files = [
        "copy_01.txt",
        "copy_09.txt",
        "para_01.txt",
        "orig_01.txt",
        "ai_01.txt",
        "non_native_01.txt",
        "spinner_01.txt",
        "spinner_ctrl_01.txt",
    ]
    mini_rows = [r for r in full_manifest if r["file"] in subset_files]
    assert len(mini_rows) == 8

    mini_manifest_path = tmp_path / "mini_manifest.csv"
    with open(mini_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mini_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mini_rows)

    out_dir = tmp_path / "results"
    runner = BenchmarkRunner(
        manifest_path=mini_manifest_path,
        out_dir=out_dir,
        semantic=False,
        ai=False,
        crosslang=False,
    )
    summary = runner.run()

    assert summary["total_documents_manifest"] == 8
    assert (out_dir / "benchmark_summary.json").exists()
    assert (out_dir / "benchmark_details.csv").exists()
    assert (out_dir / "results_table.md").exists()
    assert (out_dir / "confusion_matrices.png").exists()
