"""Tests for batch pairwise collusion matching and heatmap matrix generation."""

from pathlib import Path
from veritext.batch import run_batch_analysis


def test_batch_collusion_five_synthetic_submissions(tmp_path):
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    shared_long_passage = (
        "In this investigation, we develop a distributed winnowing architecture designed to process "
        "arbitrary document representations in linear time complexity. By computing polynomial rolling hashes "
        "over fixed character gram intervals, local minimum selections guarantee bounded window coverage "
        "even under extensive textual permutations and minor editorial modifications. Experimental evaluations "
        "demonstrate consistent retrieval precision without requiring dense vector transformer calculations. "
    )

    # 1 & 2: Two colluding submissions
    sub1_text = (
        "Title: Distributed Indexing Systems.\n\n"
        + shared_long_passage
        + "Furthermore, student one notes that database synchronization remains a vital secondary objective."
    )
    (cohort_dir / "submission_01.txt").write_text(sub1_text, encoding="utf-8")

    sub2_text = (
        "Title: High Performance Winnowing Frameworks.\n\n"
        + shared_long_passage
        + "Student two concludes that throughput scalability exceeds initial baseline estimates by twelve percent."
    )
    (cohort_dir / "submission_02.txt").write_text(sub2_text, encoding="utf-8")

    # 3, 4, 5: Three completely independent submissions
    sub3_text = (
        "Photosynthesis in vascular plants converts light energy into chemical energy within the chloroplasts. "
        "Chlorophyll pigments absorb electromagnetic radiation primarily in the blue and red portions of the visible spectrum. "
        "The Calvin-Benson cycle subsequently fixes inorganic carbon dioxide into triose phosphate precursors."
    )
    (cohort_dir / "submission_03_botany.txt").write_text(sub3_text, encoding="utf-8")

    sub4_text = (
        "The transition from the Roman Republic to the Roman Empire was catalyzed by systemic political instability, "
        "the disintegration of traditional senatorial norms, and the emergence of ambitious military commanders. "
        "Following the assassination of Julius Caesar, the second triumvirate culminated in the rise of Augustus."
    )
    (cohort_dir / "submission_04_history.txt").write_text(sub4_text, encoding="utf-8")

    sub5_text = (
        "Plate tectonics describes the large-scale motion of seven major plates and numerous minor plates across the lithosphere. "
        "At convergent boundaries, subduction zones generate deep oceanic trenches and active volcanic arcs. "
        "Transform faults exhibit lateral horizontal displacement, frequently generating severe shallow-focus seismic disturbances."
    )
    (cohort_dir / "submission_05_geology.txt").write_text(sub5_text, encoding="utf-8")

    # Run batch pairwise analysis
    result = run_batch_analysis(cohort_dir, threshold=0.30, anonymize=False)

    assert len(result.submissions) == 5
    # Matrix dimensions 5x5
    assert len(result.matrix) == 5
    for row in result.matrix:
        assert len(row) == 5

    # Diagonal is 100%
    for i in range(5):
        assert result.matrix[i][i] == 100.0

    # Index 0 (submission_01) and Index 1 (submission_02) must have high similarity (>= 30%)
    sim_01_02 = result.matrix[0][1]
    assert sim_01_02 >= 30.0

    # Submissions 03, 04, 05 must have low similarity (< 15%) to each other and to 01/02
    assert result.matrix[0][2] < 15.0
    assert result.matrix[1][3] < 15.0
    assert result.matrix[2][3] < 15.0
    assert result.matrix[3][4] < 15.0

    # Collusion pairs
    assert len(result.pairs_above_threshold) == 1
    pair = result.pairs_above_threshold[0]
    assert "submission_01" in pair.doc_a_name or "submission_01" in pair.doc_b_name
    assert "submission_02" in pair.doc_a_name or "submission_02" in pair.doc_b_name
    assert len(pair.shared_passages) >= 1

    # Exactly 1 cluster containing the two colluding submissions
    assert len(result.clusters) == 1
    assert len(result.clusters[0]) == 2

    # CSV and HTML checks
    assert "submission_01.txt" in result.csv_content
    assert "submission_02.txt" in result.csv_content
    assert "<table class=\"heatmap\">" in result.html_report
    # Self-contained check: No external http/https CDN script or link tags
    assert "http://" not in result.html_report
    assert "https://" not in result.html_report
