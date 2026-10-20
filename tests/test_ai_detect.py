"""Tests for AI-written text detection engine, stylometry, ensemble, calibration, and models."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from veritext.ai_detect.calibrate import calibrate_ensemble
from veritext.ai_detect.detector import MIN_WORD_COUNT, detect_ai
from veritext.ai_detect.ensemble import (
    calculate_ensemble_score,
    determine_confidence_band,
    label_sentence,
    load_weights_config,
    score_sentence,
)
from veritext.ai_detect.features_lm import compute_lm_features
from veritext.ai_detect.features_style import (
    compute_stylometric_features,
    extract_word_tokens,
)
from veritext.ai_detect.models import AIResult, SentenceAIResult
from veritext.ingest.models import Document, Paragraph, Sentence


def test_stylometric_features_math():
    text = (
        "Furthermore, artificial intelligence algorithms process distributed data. "
        "Consequently, modern neural networks achieve remarkable benchmark accuracy. "
        "However, researchers must investigate data privacy."
    )
    feats = compute_stylometric_features(text)

    assert feats["total_words"] > 0
    assert feats["sentence_length_mean"] > 0
    assert 0.0 <= feats["type_token_ratio"] <= 1.0
    assert 0.0 <= feats["hapax_ratio"] <= 1.0
    # Text contains "furthermore", "consequently", "however"
    assert feats["discourse_marker_density"] > 0.0

    # Empty text handling
    empty_feats = compute_stylometric_features("")
    assert empty_feats["total_words"] == 0.0
    assert empty_feats["sentence_length_mean"] == 0.0


def test_lm_features_mocked():
    """Verifies perplexity, log-rank, and burstiness mathematics with a mocked tokenizer and model."""
    import torch

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Synthetic input IDs: 8 tokens
    seq_len = 8
    dummy_input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    mock_tokenizer.return_value = MagicMock(input_ids=dummy_input_ids)

    # Synthetic model outputs: cross-entropy loss = 2.0 (ppl = exp(2.0) ~ 7.39)
    # Logits: shape (1, 8, 100)
    dummy_logits = torch.zeros(1, 8, 100)
    # Make target token have rank 2
    dummy_logits[0, :, 10] = 5.0
    dummy_logits[0, :, 20] = 4.0
    mock_output = MagicMock(loss=torch.tensor(2.0), logits=dummy_logits)
    mock_model.return_value = mock_output

    res = compute_lm_features(
        text="Sample test text for mocked language model feature extraction.",
        sentences=["Sentence one text.", "Sentence two text."],
        tokenizer=mock_tokenizer,
        model=mock_model,
    )

    assert "perplexity" in res
    assert "mean_log_rank" in res
    assert "burstiness" in res
    assert res["perplexity"] > 0.0
    assert isinstance(res["sentence_perplexities"], list)


def test_ensemble_weighting_and_logistic():
    features = {
        "perplexity": 15.0,  # Very low perplexity (AI-like)
        "burstiness": 5.0,    # Very low burstiness (AI-like)
        "mean_log_rank": 2.0,
        "sentence_length_variance": 10.0,
        "discourse_marker_density": 3.5,  # High discourse markers
        "repeated_ngram_rate": 0.15,
        "type_token_ratio": 0.60,
    }

    score, explanations = calculate_ensemble_score(features, has_lm=True)
    assert 0.0 <= score <= 100.0
    # Because all features are heavily AI-biased, score should be high
    assert score > 55.0
    assert len(explanations) >= 1

    # Human-like features
    human_features = {
        "perplexity": 95.0,
        "burstiness": 45.0,
        "mean_log_rank": 6.5,
        "sentence_length_variance": 150.0,
        "discourse_marker_density": 0.0,
        "repeated_ngram_rate": 0.0,
        "type_token_ratio": 0.85,
    }
    human_score, human_expl = calculate_ensemble_score(human_features, has_lm=True)
    assert human_score < 40.0


def test_confidence_bands_and_sentence_labels():
    # Below 150 words -> none
    assert determine_confidence_band(100, 80.0, has_lm=True) == "none"

    # Full mode >= 300 words with strong score -> high
    assert determine_confidence_band(400, 85.0, has_lm=True) == "high"
    assert determine_confidence_band(400, 15.0, has_lm=True) == "high"
    assert determine_confidence_band(400, 50.0, has_lm=True) == "low"

    # Stylometry only mode -> capped
    assert determine_confidence_band(600, 85.0, has_lm=False) == "medium"
    assert determine_confidence_band(200, 85.0, has_lm=False) == "low"

    # Sentence labeling
    assert label_sentence(0.20) == "likely human"
    assert label_sentence(0.50) == "uncertain"
    assert label_sentence(0.85) == "likely AI"


def test_insufficient_text_threshold():
    short_text = "This is a short note with far too few words to detect AI statistically."
    result = detect_ai(short_text)

    assert result.status == "insufficient_text"
    assert result.overall_probability is None
    assert result.confidence_band == "none"
    assert result.word_count < MIN_WORD_COUNT
    assert "at least 150 words" in result.explanations[0]


def test_stylometry_only_fallback_and_disclaimer():
    # Build text > 150 words
    words = ["word"] * 180
    long_text = " ".join(words)

    # Force is_distilgpt2_available to False
    with patch("veritext.ai_detect.detector.is_distilgpt2_available", return_value=False):
        result = detect_ai(long_text)

        assert result.status == "success"
        assert result.reduced_confidence is True
        assert result.mode == "stylometry_only"
        assert result.overall_probability is not None
        assert "Disclaimer" in result.disclaimer


def test_sentence_offset_mapping():
    s1_text = "The first sentence establishes the theoretical framework."
    s2_text = "The second sentence discusses empirical observations and unexpected variations."
    s3_text = "In conclusion, the third sentence summarizes the primary conclusions reached."
    full_text = f"{s1_text} {s2_text} {s3_text}"

    s1_start, s1_end = 0, len(s1_text)
    s2_start = s1_end + 1
    s2_end = s2_start + len(s2_text)
    s3_start = s2_end + 1
    s3_end = s3_start + len(s3_text)

    # Pad with filler words so total words > 150
    filler = " " + " ".join(["context"] * 150)
    full_text_padded = full_text + filler

    doc = Document(
        raw_text=full_text_padded,
        clean_text=full_text_padded,
        sentences=[
            Sentence(s1_text, s1_start, s1_end),
            Sentence(s2_text, s2_start, s2_end),
            Sentence(s3_text, s3_start, s3_end),
        ],
        paragraphs=[Paragraph(full_text_padded, 0, len(full_text_padded))],
        removed_sections=[],
        file_type=".txt",
    )

    with patch("veritext.ai_detect.detector.is_distilgpt2_available", return_value=False):
        result = detect_ai(doc)

    assert len(result.sentences) >= 3
    # Check that offsets map back accurately to clean_text
    for s_res in result.sentences:
        original_slice = doc.clean_text[s_res.doc_start : s_res.doc_end]
        assert original_slice == s_res.text


def test_json_serialization():
    res = AIResult(
        status="success",
        word_count=250,
        overall_probability=82.4,
        confidence_band="high",
        reduced_confidence=False,
        mode="full",
        sentences=[
            SentenceAIResult(
                text="A sample sentence.",
                doc_start=0,
                doc_end=18,
                label="likely AI",
                score=0.85,
                perplexity=14.2,
                explanation="Low perplexity",
            )
        ],
        features={"perplexity": 18.5, "burstiness": 6.2},
        explanations=["Low perplexity indicative of AI generation"],
    )

    json_str = res.to_json()
    parsed = json.loads(json_str)

    assert parsed["status"] == "success"
    assert parsed["overall_probability"] == 82.4
    assert parsed["confidence_band"] == "high"
    assert len(parsed["sentences"]) == 1
    assert parsed["sentences"][0]["label"] == "likely AI"
    assert "disclaimer" in parsed


def test_calibration_script_produces_metrics(tmp_path):
    starter_csv = Path(__file__).resolve().parent.parent / "data" / "calibration" / "starter.csv"
    assert starter_csv.exists()

    out_weights = tmp_path / "test_weights.json"
    report = calibrate_ensemble(starter_csv, out_config_path=out_weights, use_lm=False)

    assert report["samples_count"] == 30
    assert "accuracy" in report["stylometry_model"]
    assert "f1" in report["stylometry_model"]
    assert "confusion_matrix" in report["stylometry_model"]
    assert out_weights.exists()

    saved_data = json.loads(out_weights.read_text(encoding="utf-8"))
    assert "stylometry_only" in saved_data
    assert "intercept" in saved_data["stylometry_only"]


def test_ai_sentences_descending_sort_order():
    """Verifies that detected AI sentences are sorted strictly descending by score."""
    sample_text = (
        "Document Title On Innovation\n\n"
        "In the aftermath of technological acceleration, distributed teams have altered workflows. "
        "Modern systems facilitate virtual coordination across boundaries with reduced latency. "
        "Furthermore, automated frameworks offer unprecedented flexibility for workforce planning. "
        "Consequently, forward-thinking enterprises report elevated workforce satisfaction levels. "
        "Additionally, asynchronous messaging enables collaboration without temporal bottlenecks. "
        "In conclusion, modern operating dynamics require intentional human leadership strategies."
    )
    padded = sample_text + " " + " ".join(["operational context analysis"] * 50)

    with patch("veritext.ai_detect.detector.is_distilgpt2_available", return_value=True), \
         patch("veritext.ai_detect.detector.compute_lm_features") as mock_lm:
        mock_lm.return_value = {
            "perplexity": 35.0,
            "mean_log_rank": 3.0,
            "burstiness": 15.0,
            "sentence_perplexities": [120.0, 95.0, 45.0, 40.0, 38.0, 50.0],
        }
        res = detect_ai(padded)

    assert res.status == "success"
    assert len(res.sentences) >= 5
    for i in range(len(res.sentences) - 1):
        assert res.sentences[i].score >= res.sentences[i + 1].score, (
            f"Sentence {i} ({res.sentences[i].score}) should be >= Sentence {i+1} ({res.sentences[i+1].score})"
        )


def test_ai_sentences_exact_offsets_and_headings_skipped():
    """Verifies exact offset slicing text[start:end] == sentence and headings are not scored."""
    heading = "The Architecture of Neural Optimization"
    s1 = "Deep neural networks optimize parameters using stochastic gradient descent."
    s2 = "Furthermore, backpropagation calculates exact gradients across composite layers."
    s3 = "Consequently, empirical loss curves exhibit smooth convergence characteristics."
    s4 = "In conclusion, modern architectures achieve state-of-the-art benchmark results."
    s5 = "However, researchers must continue investigating interpretability mechanisms."

    full_text = f"{heading}\n\n{s1} {s2}\n\n{s3} {s4} {s5}"
    padded = full_text + " " + " ".join(["contextual empirical analysis"] * 60)

    with patch("veritext.ai_detect.detector.is_distilgpt2_available", return_value=False):
        res = detect_ai(padded)

    assert res.status == "success"
    scored_texts = [s.text for s in res.sentences]
    assert heading not in scored_texts

    for sent in res.sentences:
        sliced = padded[sent.doc_start : sent.doc_end]
        assert sliced == sent.text, f"Offset mismatch: expected '{sent.text}', got '{sliced}'"
        assert "\n\n" not in sent.text, "Sentence should not bridge across paragraphs"


def test_document_sentence_label_consistency():
    """Verifies that high-AI documents produce top sentences that are not all 'likely human'."""
    th = {"human_max": 35.0, "uncertain_max": 65.0}

    s1_score, s1_label, _ = score_sentence(
        "Furthermore, remote work frameworks offer employees unprecedented flexibility.",
        overall_probability=78.2,
        sentence_perplexity=85.0,
        mean_perplexity=120.0,
        thresholds=th,
    )
    s2_score, s2_label, _ = score_sentence(
        "Modern telecommunications platforms facilitate seamless virtual interactions.",
        overall_probability=78.2,
        sentence_perplexity=130.0,
        mean_perplexity=120.0,
        thresholds=th,
    )

    assert s1_score >= 0.65, f"Expected AI score >= 0.65, got {s1_score}"
    assert s1_label == "likely AI"
    assert s2_label in ("uncertain", "likely AI")

    h_score, h_label, _ = score_sentence(
        "We initiated the reduction trial by dissolving 2.45 grams in ethanol.",
        overall_probability=6.4,
        sentence_perplexity=250.0,
        mean_perplexity=180.0,
        thresholds=th,
    )
    assert h_score < 0.35, f"Expected human score < 0.35, got {h_score}"
    assert h_label == "likely human"


def test_detect_ai_consistency_end_to_end_mocked():
    """End-to-end check that detect_ai produces consistent labels with mocked AI outputs."""
    sample_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "ai_demo" / "ai_sample_1.txt"
    from veritext.ingest.pipeline import load_document
    doc = load_document(sample_path)

    with patch("veritext.ai_detect.detector.is_distilgpt2_available", return_value=True), \
         patch("veritext.ai_detect.detector.compute_lm_features") as mock_lm:
        mock_lm.return_value = {
            "perplexity": 20.0,
            "mean_log_rank": 2.2,
            "burstiness": 6.0,
            "sentence_perplexities": [35.0, 32.0, 30.0, 33.0, 28.0, 31.0, 29.0, 34.0],
        }
        res = detect_ai(doc)

    assert res.status == "success"
    assert res.overall_probability >= 65.0, f"Overall prob {res.overall_probability} should be >= 65"
    top_5 = res.sentences[:5]
    labels = [s.label for s in top_5]
    assert all(label in ("likely AI", "uncertain") for label in labels)
    assert top_5[0].label == "likely AI"
    assert top_5[0].score >= 0.65


@pytest.mark.slow
def test_real_distilgpt2_demo_detection():
    """
    Slow integration test using real distilgpt2 on demo samples.
    Verifies that AI samples receive significantly higher AI probability than human samples.
    """
    from veritext.ai_detect.features_lm import download_distilgpt2, is_distilgpt2_available

    if not is_distilgpt2_available():
        download_distilgpt2()

    demo_dir = Path(__file__).resolve().parent.parent / "data" / "samples" / "ai_demo"
    assert demo_dir.exists()

    human_res_1 = detect_ai(demo_dir / "human_sample_1.txt")
    human_res_2 = detect_ai(demo_dir / "human_sample_2.txt")
    human_res_esl = detect_ai(demo_dir / "human_sample_3_esl.txt")

    ai_res_1 = detect_ai(demo_dir / "ai_sample_1.txt")
    ai_res_2 = detect_ai(demo_dir / "ai_sample_2.txt")
    ai_res_3 = detect_ai(demo_dir / "ai_sample_3.txt")

    avg_human = (human_res_1.overall_probability + human_res_2.overall_probability + human_res_esl.overall_probability) / 3.0
    avg_ai = (ai_res_1.overall_probability + ai_res_2.overall_probability + ai_res_3.overall_probability) / 3.0

    assert avg_ai > avg_human + 20.0, f"Expected AI avg ({avg_ai:.1f}) to be > human avg ({avg_human:.1f}) + 20"

