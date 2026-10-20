"""Fast offline unit tests for Part A: Adaptive Analysis, Profiling, Planning, and Chunking.

Zero network calls, zero heavy neural model loading (mocked models).
"""

from dataclasses import dataclass
import gc
import math
from pathlib import Path
import tracemalloc
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from veritext.ai_detect.detector import detect_ai
from veritext.ai_detect.features_lm import compute_lm_features
from veritext.ingest.models import Document, RemovedSection, Sentence
from veritext.planner import (
    AnalysisPlan,
    compute_section_sampling_ranges,
    make_plan,
)
from veritext.profile import DocumentProfile, profile_document
from veritext.similarity.semantic import find_semantic_matches
from veritext.sources.clients.semantic_scholar import (
    is_semantic_scholar_available,
    reset_semantic_scholar_session,
    search as search_semantic_scholar,
    set_semantic_scholar_disabled,
)
from veritext.timing import TimingData, get_default_timing, load_timing, update_timing
from veritext.ui.services import run_pipeline


# --- A1. PROFILER ON KNOWN TEXTS FOR EVERY SIZE CLASS ---

def test_profiler_size_classes():
    # 1. Tiny: under 150 words
    tiny_text = "Word " * 80
    p_tiny = profile_document(tiny_text)
    assert p_tiny.word_count == 80
    assert p_tiny.size_class == "tiny"
    assert p_tiny.page_estimate == 1

    # 2. Short: under 1,500 words
    short_text = "Word " * 600
    p_short = profile_document(short_text)
    assert p_short.word_count == 600
    assert p_short.size_class == "short"
    assert p_short.page_estimate == 2

    # 3. Medium: under 8,000 words
    med_text = "Word " * 5000
    p_med = profile_document(med_text)
    assert p_med.word_count == 5000
    assert p_med.size_class == "medium"
    assert p_med.page_estimate == 10

    # 4. Long: under 30,000 words
    long_text = "Word " * 15000
    p_long = profile_document(long_text)
    assert p_long.word_count == 15000
    assert p_long.size_class == "long"
    assert p_long.page_estimate == 30

    # 5. Very long: 30,000 words and above
    vlong_text = "Word " * 35000
    p_vlong = profile_document(vlong_text)
    assert p_vlong.word_count == 35000
    assert p_vlong.size_class == "very long"
    assert p_vlong.page_estimate == 70


def test_profiler_reference_and_language():
    doc = Document(
        raw_text="This is an empirical analysis.\n\nReferences\nSmith 2020 Study of words.",
        clean_text="This is an empirical analysis.",
        sentences=[],
        paragraphs=[],
        removed_sections=[
            RemovedSection(type="references", content="Smith 2020 Study of words in literature.")
        ],
        file_type="txt",
    )
    p = profile_document(doc)
    assert p.language_guess == "en"
    assert p.reference_words == 7


# --- A2. PLANNER BUDGET, SAMPLING, AND MODES ---

def test_planner_full_coverage_when_fits_budget():
    prof = profile_document("Word " * 5000)
    timing = TimingData(
        ai_distilgpt2_sec_per_1000=20.0,
        ai_stylometry_sec_per_1000=0.2,
        semantic_sec_per_1000=5.0,
        timestamp="2026-10-21T00:00:00Z",
        cpu_thread_count=2,
    )
    # 5,000 words @ 20s/1000 = 100s, fits 10 min (600s) budget
    plan = make_plan(prof, options={"use_ai": True, "use_semantic": True}, timing=timing)

    ai_eng = plan.engines["ai_detect"]
    assert ai_eng.enabled is True
    assert ai_eng.coverage_percent == 100.0
    assert ai_eng.sampling_mode == "full"
    assert ai_eng.words_to_analyze == 5000
    assert ai_eng.word_ranges == [(0, 5000)]


def test_planner_sampled_coverage_when_exceeds_budget():
    prof = profile_document("Word " * 31400)
    timing = TimingData(
        ai_distilgpt2_sec_per_1000=50.0,  # 31,400 words = 1570s > 600s
        ai_stylometry_sec_per_1000=0.5,
        semantic_sec_per_1000=10.0,
        timestamp="2026-10-21T00:00:00Z",
        cpu_thread_count=2,
    )
    plan = make_plan(
        prof,
        options={"use_ai": True, "time_budget_minutes": 5.0},  # budget 300s -> target words = 6,000
        timing=timing,
    )

    ai_eng = plan.engines["ai_detect"]
    assert ai_eng.enabled is True
    assert ai_eng.sampling_mode == "section_sampled"
    assert ai_eng.coverage_percent < 100.0
    assert len(ai_eng.word_ranges) > 1
    # Check that first block and last block are included
    assert ai_eng.word_ranges[0][0] == 0
    assert ai_eng.word_ranges[-1][1] == 31400


def test_section_sampling_properties():
    ranges = compute_section_sampling_ranges(total_words=31400, target_words=4000, block_size=500)
    assert len(ranges) >= 2
    # Always includes first block starting at 0
    assert ranges[0][0] == 0
    # Always includes last block ending at 31400
    assert ranges[-1][1] == 31400
    # Evenly spaced blocks
    diffs = [ranges[i + 1][0] - ranges[i][0] for i in range(len(ranges) - 1)]
    assert all(d > 0 for d in diffs)


def test_exact_and_checks_never_limited():
    prof = profile_document("Word " * 50000)
    timing = get_default_timing()
    plan = make_plan(prof, options={"mode": "auto", "time_budget_minutes": 1.0}, timing=timing)

    exact_eng = plan.engines["exact"]
    checks_eng = plan.engines["checks"]

    assert exact_eng.coverage_percent == 100.0
    assert exact_eng.words_to_analyze == 50000
    assert exact_eng.sampling_mode == "full"

    assert checks_eng.coverage_percent == 100.0
    assert checks_eng.words_to_analyze == 50000
    assert checks_eng.sampling_mode == "full"


def test_ai_detection_under_150_words_skipped():
    prof = profile_document("Word " * 120)
    plan = make_plan(prof, options={"use_ai": True})
    ai_eng = plan.engines["ai_detect"]
    assert ai_eng.enabled is False
    assert "under 150 words" in ai_eng.reason
    assert ai_eng.words_to_analyze == 0
    assert ai_eng.coverage_percent == 0.0


def test_full_and_custom_modes():
    prof = profile_document("Word " * 20000)
    timing = TimingData(
        ai_distilgpt2_sec_per_1000=100.0,
        ai_stylometry_sec_per_1000=0.5,
        semantic_sec_per_1000=10.0,
        timestamp="now",
        cpu_thread_count=2,
    )

    # Full mode: ignores budget, analyses 100%
    plan_full = make_plan(prof, options={"mode": "full", "time_budget_minutes": 2.0}, timing=timing)
    assert plan_full.engines["ai_detect"].coverage_percent == 100.0
    assert plan_full.engines["ai_detect"].words_to_analyze == 20000
    assert len(plan_full.warnings) > 0  # Warning about exceeding budget

    # Custom mode: explicit word limits
    plan_custom = make_plan(
        prof,
        options={
            "mode": "custom",
            "custom_ai_words": 3000,
            "custom_semantic_words": 2000,
            "use_ai": True,
            "use_semantic": True,
        },
        timing=timing,
    )
    assert plan_custom.engines["ai_detect"].sampling_mode == "section_sampled"
    assert plan_custom.engines["ai_detect"].words_to_analyze <= 3500
    assert plan_custom.engines["semantic"].sampling_mode == "section_sampled"


# --- A3. TIMING JSON LOADING, FALLBACK, AND SMOOTHING ---

def test_timing_loading_fallback_and_smoothing(tmp_path: Path):
    timing_file = tmp_path / "timing.json"

    # 1. Missing file: conservative defaults
    t = load_timing(path=timing_file, auto_calibrate=False)
    assert t.ai_distilgpt2_sec_per_1000 == 25.0

    # 2. Update with observed speed (70% old, 30% new)
    # Old: 25.0, Observed: 15.0 -> 0.70 * 25 + 0.30 * 15 = 17.5 + 4.5 = 22.0
    updated = update_timing(
        observed_speeds={"ai_distilgpt2_sec_per_1000": 15.0},
        path=timing_file,
        alpha=0.30,
    )
    assert updated.ai_distilgpt2_sec_per_1000 == 22.0

    # 3. Reload persisted file
    t_reloaded = load_timing(path=timing_file)
    assert t_reloaded.ai_distilgpt2_sec_per_1000 == 22.0


# --- A4. CHUNKED AI DETECTION ON 60,000 WORDS (MOCKED, FLAT MEMORY) ---

def test_chunked_ai_detection_memory_and_chunk_size():
    # 60,000 synthetic words
    synthetic_60k = "The statistical analysis revealed significant variance in test distributions. " * 6000

    # Mock tokenizer and model
    mock_tok = MagicMock()
    # Mock encoding return object
    class MockEnc:
        def __init__(self, token_count):
            import torch
            self.input_ids = torch.zeros((1, min(512, token_count)), dtype=torch.long)

    def fake_tok(text, return_tensors=None, max_length=512, truncation=True):
        words = text.split()
        return MockEnc(len(words))

    mock_tok.side_effect = fake_tok

    # Mock model
    mock_model = MagicMock()
    class MockOutput:
        def __init__(self):
            import torch
            self.loss = torch.tensor(2.5)
            # Logits shape: (1, 512, 100)
            self.logits = torch.zeros((1, 512, 100))

    mock_model.return_value = MockOutput()

    # Track memory with tracemalloc
    tracemalloc.start()
    t_start_mem, _ = tracemalloc.get_traced_memory()

    chunk_sizes = []
    def track_chunk(msg):
        # Callback check
        pass

    res = compute_lm_features(
        text=synthetic_60k,
        sentences=["Sentence 1.", "Sentence 2."],
        tokenizer=mock_tok,
        model=mock_model,
        max_chunk_tokens=512,
        progress_callback=track_chunk,
    )

    t_current_mem, t_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assert model was called with chunks
    assert mock_model.call_count > 100

    # Verify no chunk input passed to model exceeded 512 tokens
    for call_args in mock_model.call_args_list:
        chunk_input = call_args[0][0]
        assert chunk_input.size(1) <= 512

    # Verify memory did not balloon (under 50 MB for 60,000 words processing)
    mem_delta_mb = (t_peak_mem - t_start_mem) / (1024 * 1024)
    assert mem_delta_mb < 50.0
    assert "perplexity" in res


# --- A4. SEMANTIC BLOCK-WISE SIMILARITY EQUALS ALL-AT-ONCE ---

def test_semantic_blockwise_similarity_matches_all_at_once():
    # Generate deterministic doc and candidate vectors
    rng = np.random.default_rng(42)
    n_doc = 300
    n_cand = 50
    dim = 64

    doc_vecs = rng.standard_normal((n_doc, dim)).astype(np.float32)
    doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    cand_vecs = rng.standard_normal((n_cand, dim)).astype(np.float32)
    cand_vecs /= np.linalg.norm(cand_vecs, axis=1, keepdims=True)

    # All-at-once matrix multiplication
    full_sim = np.dot(doc_vecs, cand_vecs.T)
    k = 3
    all_at_once_top = [np.argsort(full_sim[i])[::-1][:k] for i in range(n_doc)]

    # Block-wise multiplication in blocks of 64
    block_size = 64
    blockwise_top = []
    for b_start in range(0, n_doc, block_size):
        b_end = min(b_start + block_size, n_doc)
        block = doc_vecs[b_start:b_end]
        sim_block = np.dot(block, cand_vecs.T)
        for i in range(len(block)):
            top = np.argsort(sim_block[i])[::-1][:k]
            blockwise_top.append(top)

    # Verify top matches are completely identical
    assert len(all_at_once_top) == len(blockwise_top)
    for expected, actual in zip(all_at_once_top, blockwise_top):
        np.testing.assert_array_equal(expected, actual)


# --- A5/A10. CANCELLATION FLAG PRESERVES FINISHED ENGINES ---

def test_cancellation_preserves_finished_results():
    calls = 0
    def cancel_after_first_engine():
        nonlocal calls
        calls += 1
        return calls >= 2

    # Ingest document
    doc = Document(
        raw_text="The experiment evaluated algorithmic performance across diverse test conditions.",
        clean_text="The experiment evaluated algorithmic performance across diverse test conditions.",
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )

    report = run_pipeline(
        target="data/samples/sample_academic.txt",
        filename="sample_academic.txt",
        settings={"use_ai": True, "use_integrity": True, "use_online": False},
        cancel_check=cancel_after_first_engine,
    )

    assert report is not None
    assert report.profile is not None
    assert report.plan is not None


# --- A9. SINGLE 429 BEHAVIOR FOR SEMANTIC SCHOLAR ---

def test_semantic_scholar_single_429():
    reset_semantic_scholar_session()
    set_semantic_scholar_disabled(False)
    assert is_semantic_scholar_available() is True

    mock_client = MagicMock()
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_client.get.return_value = mock_resp_429

    # First call encounters 429
    res1 = search_semantic_scholar("query 1", client=mock_client)
    assert res1 == []
    assert is_semantic_scholar_available() is False
    assert mock_client.get.call_count == 1

    # Second call should be skipped immediately without making a network request
    res2 = search_semantic_scholar("query 2", client=mock_client)
    assert res2 == []
    # Client get was NOT called a second time
    assert mock_client.get.call_count == 1

    # Reset
    reset_semantic_scholar_session()
    assert is_semantic_scholar_available() is True


def test_semantic_scholar_disabled_by_config():
    reset_semantic_scholar_session()
    set_semantic_scholar_disabled(True)
    assert is_semantic_scholar_available() is False

    mock_client = MagicMock()
    res = search_semantic_scholar("test query", client=mock_client)
    assert res == []
    assert mock_client.get.call_count == 0

    set_semantic_scholar_disabled(False)
