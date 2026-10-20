"""Hardware timing calibration and adaptive speed measurement.

Measures seconds per 1,000 words on the current machine for:
- AI detection (stylometry only)
- AI detection with distilgpt2
- Semantic vector similarity matching
Persists calibration into data/config/timing.json and smooths values after each real run.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "config"
TIMING_FILE = CONFIG_DIR / "timing.json"

# Conservative baseline defaults for dual-core CPU without GPU
DEFAULT_AI_DISTILGPT2_SEC = 25.0
DEFAULT_AI_STYLOMETRY_SEC = 0.4
DEFAULT_SEMANTIC_SEC = 5.0


@dataclass
class TimingData:
    """Measured execution speeds per 1,000 words."""

    ai_distilgpt2_sec_per_1000: float
    ai_stylometry_sec_per_1000: float
    semantic_sec_per_1000: float
    timestamp: str
    cpu_thread_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimingData":
        return cls(
            ai_distilgpt2_sec_per_1000=float(data.get("ai_distilgpt2_sec_per_1000", DEFAULT_AI_DISTILGPT2_SEC)),
            ai_stylometry_sec_per_1000=float(data.get("ai_stylometry_sec_per_1000", DEFAULT_AI_STYLOMETRY_SEC)),
            semantic_sec_per_1000=float(data.get("semantic_sec_per_1000", DEFAULT_SEMANTIC_SEC)),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            cpu_thread_count=int(data.get("cpu_thread_count", os.cpu_count() or 2)),
        )


def get_default_timing() -> TimingData:
    """Returns conservative default timing when no file or calibration exists."""
    return TimingData(
        ai_distilgpt2_sec_per_1000=DEFAULT_AI_DISTILGPT2_SEC,
        ai_stylometry_sec_per_1000=DEFAULT_AI_STYLOMETRY_SEC,
        semantic_sec_per_1000=DEFAULT_SEMANTIC_SEC,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cpu_thread_count=os.cpu_count() or 2,
    )


def load_timing(path: Optional[Path] = None, auto_calibrate: bool = False) -> TimingData:
    """Loads timing.json, falls back to conservative defaults or auto-mini-calibration if missing."""
    target_path = path or TIMING_FILE
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TimingData.from_dict(data)
        except Exception as e:
            logger.warning(f"Error reading {target_path}: {e}. Using defaults.")

    if auto_calibrate:
        print("⚡ No timing calibration found. Running fast 20s calibration for this machine...")
        try:
            return calibrate(sample_words=300, verbose=False, target_path=target_path)
        except Exception as e:
            logger.warning(f"Auto-calibration failed: {e}. Falling back to defaults.")

    return get_default_timing()


def save_timing(timing: TimingData, path: Optional[Path] = None) -> None:
    """Saves timing metrics to timing.json."""
    target_path = path or TIMING_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(timing.to_dict(), f, indent=2)


def update_timing(
    observed_speeds: Dict[str, float],
    path: Optional[Path] = None,
    alpha: float = 0.30,
) -> TimingData:
    """Smooths timing calibration using an exponential moving average (e.g. 70% old, 30% new)."""
    current = load_timing(path=path, auto_calibrate=False)

    new_gpt = current.ai_distilgpt2_sec_per_1000
    if "ai_distilgpt2_sec_per_1000" in observed_speeds and observed_speeds["ai_distilgpt2_sec_per_1000"] > 0:
        new_gpt = (1.0 - alpha) * current.ai_distilgpt2_sec_per_1000 + alpha * observed_speeds["ai_distilgpt2_sec_per_1000"]

    new_sty = current.ai_stylometry_sec_per_1000
    if "ai_stylometry_sec_per_1000" in observed_speeds and observed_speeds["ai_stylometry_sec_per_1000"] > 0:
        new_sty = (1.0 - alpha) * current.ai_stylometry_sec_per_1000 + alpha * observed_speeds["ai_stylometry_sec_per_1000"]

    new_sem = current.semantic_sec_per_1000
    if "semantic_sec_per_1000" in observed_speeds and observed_speeds["semantic_sec_per_1000"] > 0:
        new_sem = (1.0 - alpha) * current.semantic_sec_per_1000 + alpha * observed_speeds["semantic_sec_per_1000"]

    updated = TimingData(
        ai_distilgpt2_sec_per_1000=round(new_gpt, 2),
        ai_stylometry_sec_per_1000=round(new_sty, 3),
        semantic_sec_per_1000=round(new_sem, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        cpu_thread_count=os.cpu_count() or 2,
    )
    save_timing(updated, path=path)
    return updated


def generate_synthetic_corpus(word_count: int) -> str:
    """Generates synthetic English academic prose for benchmarking."""
    base_sentence = (
        "The empirical evaluation demonstrated significant structural variance across experimental conditions. "
        "Furthermore, qualitative assessments aligned with theoretical frameworks proposed in earlier foundational studies. "
        "Subsequent investigations revealed systematic patterns within the observed distributions. "
    )
    words = base_sentence.split()
    repeats = (word_count // len(words)) + 1
    return " ".join((base_sentence * repeats).split()[:word_count])


def calibrate(
    sample_words: int = 500,
    verbose: bool = True,
    target_path: Optional[Path] = None,
) -> TimingData:
    """Performs calibration benchmark on synthetic text and records seconds per 1,000 words."""
    if verbose:
        print("=" * 65)
        print("  VERITEXT HARDWARE TIMING CALIBRATION")
        print("=" * 65)
        print(f"Sampling corpus: {sample_words} synthetic words")
        print(f"CPU threads:     {os.cpu_count() or 2}")

    text = generate_synthetic_corpus(sample_words)
    words = text.split()
    total_words = len(words)
    factor = 1000.0 / max(1, total_words)

    # 1. Stylometry Benchmark
    if verbose:
        print("\n[1/3] Benchmarking stylometric feature extraction...")
    from veritext.ai_detect.features_style import compute_stylometric_features
    from veritext.ingest.splitter import split_paragraphs, split_sentences

    sents = split_sentences(text)
    paras = split_paragraphs(text)
    t0 = time.time()
    for _ in range(5):
        compute_stylometric_features(text, [s.text for s in sents], [p.text for p in paras])
    t_sty = (time.time() - t0) / 5.0
    sty_sec_per_1000 = max(0.05, round(t_sty * factor, 3))
    if verbose:
        print(f"      Stylometry speed: {sty_sec_per_1000:.3f} s / 1,000 words")

    # 2. DistilGPT-2 Benchmark
    gpt_sec_per_1000 = DEFAULT_AI_DISTILGPT2_SEC
    from veritext.ai_detect.features_lm import is_distilgpt2_available, load_distilgpt2
    if is_distilgpt2_available():
        if verbose:
            print("\n[2/3] Benchmarking distilgpt2 perplexity & burstiness...")
        try:
            tokenizer, model = load_distilgpt2()
            from veritext.ai_detect.features_lm import compute_lm_features
            t0 = time.time()
            compute_lm_features(text, [s.text for s in sents], tokenizer=tokenizer, model=model)
            t_gpt = time.time() - t0
            gpt_sec_per_1000 = max(1.0, round(t_gpt * factor, 2))
            if verbose:
                print(f"      DistilGPT-2 speed: {gpt_sec_per_1000:.2f} s / 1,000 words")
        except Exception as e:
            if verbose:
                print(f"      DistilGPT-2 benchmark skipped: {e}")
    else:
        if verbose:
            print("\n[2/3] distilgpt2 weights not found locally. Using conservative baseline.")

    # 3. Semantic Vector Search Benchmark
    sem_sec_per_1000 = DEFAULT_SEMANTIC_SEC
    if verbose:
        print("\n[3/3] Benchmarking dense semantic vector embeddings...")
    try:
        from veritext.similarity.semantic import encode_texts_cached, get_embedding_model
        model = get_embedding_model()
        test_sentences = [s.text for s in sents[:10]]
        t0 = time.time()
        encode_texts_cached(test_sentences, model=model, batch_size=16)
        t_sem = time.time() - t0
        sem_factor = 1000.0 / max(1, sum(len(s.split()) for s in test_sentences))
        sem_sec_per_1000 = max(0.5, round(t_sem * sem_factor, 2))
        if verbose:
            print(f"      Semantic vector speed: {sem_sec_per_1000:.2f} s / 1,000 words")
    except Exception as e:
        if verbose:
            print(f"      Semantic benchmark skipped: {e}")

    timing = TimingData(
        ai_distilgpt2_sec_per_1000=gpt_sec_per_1000,
        ai_stylometry_sec_per_1000=sty_sec_per_1000,
        semantic_sec_per_1000=sem_sec_per_1000,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cpu_thread_count=os.cpu_count() or 2,
    )

    save_timing(timing, path=target_path)
    if verbose:
        print("\n" + "=" * 65)
        print(f"Calibration saved to: {target_path or TIMING_FILE}")
        print("=" * 65)
    return timing


def main(argv=None):
    parser = argparse.ArgumentParser(prog="veritext.timing", description="Hardware timing calibration")
    subparsers = parser.add_subparsers(dest="command")
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate speed benchmarks on this machine")
    cal_parser.add_argument("--words", type=int, default=500, help="Number of synthetic words to benchmark (default 500)")

    args = parser.parse_args(argv)
    if args.command == "calibrate" or args.command is None:
        words = getattr(args, "words", 500) or 500
        calibrate(sample_words=words, verbose=True)


if __name__ == "__main__":
    main()
