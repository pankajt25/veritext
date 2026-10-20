"""Adaptive analysis planner and per-engine resource allocation.

Determines per-engine execution parameters from document profile, measured hardware speed,
and time budgets. Guarantees 100% coverage for exact matching and integrity checks,
and uses section-sampling across the entire document for long documents under time budgets.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from veritext.profile import DocumentProfile
from veritext.timing import TimingData, load_timing

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "config"
LIMITS_FILE = CONFIG_DIR / "limits.json"

DEFAULT_TIME_BUDGET_MINUTES = 10
BLOCK_SIZE_WORDS = 500


@dataclass
class EnginePlan:
    """Execution parameters and coverage targets for a specific analysis engine."""

    engine: str
    enabled: bool
    reason: str
    words_to_analyze: int
    total_words: int
    coverage_percent: float
    estimated_seconds: float
    sampling_mode: str  # "full", "section_sampled", "none"
    word_ranges: List[Tuple[int, int]]  # [(start_word_idx, end_word_idx), ...]
    coverage_description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnginePlan":
        ranges = [tuple(r) for r in data.get("word_ranges", [])]
        return cls(
            engine=str(data.get("engine", "")),
            enabled=bool(data.get("enabled", True)),
            reason=str(data.get("reason", "")),
            words_to_analyze=int(data.get("words_to_analyze", 0)),
            total_words=int(data.get("total_words", 0)),
            coverage_percent=float(data.get("coverage_percent", 100.0)),
            estimated_seconds=float(data.get("estimated_seconds", 0.0)),
            sampling_mode=str(data.get("sampling_mode", "full")),
            word_ranges=ranges,
            coverage_description=str(data.get("coverage_description", "")),
        )


@dataclass
class AnalysisPlan:
    """Consolidated plan across all engines prior to analysis execution."""

    mode: str
    time_budget_minutes: float
    total_estimated_seconds: float
    warnings: List[str]
    engines: Dict[str, EnginePlan]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "time_budget_minutes": self.time_budget_minutes,
            "total_estimated_seconds": self.total_estimated_seconds,
            "warnings": self.warnings,
            "engines": {k: v.to_dict() for k, v in self.engines.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisPlan":
        engines = {
            k: EnginePlan.from_dict(v)
            for k, v in data.get("engines", {}).items()
        }
        return cls(
            mode=str(data.get("mode", "auto")),
            time_budget_minutes=float(data.get("time_budget_minutes", 10.0)),
            total_estimated_seconds=float(data.get("total_estimated_seconds", 0.0)),
            warnings=list(data.get("warnings", [])),
            engines=engines,
        )


def load_limits_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads default limits from data/config/limits.json."""
    target_path = path or LIMITS_FILE
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "mode": "auto",
        "time_budget_minutes": DEFAULT_TIME_BUDGET_MINUTES,
        "custom_ai_words": 0,
        "custom_semantic_words": 0,
    }


def compute_section_sampling_ranges(
    total_words: int,
    target_words: int,
    block_size: int = BLOCK_SIZE_WORDS,
) -> List[Tuple[int, int]]:
    """Partitions a document into ~500-word contiguous blocks and selects evenly spaced blocks.

    Always includes the first and last block across the entire document.
    """
    if total_words <= 0 or target_words <= 0:
        return []

    if target_words >= total_words or total_words <= block_size:
        return [(0, total_words)]

    num_blocks = math.ceil(total_words / block_size)
    if num_blocks <= 1:
        return [(0, total_words)]

    # Determine number of blocks to select
    k = max(1, min(num_blocks, round(target_words / block_size)))
    if k >= num_blocks:
        return [(i * block_size, min(total_words, (i + 1) * block_size)) for i in range(num_blocks)]

    if k == 1:
        return [(0, min(total_words, block_size))]

    # Pick k evenly spaced block indices from 0 to num_blocks - 1
    # Ensures index 0 (first block) and index num_blocks - 1 (last block) are always selected
    selected_indices = sorted(
        list({round(i * (num_blocks - 1) / (k - 1)) for i in range(k)})
    )

    # In rare cases where rounding collapsed an index, ensure k items
    cur = 1
    while len(selected_indices) < k and cur < num_blocks - 1:
        if cur not in selected_indices:
            selected_indices.append(cur)
            selected_indices.sort()
        cur += 1

    ranges: List[Tuple[int, int]] = []
    for idx in selected_indices:
        b_start = idx * block_size
        b_end = min(total_words, (idx + 1) * block_size)
        ranges.append((b_start, b_end))

    return ranges


def make_plan(
    profile: DocumentProfile,
    options: Optional[Dict[str, Any]] = None,
    timing: Optional[TimingData] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> AnalysisPlan:
    """Constructs an AnalysisPlan defining limits, word ranges, and time estimates per engine.

    Args:
        profile: Computed DocumentProfile of the target text.
        options: Runtime options passed from UI or CLI.
        timing: Calibrated machine speeds (TimingData).
        settings: Limits settings loaded from limits.json.

    Returns:
        AnalysisPlan with per-engine resource allocation.
    """
    opts = options or {}
    limits = settings or load_limits_config()
    perf = timing or load_timing()

    mode = str(opts.get("mode", limits.get("mode", "auto"))).lower()
    if mode not in ("auto", "full", "custom"):
        mode = "auto"

    budget_minutes = float(
        opts.get("time_budget_minutes", limits.get("time_budget_minutes", DEFAULT_TIME_BUDGET_MINUTES))
    )
    budget_seconds = budget_minutes * 60.0

    total_words = profile.word_count
    engines: Dict[str, EnginePlan] = {}
    warnings: List[str] = []

    # 1. Exact Matching (always 100% full document, no word cap)
    exact_enabled = opts.get("use_exact", True)
    exact_est_seconds = max(0.1, round((total_words / 1000.0) * 0.15, 2))
    engines["exact"] = EnginePlan(
        engine="exact",
        enabled=exact_enabled,
        reason="Full document: exact matching has no word cap",
        words_to_analyze=total_words if exact_enabled else 0,
        total_words=total_words,
        coverage_percent=100.0 if (exact_enabled and total_words > 0) else 0.0,
        estimated_seconds=exact_est_seconds if exact_enabled else 0.0,
        sampling_mode="full" if exact_enabled else "none",
        word_ranges=[(0, total_words)] if (exact_enabled and total_words > 0) else [],
        coverage_description="full document analysed" if exact_enabled else "exact matching skipped",
    )

    # 2. Integrity Checks (always 100% full document, no word cap)
    checks_enabled = opts.get("use_integrity", True)
    checks_est_seconds = max(0.1, round((total_words / 1000.0) * 0.20, 2))
    engines["checks"] = EnginePlan(
        engine="checks",
        enabled=checks_enabled,
        reason="Full document: integrity checks have no word cap",
        words_to_analyze=total_words if checks_enabled else 0,
        total_words=total_words,
        coverage_percent=100.0 if (checks_enabled and total_words > 0) else 0.0,
        estimated_seconds=checks_est_seconds if checks_enabled else 0.0,
        sampling_mode="full" if checks_enabled else "none",
        word_ranges=[(0, total_words)] if (checks_enabled and total_words > 0) else [],
        coverage_description="full document analysed" if checks_enabled else "integrity checks skipped",
    )

    # 3. AI Detection
    ai_enabled = opts.get("use_ai", True)
    if not ai_enabled:
        engines["ai_detect"] = EnginePlan(
            engine="ai_detect",
            enabled=False,
            reason="AI detection skipped: disabled by user",
            words_to_analyze=0,
            total_words=total_words,
            coverage_percent=0.0,
            estimated_seconds=0.0,
            sampling_mode="none",
            word_ranges=[],
            coverage_description="AI detection skipped: disabled by user",
        )
    elif total_words < 150:
        engines["ai_detect"] = EnginePlan(
            engine="ai_detect",
            enabled=False,
            reason="AI detection skipped: under 150 words",
            words_to_analyze=0,
            total_words=total_words,
            coverage_percent=0.0,
            estimated_seconds=0.0,
            sampling_mode="none",
            word_ranges=[],
            coverage_description="AI detection skipped: under 150 words",
        )
    else:
        # Determine speed model
        ai_speed = perf.ai_distilgpt2_sec_per_1000
        full_ai_time = (total_words / 1000.0) * ai_speed

        if mode == "full":
            ai_words = total_words
            ai_cov = 100.0
            ai_mode = "full"
            ai_ranges = [(0, total_words)]
            ai_est = full_ai_time
            ai_desc = "full document analysed"
            if full_ai_time > budget_seconds:
                warnings.append(
                    f"Full document AI detection estimated at {int(full_ai_time // 60)}m {int(full_ai_time % 60)}s "
                    f"(exceeds budget of {int(budget_minutes)}m)."
                )
        elif mode == "custom":
            custom_max = int(opts.get("custom_ai_words", limits.get("custom_ai_words", 0)))
            if custom_max <= 0 or custom_max >= total_words:
                ai_words = total_words
                ai_cov = 100.0
                ai_mode = "full"
                ai_ranges = [(0, total_words)]
                ai_est = full_ai_time
                ai_desc = "full document analysed"
            else:
                ai_ranges = compute_section_sampling_ranges(total_words, custom_max)
                ai_words = sum(end - start for start, end in ai_ranges)
                ai_cov = round((ai_words / total_words) * 100.0, 1)
                ai_mode = "section_sampled"
                ai_est = (ai_words / 1000.0) * ai_speed
                ai_desc = f"AI detection analysed {ai_words:,} of {total_words:,} words ({ai_cov:.0f}%) using evenly spaced sections"
        else:
            # Auto mode: budget driven
            if full_ai_time <= budget_seconds:
                ai_words = total_words
                ai_cov = 100.0
                ai_mode = "full"
                ai_ranges = [(0, total_words)]
                ai_est = full_ai_time
                ai_desc = "full document analysed"
            else:
                target_words = int((budget_seconds / ai_speed) * 1000)
                ai_ranges = compute_section_sampling_ranges(total_words, target_words)
                ai_words = sum(end - start for start, end in ai_ranges)
                ai_cov = round((ai_words / total_words) * 100.0, 1)
                ai_mode = "section_sampled"
                ai_est = (ai_words / 1000.0) * ai_speed
                ai_desc = f"AI detection analysed {ai_words:,} of {total_words:,} words ({ai_cov:.0f}%) using evenly spaced sections"

        engines["ai_detect"] = EnginePlan(
            engine="ai_detect",
            enabled=True,
            reason=f"AI detection enabled: {ai_desc}",
            words_to_analyze=ai_words,
            total_words=total_words,
            coverage_percent=ai_cov,
            estimated_seconds=round(ai_est, 1),
            sampling_mode=ai_mode,
            word_ranges=ai_ranges,
            coverage_description=ai_desc,
        )

    # 4. Semantic Matching
    sem_enabled = opts.get("use_semantic", False)
    if not sem_enabled:
        engines["semantic"] = EnginePlan(
            engine="semantic",
            enabled=False,
            reason="Semantic matching skipped: disabled by user",
            words_to_analyze=0,
            total_words=total_words,
            coverage_percent=0.0,
            estimated_seconds=0.0,
            sampling_mode="none",
            word_ranges=[],
            coverage_description="Semantic matching skipped: disabled by user",
        )
    elif total_words == 0:
        engines["semantic"] = EnginePlan(
            engine="semantic",
            enabled=False,
            reason="Semantic matching skipped: empty document",
            words_to_analyze=0,
            total_words=total_words,
            coverage_percent=0.0,
            estimated_seconds=0.0,
            sampling_mode="none",
            word_ranges=[],
            coverage_description="Semantic matching skipped: empty document",
        )
    else:
        sem_speed = perf.semantic_sec_per_1000
        full_sem_time = (total_words / 1000.0) * sem_speed

        if mode == "full":
            sem_words = total_words
            sem_cov = 100.0
            sem_mode = "full"
            sem_ranges = [(0, total_words)]
            sem_est = full_sem_time
            sem_desc = "full document analysed"
            if full_sem_time > budget_seconds:
                warnings.append(
                    f"Full document semantic matching estimated at {int(full_sem_time // 60)}m {int(full_sem_time % 60)}s "
                    f"(exceeds budget of {int(budget_minutes)}m)."
                )
        elif mode == "custom":
            custom_max = int(opts.get("custom_semantic_words", limits.get("custom_semantic_words", 0)))
            if custom_max <= 0 or custom_max >= total_words:
                sem_words = total_words
                sem_cov = 100.0
                sem_mode = "full"
                sem_ranges = [(0, total_words)]
                sem_est = full_sem_time
                sem_desc = "full document analysed"
            else:
                sem_ranges = compute_section_sampling_ranges(total_words, custom_max)
                sem_words = sum(end - start for start, end in sem_ranges)
                sem_cov = round((sem_words / total_words) * 100.0, 1)
                sem_mode = "section_sampled"
                sem_est = (sem_words / 1000.0) * sem_speed
                sem_desc = f"Semantic matching analysed {sem_words:,} of {total_words:,} words ({sem_cov:.0f}%) using evenly spaced sections"
        else:
            # Auto mode
            if full_sem_time <= budget_seconds:
                sem_words = total_words
                sem_cov = 100.0
                sem_mode = "full"
                sem_ranges = [(0, total_words)]
                sem_est = full_sem_time
                sem_desc = "full document analysed"
            else:
                target_words = int((budget_seconds / sem_speed) * 1000)
                sem_ranges = compute_section_sampling_ranges(total_words, target_words)
                sem_words = sum(end - start for start, end in sem_ranges)
                sem_cov = round((sem_words / total_words) * 100.0, 1)
                sem_mode = "section_sampled"
                sem_est = (sem_words / 1000.0) * sem_speed
                sem_desc = f"Semantic matching analysed {sem_words:,} of {total_words:,} words ({sem_cov:.0f}%) using evenly spaced sections"

        engines["semantic"] = EnginePlan(
            engine="semantic",
            enabled=True,
            reason=f"Semantic matching enabled: {sem_desc}",
            words_to_analyze=sem_words,
            total_words=total_words,
            coverage_percent=sem_cov,
            estimated_seconds=round(sem_est, 1),
            sampling_mode=sem_mode,
            word_ranges=sem_ranges,
            coverage_description=sem_desc,
        )

    total_est = sum(e.estimated_seconds for e in engines.values() if e.enabled)
    if total_est > 15 * 60:
        warnings.append(
            f"Total estimated analysis time is {int(total_est // 60)}m {int(total_est % 60)}s (over 15 minutes). "
            f"Suggestions: untick the semantic paraphrase check or lower the time budget."
        )

    return AnalysisPlan(
        mode=mode,
        time_budget_minutes=budget_minutes,
        total_estimated_seconds=round(total_est, 1),
        warnings=warnings,
        engines=engines,
    )


def print_profile_and_plan(profile: DocumentProfile, plan: AnalysisPlan) -> None:
    """Prints formatted profile summary and plan table to console."""
    print("=" * 72)
    print("  DOCUMENT PROFILE")
    print("=" * 72)
    print(f"  Words:            {profile.word_count:,}")
    print(f"  Characters:       {profile.character_count:,}")
    print(f"  Sentences:        {profile.sentence_count:,}")
    print(f"  Paragraphs:       {profile.paragraph_count:,}")
    print(f"  Estimated Pages:  {profile.page_estimate:,} (at 500 words/page)")
    print(f"  Estimated Tokens: {profile.estimated_tokens:,}")
    print(f"  Reference Words:  {profile.reference_words:,}")
    print(f"  Language Guess:   {profile.language_guess}")
    print(f"  Size Category:    {profile.size_class.upper()}")

    print("\n" + "=" * 72)
    print(f"  ANALYSIS PLAN (Mode: {plan.mode.upper()}, Time Budget: {plan.time_budget_minutes:.1f} min)")
    print("=" * 72)
    header = f"  {'Engine':<12} | {'Status':<6} | {'Words':<8} | {'Coverage':<9} | {'Est Time':<9} | {'Mode'}"
    print(header)
    print("  " + "-" * 68)

    for eng_name, ep in plan.engines.items():
        status = "Run" if ep.enabled else "Skip"
        cov = f"{ep.coverage_percent:.1f}%"
        est = f"{ep.estimated_seconds:.1f}s"
        print(f"  {eng_name:<12} | {status:<6} | {ep.words_to_analyze:<8,d} | {cov:<9} | {est:<9} | {ep.sampling_mode}")

    print("  " + "-" * 68)
    total_min = plan.total_estimated_seconds / 60.0
    print(f"  Total Estimated Execution Time: {plan.total_estimated_seconds:.1f}s (~{total_min:.1f} min)")

    if plan.warnings:
        print("\n  ⚠️ Plan Notices:")
        for w in plan.warnings:
            print(f"    • {w}")
    print("=" * 72 + "\n")
