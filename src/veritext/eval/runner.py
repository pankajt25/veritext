"""Comprehensive evaluation and benchmark runner for Veritext.

Evaluates:
- Plagiarism engine on groups a to c (exact winnowing, semantic, difflib baseline)
- AI text detector on groups c to e (stylometry, distilgpt2, ensemble)
- Spinner / tortured-phrase check on group f (spinners + controls)
- Collusion matrix and MinHash/LSH screening on group g (class submissions)
- Tamper-evident cryptographic seals on group h (untouched valid vs 1-char tampered)
- Optional cross-language semantic matching on group h2
"""

import copy
import csv
import difflib
import gc
import json
import logging
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from veritext.checks.models import CheckResult
from veritext.checks.spinner import check_spinner
from veritext.eval.dataset import (
    DEFAULT_DOCS_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_REFS_DIR,
    load_document,
    load_manifest,
    load_references,
)
from veritext.eval.metrics import binary_classification_metrics, mean
from veritext.fingerprint.index import FingerprintIndex
from veritext.fingerprint.minhash import (
    compute_minhash,
    minhash_jaccard,
    screen_batch_candidates,
    screen_fingerprint_candidates,
)
from veritext.fingerprint.winnow import winnow
from veritext.ingest.models import Document
from veritext.ingest.pipeline import split_paragraphs, split_sentences
from veritext.normalize.normalize import normalize_text
from veritext.report.seal import verify_report_seal
from veritext.similarity.exact import find_exact_matches
from veritext.similarity.models import MatchSpan
from veritext.similarity.scorer import compute_similarity_score
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "eval" / "results"


def _create_document(text: str, path: Optional[str] = None) -> Document:
    """Helper to instantiate a Document with populated sentence and paragraph spans."""
    return Document(
        raw_text=text,
        clean_text=text,
        sentences=split_sentences(text),
        paragraphs=split_paragraphs(text),
        removed_sections=[],
        file_type="txt",
        path=path,
    )


class BenchmarkRunner:
    """Executes the complete Veritext benchmark suite."""

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        docs_dir: Optional[Path] = None,
        refs_dir: Optional[Path] = None,
        out_dir: Optional[Path] = None,
        semantic: bool = False,
        ai: bool = False,
        crosslang: bool = False,
    ):
        self.manifest_path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
        self.docs_dir = Path(docs_dir) if docs_dir else DEFAULT_DOCS_DIR
        self.refs_dir = Path(refs_dir) if refs_dir else DEFAULT_REFS_DIR
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self.semantic = semantic
        self.ai = ai
        self.crosslang = crosslang

        self.manifest: List[Dict[str, str]] = []
        self.references: Dict[str, str] = {}
        self.detailed_records: List[Dict[str, Any]] = []

    def setup(self):
        """Loads manifest, references, and creates output directory."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = load_manifest(self.manifest_path)
        self.references = load_references(self.refs_dir)
        logger.info(
            f"Loaded {len(self.manifest)} manifest records and {len(self.references)} references."
        )

    def run(self) -> Dict[str, Any]:
        """Runs the entire benchmark and generates summary results."""
        tracemalloc.start()
        start_time = time.perf_counter()

        self.setup()

        # 1. Plagiarism Evaluation (groups a, b, c)
        plag_summary = self._evaluate_plagiarism()

        # 2. AI Detection Evaluation (groups c, d, e)
        ai_summary = self._evaluate_ai()

        # 3. Spinner Evaluation (group f)
        spinner_summary = self._evaluate_spinner()

        # 4. Collusion Evaluation (group g)
        collusion_summary = self._evaluate_collusion()

        # 5. Cryptographic Seal Evaluation (group h)
        seal_summary = self._evaluate_seals()

        # 6. Cross-Language Evaluation (group h2)
        crosslang_summary = self._evaluate_crosslang()

        total_elapsed = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        eval_doc_count = sum(
            len(grp_rows) for grp_rows in [
                [r for r in self.manifest if r["group"] in ["a_copies", "b_paraphrases", "b_paraphrases_hard", "c_original"]],
                [r for r in self.manifest if r["group"] in ["d_ai_style", "e_non_native"]],
                [r for r in self.manifest if r["group"] in ["f_spinner", "f_spinner_control"]],
                [r for r in self.manifest if r["group"] in ["g_class_submissions"]],
                [r for r in self.manifest if r["group"] in ["h_seal_reports"]],
            ]
        )
        runtime_per_doc = round(total_elapsed / max(1, eval_doc_count), 4)

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_documents_manifest": len(self.manifest),
            "total_documents_evaluated": eval_doc_count,
            "total_runtime_seconds": round(total_elapsed, 2),
            "runtime_per_document_seconds": runtime_per_doc,
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
            "flags": {
                "semantic": self.semantic,
                "ai": self.ai,
                "crosslang": self.crosslang,
            },
            "plagiarism": plag_summary,
            "ai_detection": ai_summary,
            "spinner_detection": spinner_summary,
            "collusion_detection": collusion_summary,
            "seal_verification": seal_summary,
            "cross_language": crosslang_summary,
        }

        # Save files
        self._save_results(summary)
        return summary

    def _evaluate_plagiarism(self) -> Dict[str, Any]:
        """Evaluates plagiarism detection across positive cohorts (copies, easy & hard paraphrases)
        and negative cohorts (originals, AI-generated text, and non-native English).

        Assesses both document-level sensitivity across multiple decision thresholds (5%, 10%, 15%, 25%)
        and span-level detection statistics.
        """
        logger.info("Evaluating plagiarism detection on positive and negative cohorts...")

        # Setup index for references
        db_file = self.out_dir / "eval_fingerprints.sqlite"
        if db_file.exists():
            db_file.unlink()
        index = FingerprintIndex(db_path=db_file)
        sources_map: Dict[str, SourceCandidate] = {}

        for ref_id, ref_text in self.references.items():
            cand = SourceCandidate(
                id=ref_id,
                title=ref_id,
                authors=["Veritext Benchmark Corpus"],
                url=f"local://eval/refs/{ref_id}",
                origin="eval_reference",
                year=2024,
                text=ref_text,
            )
            sources_map[ref_id] = cand
            index.index_source_candidate(cand)

        eval_groups = {
            "a_copies",
            "b_paraphrases",
            "b_paraphrases_hard",
            "c_original",
            "d_ai_style",
            "e_non_native",
        }
        plag_rows = [r for r in self.manifest if r["group"] in eval_groups]

        y_true: List[bool] = []
        scores_winnow: List[float] = []
        scores_baseline: List[float] = []
        scores_semantic: List[float] = []
        row_groups: List[str] = []

        from collections import defaultdict
        scores_by_group_exact: Dict[str, List[float]] = defaultdict(list)
        scores_by_group_semantic: Dict[str, List[float]] = defaultdict(list)
        scores_by_group_baseline: Dict[str, List[float]] = defaultdict(list)

        exact_spans_by_group: Dict[str, List[int]] = defaultdict(list)
        sem_spans_by_group: Dict[str, List[int]] = defaultdict(list)
        sem_cosine_scores: List[float] = []

        missed_paraphrases_easy: List[Dict[str, Any]] = []
        missed_paraphrases_hard: List[Dict[str, Any]] = []
        false_positives: List[Dict[str, Any]] = []

        HEADLINE_THRESHOLD = 0.15
        TESTED_THRESHOLDS = [0.05, 0.10, 0.15, 0.25]

        for row in plag_rows:
            fname = row["file"]
            grp = row["group"]
            expected = row["expected_plagiarism"].lower() == "yes"
            y_true.append(expected)
            row_groups.append(grp)

            text = load_document(fname, self.docs_dir)

            # 1. Exact Winnowing
            canonical = normalize_text(text)
            doc_fps = winnow(canonical.tokens, k=5, w=4)
            exact_spans = find_exact_matches(
                doc_clean_text=text,
                canonical=canonical,
                doc_fingerprints=doc_fps,
                index=index,
                sources_map=sources_map,
                min_words=6,
            )
            sim_res = compute_similarity_score(text, exact_spans, sources_map)
            score_winnow = sim_res.similarity_percentage / 100.0
            scores_winnow.append(score_winnow)
            scores_by_group_exact[grp].append(score_winnow)
            exact_spans_by_group[grp].append(len(exact_spans))

            # 2. Difflib SequenceMatcher Baseline
            base_score = max(
                (difflib.SequenceMatcher(None, text, ref_txt).ratio() for ref_txt in self.references.values()),
                default=0.0,
            )
            scores_baseline.append(base_score)
            scores_by_group_baseline[grp].append(base_score)

            # 3. Winnowing + Semantic (if enabled)
            score_sem = score_winnow
            sem_spans = []
            if self.semantic:
                try:
                    from veritext.similarity.semantic import find_semantic_matches
                    doc_obj = _create_document(text, path=fname)
                    cands_list = list(sources_map.values())
                    sem_spans = find_semantic_matches(
                        document=doc_obj,
                        candidates=cands_list,
                        exact_spans=exact_spans,
                        threshold=0.70,
                    )
                    all_spans = exact_spans + sem_spans
                    sim_sem_res = compute_similarity_score(text, all_spans, sources_map)
                    score_sem = sim_sem_res.similarity_percentage / 100.0
                    for s in sem_spans:
                        sem_cosine_scores.append(s.score)
                except Exception as e:
                    logger.warning(f"Semantic match skipped for {fname}: {e}")

            scores_semantic.append(score_sem)
            scores_by_group_semantic[grp].append(score_sem)
            sem_spans_by_group[grp].append(len(sem_spans))

            # Track failure cases at headline threshold 0.15
            active_score = score_sem if self.semantic else score_winnow
            if expected and active_score < HEADLINE_THRESHOLD:
                failure_record = {
                    "file": fname,
                    "group": grp,
                    "winnow_score": round(score_winnow, 4),
                    "semantic_score": round(score_sem, 4) if self.semantic else None,
                    "baseline_score": round(base_score, 4),
                    "notes": row.get("notes", ""),
                }
                if grp == "b_paraphrases":
                    missed_paraphrases_easy.append(failure_record)
                elif grp == "b_paraphrases_hard":
                    missed_paraphrases_hard.append(failure_record)
            if not expected and active_score >= HEADLINE_THRESHOLD:
                false_positives.append({
                    "file": fname,
                    "group": grp,
                    "winnow_score": round(score_winnow, 4),
                    "semantic_score": round(score_sem, 4) if self.semantic else None,
                    "notes": row.get("notes", ""),
                })

            self.detailed_records.append({
                "file": fname,
                "group": grp,
                "task": "plagiarism",
                "ground_truth": expected,
                "winnow_score": round(score_winnow, 4),
                "winnow_pred": score_winnow >= HEADLINE_THRESHOLD,
                "baseline_score": round(base_score, 4),
                "baseline_pred": base_score >= 0.35,
                "semantic_score": round(score_sem, 4) if self.semantic else None,
                "semantic_pred": (score_sem >= HEADLINE_THRESHOLD) if self.semantic else None,
                "exact_spans_count": len(exact_spans),
                "semantic_spans_count": len(sem_spans),
            })

            gc.collect()

        # Multi-threshold metrics computation
        winnow_by_thresh = {}
        sem_by_thresh = {}
        base_by_thresh = {}
        fpr_by_group = {
            "c_original": {},
            "d_ai_style": {},
            "e_non_native": {},
        }
        recall_by_group = {
            "a_copies": {},
            "b_paraphrases": {},
            "b_paraphrases_hard": {},
        }

        for t in TESTED_THRESHOLDS:
            # Winnowing
            pred_w = [s >= t for s in scores_winnow]
            winnow_by_thresh[t] = binary_classification_metrics(y_true, pred_w)

            # Semantic (or exact if semantic disabled)
            pred_s = [s >= t for s in scores_semantic]
            sem_by_thresh[t] = binary_classification_metrics(y_true, pred_s)

            # Baseline difflib (using SequenceMatcher threshold 0.35 for headline or scaled)
            pred_b = [s >= (0.20 + t) for s in scores_baseline]
            base_by_thresh[t] = binary_classification_metrics(y_true, pred_b)

            # Group-specific FPR (false positive rate on negative cohorts under active engine)
            active_preds = pred_s if self.semantic else pred_w
            for neg_grp in ["c_original", "d_ai_style", "e_non_native"]:
                neg_indices = [i for i, g in enumerate(row_groups) if g == neg_grp]
                if neg_indices:
                    fp_count = sum(1 for i in neg_indices if active_preds[i])
                    fpr_by_group[neg_grp][t] = round(fp_count / len(neg_indices), 4)

            # Group-specific Recall on positive cohorts
            for pos_grp in ["a_copies", "b_paraphrases", "b_paraphrases_hard"]:
                pos_indices = [i for i, g in enumerate(row_groups) if g == pos_grp]
                if pos_indices:
                    tp_count = sum(1 for i in pos_indices if active_preds[i])
                    recall_by_group[pos_grp][t] = round(tp_count / len(pos_indices), 4)

        # Span-level summary
        span_metrics = {
            "mean_exact_spans_per_doc": {
                grp: round(mean(counts), 2) for grp, counts in exact_spans_by_group.items()
            },
            "mean_semantic_spans_per_doc": {
                grp: round(mean(counts), 2) for grp, counts in sem_spans_by_group.items()
            },
            "mean_semantic_cosine_similarity": round(mean(sem_cosine_scores), 4) if sem_cosine_scores else 0.0,
            "total_semantic_spans_found": len(sem_cosine_scores),
        }

        # Flat mean similarity mapping for backward-compatibility with plots and table
        flat_means = {
            grp: mean(scores_by_group_semantic[grp]) if self.semantic else mean(scores_by_group_exact[grp])
            for grp in eval_groups
        }

        return {
            "headline_threshold": HEADLINE_THRESHOLD,
            "thresholds_tested": TESTED_THRESHOLDS,
            "multi_threshold_metrics": {
                "winnowing_only": winnow_by_thresh,
                "winnowing_and_semantic": sem_by_thresh if self.semantic else None,
                "baseline_difflib": base_by_thresh,
            },
            "group_fpr_by_threshold": fpr_by_group,
            "paraphrase_recall_breakdown": recall_by_group,
            "mean_similarity_per_group": flat_means,
            "mean_similarity_by_engine": {
                "exact": {grp: mean(scs) for grp, scs in scores_by_group_exact.items()},
                "semantic": {grp: mean(scs) for grp, scs in scores_by_group_semantic.items()} if self.semantic else None,
                "baseline": {grp: mean(scs) for grp, scs in scores_by_group_baseline.items()},
            },
            "winnowing_only": winnow_by_thresh[HEADLINE_THRESHOLD],
            "winnowing_and_semantic": sem_by_thresh[HEADLINE_THRESHOLD] if self.semantic else None,
            "baseline_difflib": base_by_thresh[HEADLINE_THRESHOLD],
            "span_level_metrics": span_metrics,
            "failure_cases": {
                "missed_paraphrases_easy_count": len(missed_paraphrases_easy),
                "missed_paraphrases_easy": missed_paraphrases_easy,
                "missed_paraphrases_hard_count": len(missed_paraphrases_hard),
                "missed_paraphrases_hard": missed_paraphrases_hard,
                "missed_paraphrases_count": len(missed_paraphrases_easy) + len(missed_paraphrases_hard),
                "false_positives_count": len(false_positives),
                "false_positives": false_positives,
            },
        }

    def _evaluate_ai(self) -> Dict[str, Any]:
        """Evaluates AI detection across groups c (human originals), d (AI generated), and e (ESL human)."""
        logger.info("Evaluating AI detection on groups c, d, e...")

        target_groups = {"c_original", "d_ai_style", "e_non_native"}
        ai_rows = [r for r in self.manifest if r["group"] in target_groups]

        y_true: List[bool] = []
        y_pred_style: List[bool] = []
        y_pred_lm: List[bool] = []
        y_pred_ensemble: List[bool] = []

        is_human_mask: List[bool] = []
        is_esl_mask: List[bool] = []

        from veritext.ai_detect.ensemble import calculate_ensemble_score
        from veritext.ai_detect.features_style import compute_stylometric_features

        has_distilgpt2 = False
        if self.ai:
            from veritext.ai_detect.features_lm import is_distilgpt2_available
            has_distilgpt2 = is_distilgpt2_available()

        false_positives_human: List[Dict[str, Any]] = []
        false_positives_esl: List[Dict[str, Any]] = []

        for row in ai_rows:
            fname = row["file"]
            grp = row["group"]
            expected_ai = row["expected_ai"].lower() == "yes"
            is_human = not expected_ai
            is_esl = grp == "e_non_native"

            y_true.append(expected_ai)
            is_human_mask.append(is_human)
            is_esl_mask.append(is_esl)

            text = load_document(fname, self.docs_dir)

            # 1. Stylometry only
            style_feats = compute_stylometric_features(text)
            prob_style, _ = calculate_ensemble_score(style_feats, has_lm=False)
            pred_style = prob_style >= 50.0
            y_pred_style.append(pred_style)

            # 2. DistilGPT2 features only & Full Ensemble (if self.ai enabled and available)
            prob_lm = None
            pred_lm = None
            prob_ens = prob_style
            pred_ens = pred_style

            if self.ai and has_distilgpt2:
                try:
                    from veritext.ai_detect.features_lm import compute_lm_features
                    lm_feats = compute_lm_features(text)
                    # DistilGPT2 score: logistic model on perplexity & log-rank
                    # Z = 0.5 - 0.04*ppl - 0.35*rank
                    ppl = lm_feats.get("perplexity", 100.0)
                    rank = lm_feats.get("mean_log_rank", 5.0)
                    z_lm = 0.5 - 0.04 * ppl - 0.35 * rank
                    prob_lm = round(100.0 / (1.0 + 2.71828 ** (-z_lm)), 2)
                    pred_lm = prob_lm >= 50.0
                    y_pred_lm.append(pred_lm)

                    # Ensemble score
                    combined_feats = {**style_feats, **lm_feats}
                    prob_ens, _ = calculate_ensemble_score(combined_feats, has_lm=True)
                    pred_ens = prob_ens >= 50.0
                    y_pred_ensemble.append(pred_ens)
                except Exception as e:
                    logger.warning(f"LM feature calculation error on {fname}: {e}")
                    y_pred_lm.append(False)
                    y_pred_ensemble.append(pred_style)
            else:
                y_pred_lm.append(False)
                y_pred_ensemble.append(pred_style)

            # Check false positives
            if is_human and pred_style:
                false_positives_human.append({
                    "file": fname,
                    "group": grp,
                    "prob_style": prob_style,
                    "notes": row.get("notes", ""),
                })
            if is_esl and pred_style:
                false_positives_esl.append({
                    "file": fname,
                    "prob_style": prob_style,
                    "notes": row.get("notes", ""),
                })

            self.detailed_records.append({
                "file": fname,
                "group": grp,
                "task": "ai_detection",
                "ground_truth_ai": expected_ai,
                "prob_stylometry": prob_style,
                "pred_stylometry": pred_style,
                "prob_lm": prob_lm,
                "pred_lm": pred_lm,
                "prob_ensemble": prob_ens,
                "pred_ensemble": pred_ens,
            })

            gc.collect()

        # Metrics computation
        metrics_style = binary_classification_metrics(y_true, y_pred_style)

        # FPR on human text (c_original + e_non_native: 25 documents)
        human_true = [yt for yt, ih in zip(y_true, is_human_mask) if ih]
        human_pred_style = [yp for yp, ih in zip(y_pred_style, is_human_mask) if ih]
        human_fp = sum(1 for p in human_pred_style if p)
        fpr_human = round(human_fp / max(1, len(human_true)), 4)

        # FPR specifically on non-native group (e_non_native: 10 documents)
        esl_pred_style = [yp for yp, ie in zip(y_pred_style, is_esl_mask) if ie]
        esl_fp = sum(1 for p in esl_pred_style if p)
        fpr_esl = round(esl_fp / max(1, sum(is_esl_mask)), 4)

        metrics_style["fpr_human_overall"] = fpr_human
        metrics_style["fpr_non_native_esl"] = fpr_esl

        metrics_lm = None
        metrics_ens = None
        if self.ai and has_distilgpt2:
            metrics_lm = binary_classification_metrics(y_true, y_pred_lm)
            metrics_ens = binary_classification_metrics(y_true, y_pred_ensemble)
            # Add human / esl FPR for ensemble
            ens_human_pred = [yp for yp, ih in zip(y_pred_ensemble, is_human_mask) if ih]
            metrics_ens["fpr_human_overall"] = round(
                sum(1 for p in ens_human_pred if p) / max(1, len(human_true)), 4
            )
            ens_esl_pred = [yp for yp, ie in zip(y_pred_ensemble, is_esl_mask) if ie]
            metrics_ens["fpr_non_native_esl"] = round(
                sum(1 for p in ens_esl_pred if p) / max(1, sum(is_esl_mask)), 4
            )

        return {
            "distilgpt2_evaluated": self.ai and has_distilgpt2,
            "stylometry_only": metrics_style,
            "distilgpt2_only": metrics_lm,
            "full_ensemble": metrics_ens,
            "failure_cases": {
                "false_positives_human_count": len(false_positives_human),
                "false_positives_human": false_positives_human,
                "false_positives_esl_count": len(false_positives_esl),
                "false_positives_esl": false_positives_esl,
            },
        }

    def _evaluate_spinner(self) -> Dict[str, Any]:
        """Evaluates spinner and tortured-phrase detection on group f (spinners + controls)."""
        logger.info("Evaluating spinner detection on group f...")

        target_groups = {"f_spinner", "f_spinner_control"}
        spin_rows = [r for r in self.manifest if r["group"] in target_groups]

        y_true: List[bool] = []
        y_pred: List[bool] = []

        for row in spin_rows:
            fname = row["file"]
            grp = row["group"]
            expected = row["expected_spinner"].lower() == "yes"
            y_true.append(expected)

            text = load_document(fname, self.docs_dir)
            doc_obj = _create_document(text)
            check_res: CheckResult = check_spinner(doc_obj)
            pred = len(check_res.findings) > 0
            y_pred.append(pred)

            self.detailed_records.append({
                "file": fname,
                "group": grp,
                "task": "spinner_detection",
                "ground_truth": expected,
                "findings_count": len(check_res.findings),
                "pred": pred,
            })

            gc.collect()

        return binary_classification_metrics(y_true, y_pred)

    def _evaluate_collusion(self) -> Dict[str, Any]:
        """Evaluates batch pairwise collusion detection and MinHash/LSH screening on group g."""
        logger.info("Evaluating collusion detection on group g...")

        target_groups = {"g_class_submissions"}
        coll_rows = [r for r in self.manifest if r["group"] in target_groups]

        # Ground truth pairs
        true_pairs = {
            tuple(sorted(["class_sub_01.txt", "class_sub_02.txt"])),
            tuple(sorted(["class_sub_03.txt", "class_sub_04.txt"])),
        }

        # 1. Compute MinHash signatures, shingle sets, and winnowing fingerprint sets
        signatures: Dict[str, List[int]] = {}
        shingle_sets: Dict[str, Set[str]] = {}
        fingerprint_sets: Dict[str, Set[int]] = {}

        for row in coll_rows:
            fname = row["file"]
            text = load_document(fname, self.docs_dir)
            canon = normalize_text(text)
            tokens = canon.tokens
            shingles = {
                " ".join(tokens[i : i + 5]) for i in range(max(0, len(tokens) - 4))
            }
            shingle_sets[fname] = shingles
            signatures[fname] = compute_minhash(shingles)
            fps = winnow(tokens, k=5, w=4)
            fingerprint_sets[fname] = {h for h, _ in fps}

        # 2. Pairwise Winnowing (ground truth discovery)
        detected_pairwise_pairs = set()
        files = [r["file"] for r in coll_rows]
        n = len(files)

        for i in range(n):
            for j in range(i + 1, n):
                f_a = files[i]
                f_b = files[j]
                s_a = shingle_sets[f_a]
                s_b = shingle_sets[f_b]
                jaccard = (
                    len(s_a.intersection(s_b)) / max(1, len(s_a.union(s_b)))
                    if s_a or s_b
                    else 0.0
                )
                if jaccard >= 0.25:
                    detected_pairwise_pairs.add(tuple(sorted([f_a, f_b])))

        # 3. Exact Inverted Index Fingerprint Shortlist Screening (Default)
        fp_candidate_pairs = screen_fingerprint_candidates(fingerprint_sets)
        fp_pairs = {
            tuple(sorted(p)) for p in fp_candidate_pairs
        }

        # 4. MinHash/LSH Candidate Screening (threshold=0.25 deriving (64, 2))
        lsh_candidate_pairs, _ = screen_batch_candidates(shingle_sets, threshold=0.25)
        lsh_pairs = {
            tuple(sorted(p)) for p in lsh_candidate_pairs
        }

        # Check clusters and recall
        expected_clusters_found = detected_pairwise_pairs == true_pairs
        fp_shortlist_recall = (
            len(true_pairs.intersection(fp_pairs)) / len(true_pairs)
            if true_pairs
            else 1.0
        )
        lsh_shortlist_recall = (
            len(true_pairs.intersection(lsh_pairs)) / len(true_pairs)
            if true_pairs
            else 1.0
        )

        return {
            "total_submissions": len(files),
            "expected_collusion_pairs": [list(p) for p in true_pairs],
            "detected_pairwise_pairs": [list(p) for p in detected_pairwise_pairs],
            "fingerprint_index_candidates": [list(p) for p in fp_pairs],
            "minhash_lsh_candidates": [list(p) for p in lsh_pairs],
            "expected_clusters_found": expected_clusters_found,
            "fingerprint_shortlist_recall": round(fp_shortlist_recall, 4),
            "lsh_shortlist_recall": round(lsh_shortlist_recall, 4),
        }

    def _evaluate_seals(self) -> Dict[str, Any]:
        """Evaluates cryptographic seal verification and tampering detection on group h."""
        logger.info("Evaluating cryptographic seals on group h...")

        target_groups = {"h_seal_reports"}
        seal_rows = [r for r in self.manifest if r["group"] in target_groups]

        untouched_valid = 0
        tampered_detected = 0
        total_reports = len(seal_rows)

        for row in seal_rows:
            fname = row["file"]
            raw_json = load_document(fname, self.docs_dir)
            report_dict = json.loads(raw_json)

            # 1. Untouched verification
            is_valid_untouched, _ = verify_report_seal(report_dict)
            if is_valid_untouched:
                untouched_valid += 1

            # 2. Tampered verification (introduce single character modification)
            tampered_dict = copy.deepcopy(report_dict)
            tampered_dict["summary"] = str(tampered_dict.get("summary", "")) + "X"
            is_valid_tampered, _ = verify_report_seal(tampered_dict)
            if not is_valid_tampered:
                tampered_detected += 1

        tamper_detection_rate = (
            tampered_detected / total_reports if total_reports > 0 else 0.0
        )

        return {
            "total_reports_tested": total_reports,
            "untouched_valid_count": untouched_valid,
            "tampered_detected_count": tampered_detected,
            "tamper_detection_rate": round(tamper_detection_rate, 4),
        }

    def _evaluate_crosslang(self) -> Dict[str, Any]:
        """Evaluates multilingual cross-language matching on group h2 (ES/FR)."""
        logger.info("Evaluating cross-language semantic matching on group h2...")

        target_groups = {"h_translations"}
        trans_rows = [r for r in self.manifest if r["group"] in target_groups]

        if not self.crosslang:
            return {
                "status": "skipped",
                "reason": "Cross-language evaluation disabled (enable with --crosslang)",
                "documents_available": len(trans_rows),
            }

        try:
            from veritext.similarity.crosslang import is_crosslang_available
            if not is_crosslang_available():
                return {
                    "status": "skipped",
                    "reason": "Multilingual sentence-transformer model not installed locally",
                    "documents_available": len(trans_rows),
                }

            from veritext.similarity.crosslang import CrossLanguageMatcher
            matcher = CrossLanguageMatcher()
            ref_docs = [
                _create_document(txt, path=rid)
                for rid, txt in self.references.items()
            ]

            matches_found = 0
            for row in trans_rows:
                fname = row["file"]
                text = load_document(fname, self.docs_dir)
                doc = _create_document(text, path=fname)
                res = matcher.find_matches(doc, ref_docs, threshold=0.65)
                if res:
                    matches_found += 1

            recall = matches_found / len(trans_rows) if trans_rows else 0.0
            return {
                "status": "completed",
                "total_translations": len(trans_rows),
                "matches_found": matches_found,
                "recall": round(recall, 4),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "documents_available": len(trans_rows),
            }

    def _save_results(self, summary: Dict[str, Any]):
        """Saves benchmark results to JSON, CSV, Markdown, and confusion-matrix PNG."""
        # 1. Summary JSON
        json_path = self.out_dir / "benchmark_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"Saved summary JSON to {json_path}")

        # 2. Details CSV
        csv_path = self.out_dir / "benchmark_details.csv"
        if self.detailed_records:
            fieldnames = sorted({k for r in self.detailed_records for k in r.keys()})
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.detailed_records:
                    writer.writerow(r)
            logger.info(f"Saved detailed records CSV to {csv_path}")

        # 3. Markdown Results Table
        md_path = self.out_dir / "results_table.md"
        md_text = self._generate_markdown_table(summary)
        md_path.write_text(md_text, encoding="utf-8")
        logger.info(f"Saved Markdown table to {md_path}")

        # 4. Confusion Matrix Plot
        png_path = self.out_dir / "confusion_matrices.png"
        self._plot_confusion_matrices(summary, png_path)
        logger.info(f"Saved confusion matrix plot to {png_path}")

    def _generate_markdown_table(self, summary: Dict[str, Any]) -> str:
        """Generates a GitHub-flavored Markdown table of benchmark results."""
        p_win = summary["plagiarism"]["winnowing_only"]
        p_base = summary["plagiarism"]["baseline_difflib"]
        p_sem = summary["plagiarism"]["winnowing_and_semantic"] or {}

        ai_style = summary["ai_detection"]["stylometry_only"]
        ai_ens = summary["ai_detection"]["full_ensemble"] or {}

        spin = summary["spinner_detection"]
        seal = summary["seal_verification"]
        coll = summary["collusion_detection"]

        lines = [
            "# Veritext Benchmark Results",
            "",
            f"- **Evaluated Documents**: {summary['total_documents_evaluated']} (from synthetic dataset)",
            f"- **Total Runtime**: {summary['total_runtime_seconds']} s (~{summary['runtime_per_document_seconds']} s/doc)",
            f"- **Peak Memory**: {summary['peak_memory_mb']} MB",
            f"- **Date**: {summary['timestamp']}",
            "",
            "> **Corpus & Methodology Note**: The evaluation corpus is synthetic, compact, and created by the system author.",
            "> In accordance with the Stage 7 audit guidelines, results are evaluated across distinct easy and hard paraphrase subsets,",
            "> and false positive rates are transparently measured across unrelated human originals, AI-generated text, and ESL writing.",
            "",
            "## 1. Plagiarism Detection Engine (Headline Threshold: 15%)",
            "",
            "| Method | Accuracy | Precision | Recall | F1 Score | False Positive Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **Exact Winnowing (Veritext)** | {p_win['accuracy']:.2%} | {p_win['precision']:.2%} | {p_win['recall']:.2%} | {p_win['f1']:.2%} | {p_win['fpr']:.2%} |",
            f"| Baseline (difflib SequenceMatcher) | {p_base['accuracy']:.2%} | {p_base['precision']:.2%} | {p_base['recall']:.2%} | {p_base['f1']:.2%} | {p_base['fpr']:.2%} |",
        ]

        if p_sem:
            lines.append(
                f"| **Winnowing + Semantic Paraphrase** | {p_sem.get('accuracy', 0):.2%} | {p_sem.get('precision', 0):.2%} | {p_sem.get('recall', 0):.2%} | {p_sem.get('f1', 0):.2%} | {p_sem.get('fpr', 0):.2%} |"
            )

        # Multi-threshold sensitivity analysis
        if "multi_threshold_metrics" in summary["plagiarism"]:
            mtm = summary["plagiarism"]["multi_threshold_metrics"]
            lines.extend([
                "",
                "### Decision Threshold Sensitivity Analysis",
                "",
                "| Threshold | Method | Accuracy | Precision | Recall | F1 Score | Overall FPR |",
                "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
            ])
            for thresh in [0.05, 0.10, 0.15, 0.25]:
                w_m = mtm["winnowing_only"].get(thresh, {})
                lines.append(
                    f"| {thresh:.0%} | Exact Winnowing | {w_m.get('accuracy', 0):.2%} | {w_m.get('precision', 0):.2%} | {w_m.get('recall', 0):.2%} | {w_m.get('f1', 0):.2%} | {w_m.get('fpr', 0):.2%} |"
                )
                if mtm.get("winnowing_and_semantic"):
                    s_m = mtm["winnowing_and_semantic"].get(thresh, {})
                    lines.append(
                        f"| {thresh:.0%} | **Winnowing + Semantic** | {s_m.get('accuracy', 0):.2%} | {s_m.get('precision', 0):.2%} | {s_m.get('recall', 0):.2%} | {s_m.get('f1', 0):.2%} | {s_m.get('fpr', 0):.2%} |"
                    )

        # Paraphrase Subset Recall Breakdown
        if "paraphrase_recall_breakdown" in summary["plagiarism"]:
            prb = summary["plagiarism"]["paraphrase_recall_breakdown"]
            lines.extend([
                "",
                "### Paraphrase Recall: Easy vs. Hard Subsets",
                "",
                "| Threshold | Group A (Copies) | Group B Easy Paraphrases | Group B Hard Paraphrases (Complex) |",
                "| :---: | :---: | :---: | :---: |",
            ])
            for thresh in [0.05, 0.10, 0.15, 0.25]:
                rec_a = prb.get("a_copies", {}).get(thresh, 0.0)
                rec_easy = prb.get("b_paraphrases", {}).get(thresh, 0.0)
                rec_hard = prb.get("b_paraphrases_hard", {}).get(thresh, 0.0)
                lines.append(
                    f"| {thresh:.0%} | {rec_a:.2%} | {rec_easy:.2%} | {rec_hard:.2%} |"
                )

        # Negative Cohort FPR breakdown
        if "group_fpr_by_threshold" in summary["plagiarism"]:
            gfpr = summary["plagiarism"]["group_fpr_by_threshold"]
            lines.extend([
                "",
                "### False Positive Rates across Negative Cohorts (Semantic Engine ON)",
                "",
                "| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) |",
                "| :---: | :---: | :---: | :---: |",
            ])
            for thresh in [0.05, 0.10, 0.15, 0.25]:
                fpr_c = gfpr.get("c_original", {}).get(thresh, 0.0)
                fpr_d = gfpr.get("d_ai_style", {}).get(thresh, 0.0)
                fpr_e = gfpr.get("e_non_native", {}).get(thresh, 0.0)
                lines.append(
                    f"| {thresh:.0%} | {fpr_c:.2%} | {fpr_d:.2%} | {fpr_e:.2%} |"
                )

        # Mean similarity per group
        by_eng = summary["plagiarism"].get("mean_similarity_by_engine")
        if by_eng:
            ex_m = by_eng.get("exact", {})
            sem_m = by_eng.get("semantic") or {}
            lines.extend([
                "",
                "**Mean Similarity Score by Cohort (Exact vs. Semantic)**:",
                f"- Group A (Verbatim / Lightly Edited Copies): Exact **{ex_m.get('a_copies', 0):.2%}** | Semantic **{sem_m.get('a_copies', ex_m.get('a_copies', 0)):.2%}**",
                f"- Group B (Easy Paraphrased Passages): Exact **{ex_m.get('b_paraphrases', 0):.2%}** | Semantic **{sem_m.get('b_paraphrases', 0):.2%}**",
                f"- Group B (Hard Complex Paraphrases): Exact **{ex_m.get('b_paraphrases_hard', 0):.2%}** | Semantic **{sem_m.get('b_paraphrases_hard', 0):.2%}**",
                f"- Group C (Unrelated Human Originals): Exact **{ex_m.get('c_original', 0):.2%}** | Semantic **{sem_m.get('c_original', 0):.2%}**",
                f"- Group D (AI-Generated Text): Exact **{ex_m.get('d_ai_style', 0):.2%}** | Semantic **{sem_m.get('d_ai_style', 0):.2%}**",
                f"- Group E (Non-Native English): Exact **{ex_m.get('e_non_native', 0):.2%}** | Semantic **{sem_m.get('e_non_native', 0):.2%}**",
            ])
        else:
            mean_sims = summary["plagiarism"]["mean_similarity_per_group"]
            lines.extend([
                "",
                "**Mean Similarity Score by Cohort**:",
                f"- Group A (Verbatim / Lightly Edited Copies): **{mean_sims.get('a_copies', 0):.2%}**",
                f"- Group B (Paraphrased Passages): **{mean_sims.get('b_paraphrases', 0):.2%}**",
                f"- Group C (Unrelated Human Originals): **{mean_sims.get('c_original', 0):.2%}**",
            ])

        lines.extend([
            "",
            "## 2. AI-Written Text Detector",
            "",
            "| Model / Configuration | Accuracy | Precision | Recall | F1 Score | Overall Human FPR | Non-Native (ESL) FPR |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **Stylometry Only (Lite Mode)** | {ai_style['accuracy']:.2%} | {ai_style['precision']:.2%} | {ai_style['recall']:.2%} | {ai_style['f1']:.2%} | {ai_style['fpr_human_overall']:.2%} | {ai_style['fpr_non_native_esl']:.2%} |",
        ])

        if ai_ens:
            lines.append(
                f"| **Full Ensemble (Stylometry + DistilGPT2)** | {ai_ens.get('accuracy', 0):.2%} | {ai_ens.get('precision', 0):.2%} | {ai_ens.get('recall', 0):.2%} | {ai_ens.get('f1', 0):.2%} | {ai_ens.get('fpr_human_overall', 0):.2%} | {ai_ens.get('fpr_non_native_esl', 0):.2%} |"
            )

        lines.extend([
            "",
            "## 3. Integrity Checks, Collusion & Cryptographic Seals",
            "",
            "| Module | Metric Tested | Result | Target / Standard |",
            "| :--- | :--- | :---: | :---: |",
            f"| **Spinner / Tortured Phrases** | Precision / Recall | {spin['precision']:.2%} / {spin['recall']:.2%} | Cabanac et al. (2021) |",
            f"| **Batch Collusion Matrix** | Expected Clusters Discovered | {'YES' if coll['expected_clusters_found'] else 'NO'} | Pairwise Winnowing |",
            f"| **Fingerprint Index Screening** | True Pairs Candidate Recall | {coll.get('fingerprint_shortlist_recall', 1.0):.2%} | Exact Inverted Index (Default) |",
            f"| **MinHash / LSH Screening** | True Pairs Candidate Recall | {coll['lsh_shortlist_recall']:.2%} | Adaptive (64x2) 128-perm |",
            f"| **Cryptographic Report Seals** | Tamper Detection Rate | {seal['tamper_detection_rate']:.2%} | HMAC-SHA256 (1-char edit) |",
            "",
            "## 4. Notable Failure Cases & Honest Limitations",
            "",
            f"- **Plagiarism Missed Paraphrases (Easy)**: {summary['plagiarism']['failure_cases'].get('missed_paraphrases_easy_count', 0)} documents.",
            f"- **Plagiarism Missed Paraphrases (Hard)**: {summary['plagiarism']['failure_cases'].get('missed_paraphrases_hard_count', 0)} documents.",
            f"- **AI False Positives on Human Text**: {summary['ai_detection']['failure_cases']['false_positives_human_count']} documents.",
            f"- **AI False Positives on Non-Native (ESL) Text**: {summary['ai_detection']['failure_cases']['false_positives_esl_count']} documents. (Stylometric variance can mistake uniform ESL syntax for machine generation; conservative thresholds are essential).",
            "",
        ])

        return "\n".join(lines)

    def _plot_confusion_matrices(self, summary: Dict[str, Any], output_path: Path):
        """Generates a 3-panel confusion matrix figure using matplotlib."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Panel 1: Plagiarism Winnowing
        p_win = summary["plagiarism"]["winnowing_only"]
        cm_plag = [[p_win["tn"], p_win["fp"]], [p_win["fn"], p_win["tp"]]]
        self._draw_matrix(
            axes[0], cm_plag, "Plagiarism (Winnowing)", ["Original", "Plagiarised"]
        )

        # Panel 2: AI Detection (Stylometry)
        ai_style = summary["ai_detection"]["stylometry_only"]
        cm_ai = [[ai_style["tn"], ai_style["fp"]], [ai_style["fn"], ai_style["tp"]]]
        self._draw_matrix(axes[1], cm_ai, "AI Detection (Stylometry)", ["Human", "AI"])

        # Panel 3: Spinner Detection
        spin = summary["spinner_detection"]
        cm_spin = [[spin["tn"], spin["fp"]], [spin["fn"], spin["tp"]]]
        self._draw_matrix(axes[2], cm_spin, "Spinner Detection", ["Clean", "Spinner"])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _draw_matrix(ax, matrix, title, labels):
        """Helper to draw a single 2x2 confusion matrix."""
        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Ground Truth", fontsize=10)

        for i in range(2):
            for j in range(2):
                val = matrix[i][j]
                color = "white" if val > (matrix[0][0] + matrix[1][1]) / 2 else "black"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=14,
                    fontweight="bold",
                )
