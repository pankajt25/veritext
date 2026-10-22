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
)
from veritext.fingerprint.winnow import winnow
from veritext.ingest.models import Document
from veritext.normalize.normalize import normalize_text
from veritext.report.seal import verify_report_seal
from veritext.similarity.exact import find_exact_matches
from veritext.similarity.models import MatchSpan
from veritext.similarity.scorer import compute_similarity_score
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "eval" / "results"


def _create_document(text: str, path: Optional[str] = None) -> Document:
    """Helper to instantiate a Document for evaluation checks."""
    return Document(
        raw_text=text,
        clean_text=text,
        sentences=[],
        paragraphs=[],
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
                [r for r in self.manifest if r["group"] in ["a_copies", "b_paraphrases", "c_original"]],
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
        """Evaluates plagiarism detection across groups a (copies), b (paraphrases), and c (originals)."""
        logger.info("Evaluating plagiarism detection on groups a, b, c...")

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

        target_groups = {"a_copies", "b_paraphrases", "c_original"}
        plag_rows = [r for r in self.manifest if r["group"] in target_groups]

        y_true: List[bool] = []
        y_pred_winnow: List[bool] = []
        y_pred_baseline: List[bool] = []
        y_pred_semantic: List[bool] = []

        scores_by_group: Dict[str, List[float]] = {
            "a_copies": [],
            "b_paraphrases": [],
            "c_original": [],
        }

        missed_paraphrases: List[Dict[str, Any]] = []
        false_positives: List[Dict[str, Any]] = []

        SIM_THRESHOLD = 0.15  # 15% similarity threshold

        for row in plag_rows:
            fname = row["file"]
            grp = row["group"]
            expected = row["expected_plagiarism"].lower() == "yes"
            y_true.append(expected)

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
            pred_winnow = score_winnow >= SIM_THRESHOLD
            y_pred_winnow.append(pred_winnow)
            scores_by_group[grp].append(score_winnow)

            # 2. Difflib SequenceMatcher Baseline
            base_score = max(
                (difflib.SequenceMatcher(None, text, ref_txt).ratio() for ref_txt in self.references.values()),
                default=0.0,
            )
            pred_base = base_score >= 0.35  # SequenceMatcher standard threshold
            y_pred_baseline.append(pred_base)

            # 3. Winnowing + Semantic (if enabled)
            score_sem = score_winnow
            pred_sem = pred_winnow
            if self.semantic:
                try:
                    from veritext.similarity.semantic import find_semantic_matches
                    doc_obj = _create_document(text)
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
                    pred_sem = score_sem >= SIM_THRESHOLD
                except Exception as e:
                    logger.warning(f"Semantic match skipped for {fname}: {e}")
            y_pred_semantic.append(pred_sem)

            # Track failure cases
            if expected and not pred_winnow:
                missed_paraphrases.append({
                    "file": fname,
                    "group": grp,
                    "winnow_score": round(score_winnow, 4),
                    "baseline_score": round(base_score, 4),
                    "notes": row.get("notes", ""),
                })
            if not expected and pred_winnow:
                false_positives.append({
                    "file": fname,
                    "group": grp,
                    "winnow_score": round(score_winnow, 4),
                    "notes": row.get("notes", ""),
                })

            self.detailed_records.append({
                "file": fname,
                "group": grp,
                "task": "plagiarism",
                "ground_truth": expected,
                "winnow_score": round(score_winnow, 4),
                "winnow_pred": pred_winnow,
                "baseline_score": round(base_score, 4),
                "baseline_pred": pred_base,
                "semantic_score": round(score_sem, 4) if self.semantic else None,
                "semantic_pred": pred_sem if self.semantic else None,
            })

            gc.collect()

        metrics_winnow = binary_classification_metrics(y_true, y_pred_winnow)
        metrics_baseline = binary_classification_metrics(y_true, y_pred_baseline)
        metrics_semantic = (
            binary_classification_metrics(y_true, y_pred_semantic) if self.semantic else None
        )

        return {
            "threshold": SIM_THRESHOLD,
            "mean_similarity_per_group": {
                grp: mean(scs) for grp, scs in scores_by_group.items()
            },
            "winnowing_only": metrics_winnow,
            "baseline_difflib": metrics_baseline,
            "winnowing_and_semantic": metrics_semantic,
            "failure_cases": {
                "missed_paraphrases_count": len(missed_paraphrases),
                "missed_paraphrases": missed_paraphrases,
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

        # 1. Compute MinHash signatures and shingle sets
        signatures: Dict[str, List[int]] = {}
        shingle_sets: Dict[str, Set[str]] = {}

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

        # 3. MinHash/LSH Candidate Screening
        lsh_candidate_pairs, _ = screen_batch_candidates(shingle_sets)
        lsh_pairs = {
            tuple(sorted(p)) for p in lsh_candidate_pairs
        }

        # Check clusters and recall
        expected_clusters_found = detected_pairwise_pairs == true_pairs
        lsh_shortlist_recall = (
            len(true_pairs.intersection(lsh_pairs)) / len(true_pairs)
            if true_pairs
            else 1.0
        )

        return {
            "total_submissions": len(files),
            "expected_collusion_pairs": [list(p) for p in true_pairs],
            "detected_pairwise_pairs": [list(p) for p in detected_pairwise_pairs],
            "minhash_lsh_candidates": [list(p) for p in lsh_pairs],
            "expected_clusters_found": expected_clusters_found,
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
            "## 1. Plagiarism Detection Engine",
            "",
            "| Method | Accuracy | Precision | Recall | F1 Score | False Positive Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **Exact Winnowing (Veritext)** | {p_win['accuracy']:.2%} | {p_win['precision']:.2%} | {p_win['recall']:.2%} | {p_win['f1']:.2%} | {p_win['fpr']:.2%} |",
            f"| Baseline (difflib SequenceMatcher) | {p_base['accuracy']:.2%} | {p_base['precision']:.2%} | {p_base['recall']:.2%} | {p_base['f1']:.2%} | {p_base['fpr']:.2%} |",
        ]

        if p_sem:
            lines.append(
                f"| Winnowing + Semantic Paraphrase | {p_sem.get('accuracy', 0):.2%} | {p_sem.get('precision', 0):.2%} | {p_sem.get('recall', 0):.2%} | {p_sem.get('f1', 0):.2%} | {p_sem.get('fpr', 0):.2%} |"
            )

        mean_sims = summary["plagiarism"]["mean_similarity_per_group"]
        lines.extend([
            "",
            "**Mean Similarity Score by Cohort**:",
            f"- Group A (Verbatim / Lightly Edited Copies): **{mean_sims.get('a_copies', 0):.2%}**",
            f"- Group B (Paraphrased Passages): **{mean_sims.get('b_paraphrases', 0):.2%}**",
            f"- Group C (Unrelated Human Originals): **{mean_sims.get('c_original', 0):.2%}**",
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
            f"| **MinHash / LSH Screening** | True Pairs Candidate Recall | {coll['lsh_shortlist_recall']:.2%} | Carter-Wegman 128-perm |",
            f"| **Cryptographic Report Seals** | Tamper Detection Rate | {seal['tamper_detection_rate']:.2%} | HMAC-SHA256 (1-char edit) |",
            "",
            "## 4. Notable Failure Cases & Honest Limitations",
            "",
            f"- **Plagiarism Missed Paraphrases**: {summary['plagiarism']['failure_cases']['missed_paraphrases_count']} documents. (Exact winnowing with k=5 relies on identical token n-grams; heavily rephrased text with 0 shared 5-grams requires semantic vector matching).",
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
