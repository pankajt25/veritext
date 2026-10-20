"""CLI for AI-written text detection, model downloading, and ensemble calibration: python -m veritext.ai_detect"""

import argparse
from pathlib import Path
import sys

from veritext.ai_detect.calibrate import calibrate_ensemble
from veritext.ai_detect.detector import detect_ai
from veritext.ai_detect.features_lm import download_distilgpt2, is_distilgpt2_available
from veritext.ai_detect.models import DISCLAIMER_TEXT
from veritext.ai_detect.roberta import download_roberta, is_roberta_available
from veritext.ingest.pipeline import load_document


def handle_download_models(args):
    """Downloads required models explicitly upon user request."""
    print("=" * 70)
    print("Veritext AI Model Downloader")
    print("=" * 70)

    print("Downloading distilgpt2 causal language model...")
    gpt_path = download_distilgpt2()
    print(f"distilgpt2 successfully saved to: {gpt_path}")

    if args.roberta:
        print("\nDownloading opt-in RoBERTa classifier (Hello-SimpleAI/chatgpt-detector-roberta)...")
        roberta_path = download_roberta()
        print(f"RoBERTa classifier successfully saved to: {roberta_path}")

    print("\nModel setup complete! AI text detection can now run in full neural mode.")


def handle_calibrate(args):
    """Calibrates ensemble weights on a labeled CSV dataset."""
    csv_file = Path(args.calibrate).resolve()
    if not csv_file.exists():
        print(f"Error: Calibration CSV not found: {csv_file}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Veritext AI Detection Ensemble Calibration")
    print("=" * 70)
    print(f"Dataset: {csv_file.name}")
    print("Fitting logistic regression models...")

    try:
        report = calibrate_ensemble(csv_file)
    except Exception as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nCalibration Results Summary:")
    print(f"  Samples:      {report['samples_count']} ({report['class_distribution']['human']} Human, {report['class_distribution']['ai']} AI)")
    print(f"  Stylometry:   Accuracy={report['stylometry_model']['accuracy']:.3f}, Precision={report['stylometry_model']['precision']:.3f}, Recall={report['stylometry_model']['recall']:.3f}, F1={report['stylometry_model']['f1']:.3f}")
    print(f"  False Pos %:  {report['stylometry_model']['false_positive_rate'] * 100:.1f}%")
    if report.get("full_model"):
        fm = report["full_model"]
        print(f"  Full Model:   Accuracy={fm['accuracy']:.3f}, Precision={fm['precision']:.3f}, Recall={fm['recall']:.3f}, F1={fm['f1']:.3f}")
    print(f"\nCalibrated weights saved to configuration.")


def handle_detect(args):
    """Executes AI detection on a document file."""
    doc_path = Path(args.file).resolve()
    if not doc_path.exists():
        print(f"Error: File not found: {doc_path}", file=sys.stderr)
        sys.exit(1)

    try:
        doc = load_document(doc_path)
    except Exception as e:
        print(f"Error loading document: {e}", file=sys.stderr)
        sys.exit(1)

    has_gpt = is_distilgpt2_available()

    print("=" * 70)
    print("Veritext AI-Written Text Analysis")
    print("=" * 70)
    print(f"Document: {doc_path.name}")
    print(f"Words:    {len(doc.clean_text.split())}")
    print(f"Engine:   {'Full Neural (distilgpt2 + Stylometry)' if has_gpt else 'Stylometry-Only Fallback (reduced confidence)'}")
    if not has_gpt:
        print("Note:     Neural model weights not found locally. To download: python -m veritext.ai_detect --download-models")
    print("-" * 70)

    result = detect_ai(
        document=doc,
        use_roberta=args.use_roberta,
        max_words=args.max_words,
    )

    if result.status == "insufficient_text":
        print(f"\nResult: INSUFFICIENT TEXT ({result.word_count} words)")
        print(result.explanations[0])
        print("\n" + "=" * 70)
        print(result.disclaimer)
        print("=" * 70)
        return

    print(f"\nOVERALL AI PROBABILITY: {result.overall_probability:.1f}%")
    print(f"Confidence Band:        {result.confidence_band.upper()}")
    if result.reduced_confidence:
        print("Reduced Confidence:     YES (Operating on stylometry alone without language model)")

    print("\nKey Signal Drivers:")
    for exp in result.explanations[:5]:
        print(f"  • {exp}")

    print("\nMost AI-Like Sentences:")
    top_sentences = sorted(result.sentences, key=lambda s: s.score, reverse=True)[:5]

    for idx, sent in enumerate(top_sentences, 1):
        full_text = sent.text.strip().replace("\n", " ")
        print(f"  {idx}. [{sent.label.upper()} - Score: {sent.score:.2f}] (Offsets: {sent.doc_start}:{sent.doc_end})")
        print(f"     \"{full_text}\"")

    print("\n" + "=" * 70)
    print(result.disclaimer)
    print("=" * 70)

    if args.json_output:
        out_path = Path(args.json_output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.to_json(), encoding="utf-8")
        print(f"\nSaved complete JSON analysis report to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m veritext.ai_detect",
        description="Inspect document for AI-generated text patterns using stylometry and perplexity",
    )

    parser.add_argument("file", nargs="?", help="Path to document to analyze")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download required open model weights (distilgpt2) locally",
    )
    parser.add_argument(
        "--roberta",
        action="store_true",
        help="Also download opt-in RoBERTa classifier with --download-models",
    )
    parser.add_argument(
        "--use-roberta",
        action="store_true",
        help="Enable opt-in RoBERTa classifier during document analysis",
    )
    parser.add_argument(
        "--calibrate",
        metavar="CSV_FILE",
        help="Calibrate ensemble weights on a labeled CSV dataset (text,label)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=5000,
        help="Maximum word count to evaluate (default: 5000 words)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        help="Optional file path to output complete analysis in JSON format",
    )

    args = parser.parse_args()

    if args.download_models:
        handle_download_models(args)
    elif args.calibrate:
        handle_calibrate(args)
    elif args.file:
        handle_detect(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
