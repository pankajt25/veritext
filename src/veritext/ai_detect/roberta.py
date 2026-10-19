"""Opt-in RoBERTa-based AI classifier wrapper (Hello-SimpleAI/chatgpt-detector-roberta)."""

import gc
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROBERTA_MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
DEFAULT_LOCAL_ROBERTA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "models"
    / "roberta"
)


def is_roberta_available(local_dir: Optional[Path] = None) -> bool:
    """Checks whether RoBERTa weights are available locally without network calls."""
    target_dir = local_dir or DEFAULT_LOCAL_ROBERTA_DIR
    if target_dir.exists() and (
        any(target_dir.glob("*.safetensors")) or any(target_dir.glob("*.bin"))
    ):
        return True

    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME, local_files_only=True)
        return True
    except Exception:
        return False


def download_roberta(target_dir: Optional[Path] = None) -> Path:
    """Explicitly downloads the RoBERTa classifier into local directory."""
    dest = target_dir or DEFAULT_LOCAL_ROBERTA_DIR
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {ROBERTA_MODEL_NAME} to {dest}...")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)

    tokenizer.save_pretrained(str(dest))
    model.save_pretrained(str(dest))
    logger.info(f"Successfully saved {ROBERTA_MODEL_NAME} to {dest}")
    return dest


def classify_roberta(
    text: str,
    local_dir: Optional[Path] = None,
    max_chunk_tokens: int = 512,
    stride: int = 448,
) -> Optional[float]:
    """
    Classifies text using RoBERTa detector under Lite Mode constraints.
    Returns AI probability (0.0 to 1.0), or None if unavailable.
    """
    if not text.strip() or not is_roberta_available(local_dir=local_dir):
        return None

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)

    target_dir = local_dir or DEFAULT_LOCAL_ROBERTA_DIR
    model_source = (
        str(target_dir)
        if (target_dir.exists() and any(target_dir.iterdir()))
        else ROBERTA_MODEL_NAME
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source, local_files_only=True
        )
        model.eval()

        chunk_probs = []
        with torch.inference_mode():
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
            )
            input_ids = encodings.input_ids
            seq_len = input_ids.size(1)

            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_chunk_tokens, seq_len)
                chunk_ids = input_ids[:, begin_loc:end_loc]
                outputs = model(chunk_ids)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                # Index 1 corresponds to ChatGPT / AI class in Hello-SimpleAI/chatgpt-detector-roberta
                ai_prob = float(probs[1].item())
                chunk_probs.append(ai_prob)
                if end_loc == seq_len:
                    break

        return sum(chunk_probs) / len(chunk_probs) if chunk_probs else 0.5

    except Exception as e:
        logger.warning(f"Error during RoBERTa classification: {e}")
        return None

    finally:
        if "model" in locals():
            del model
        if "tokenizer" in locals():
            del tokenizer
        gc.collect()
