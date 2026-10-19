"""Language model feature extraction using distilgpt2 (perplexity, burstiness, log-rank)."""

import gc
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "distilgpt2"
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "models" / "distilgpt2"


def is_distilgpt2_available(local_dir: Optional[Path] = None) -> bool:
    """Checks whether distilgpt2 weights are available locally without hitting the network."""
    target_dir = local_dir or DEFAULT_LOCAL_MODEL_DIR
    if target_dir.exists() and any(target_dir.glob("*.safetensors")) or any(target_dir.glob("*.bin")):
        return True

    # Check local Hugging Face cache
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, local_files_only=True)
        return True
    except Exception:
        return False


def download_distilgpt2(target_dir: Optional[Path] = None) -> Path:
    """Downloads distilgpt2 tokenizer and model into local directory."""
    dest = target_dir or DEFAULT_LOCAL_MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {DEFAULT_MODEL_NAME} to {dest}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME)

    tokenizer.save_pretrained(str(dest))
    model.save_pretrained(str(dest))
    logger.info(f"Successfully saved {DEFAULT_MODEL_NAME} to {dest}")
    return dest


def load_distilgpt2(local_dir: Optional[Path] = None):
    """
    Lazily loads distilgpt2 model and tokenizer under Lite Mode constraints:
    - torch.set_num_threads(2)
    - local_files_only
    - CPU inference
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)

    target_dir = local_dir or DEFAULT_LOCAL_MODEL_DIR
    model_source = str(target_dir) if (target_dir.exists() and any(target_dir.iterdir())) else DEFAULT_MODEL_NAME

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True)
    except Exception:
        # Fallback to model name if stored in default HF cache
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME, local_files_only=True)

    model.eval()
    return tokenizer, model


def compute_lm_features(
    text: str,
    sentences: Optional[List[str]] = None,
    tokenizer=None,
    model=None,
    max_chunk_tokens: int = 512,
    stride: int = 448,  # 512 max - 64 overlap
) -> Dict[str, Any]:
    """
    Computes language model features using distilgpt2 under Lite Mode:
    - Perplexity (whole document)
    - Mean token log-rank
    - Burstiness (std dev of per-sentence perplexities)
    - Per-sentence perplexity values
    """
    import torch

    torch.set_num_threads(2)

    if not text.strip():
        return {
            "perplexity": 0.0,
            "mean_log_rank": 0.0,
            "burstiness": 0.0,
            "sentence_perplexities": [],
        }

    cleanup_model = False
    if tokenizer is None or model is None:
        tokenizer, model = load_distilgpt2()
        cleanup_model = True

    try:
        with torch.inference_mode():
            # 1. Whole Document Perplexity and Token Log-Ranks using Strided Chunks
            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids
            seq_len = input_ids.size(1)

            if seq_len <= 1:
                return {
                    "perplexity": 1000.0,
                    "mean_log_rank": 10.0,
                    "burstiness": 0.0,
                    "sentence_perplexities": [],
                }

            nlls: List[float] = []
            log_ranks: List[float] = []

            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_chunk_tokens, seq_len)
                trg_len = end_loc - prev_end_loc
                chunk_input_ids = input_ids[:, begin_loc:end_loc]

                target_ids = chunk_input_ids.clone()
                # Mask out tokens in overlap region that were already evaluated
                target_ids[:, :-trg_len] = -100

                outputs = model(chunk_input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

                logits = outputs.logits  # (1, chunk_len, vocab_size)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = chunk_input_ids[..., 1:].contiguous()

                # Calculate per-token log-rank for the target slice
                slice_offset = chunk_input_ids.size(1) - trg_len
                for t_idx in range(slice_offset, shift_labels.size(1)):
                    target_token = shift_labels[0, t_idx].item()
                    step_logits = shift_logits[0, t_idx]
                    target_logit = step_logits[target_token].item()
                    # Count tokens with higher logit
                    rank = (step_logits > target_logit).sum().item() + 1
                    log_ranks.append(math.log(rank))

                nlls.append(neg_log_likelihood.item() * trg_len)
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            total_evaluated_tokens = max(1, sum(1 for _ in log_ranks))
            doc_ppl = math.exp(min(20.0, sum(nlls) / total_evaluated_tokens))
            mean_log_rank = sum(log_ranks) / len(log_ranks) if log_ranks else 0.0

            # 2. Per-Sentence Perplexity and Burstiness
            s_list = sentences or [text]
            sentence_perplexities: List[float] = []

            for s in s_list:
                s_strip = s.strip()
                if len(s_strip.split()) < 3:
                    continue
                s_enc = tokenizer(s_strip, return_tensors="pt")
                s_ids = s_enc.input_ids
                if s_ids.size(1) <= 1:
                    continue
                s_out = model(s_ids, labels=s_ids)
                s_loss = s_out.loss.item()
                s_ppl = math.exp(min(20.0, s_loss))
                sentence_perplexities.append(s_ppl)

            # Burstiness = Standard Deviation of Sentence Perplexities
            if len(sentence_perplexities) > 1:
                ppl_mean = sum(sentence_perplexities) / len(sentence_perplexities)
                burstiness = math.sqrt(
                    sum((p - ppl_mean) ** 2 for p in sentence_perplexities)
                    / len(sentence_perplexities)
                )
            else:
                burstiness = 0.0

            return {
                "perplexity": doc_ppl,
                "mean_log_rank": mean_log_rank,
                "burstiness": burstiness,
                "sentence_perplexities": sentence_perplexities,
            }

    finally:
        if cleanup_model:
            del model
            del tokenizer
            gc.collect()
