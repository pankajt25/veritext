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
    progress_callback=None,
    cancel_check=None,
) -> Dict[str, Any]:
    """
    Computes language model features using distilgpt2 under Lite Mode:
    - Perplexity (whole document) via memory-safe 512-token chunks with 64-token overlap
    - Mean token log-rank
    - Burstiness (std dev of per-sentence perplexities)
    - Per-sentence perplexity values
    Tokenizes per-chunk to eliminate tokenizer sequence length warnings and prevent memory bloat.
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
            words = text.split()
            total_words = len(words)

            if total_words <= 1:
                return {
                    "perplexity": 1000.0,
                    "mean_log_rank": 10.0,
                    "burstiness": 0.0,
                    "sentence_perplexities": [],
                }

            # Words per chunk estimate: ~380 words produce ~490-512 tokens in distilgpt2
            # Stride words: ~332 words (overlap ~48 words ≈ 64 tokens)
            chunk_word_size = max(20, int(max_chunk_tokens / 1.35))
            stride_word_size = max(10, int(stride / 1.35))

            word_starts = list(range(0, total_words, stride_word_size))
            if not word_starts:
                word_starts = [0]
            total_chunks = len(word_starts)

            total_nll = 0.0
            total_evaluated_tokens = 0
            total_log_rank = 0.0
            count_log_rank = 0

            for chunk_idx, w_start in enumerate(word_starts):
                if cancel_check and cancel_check():
                    logger.info("AI detection cancelled by user during LM chunking.")
                    break

                if progress_callback:
                    progress_callback(f"AI detection: chunk {chunk_idx + 1} of {total_chunks}")

                w_end = min(total_words, w_start + chunk_word_size)
                chunk_str = " ".join(words[w_start:w_end])

                # Tokenize ONLY this chunk with max_length=512, truncation=True
                # This guarantees no sequence length warnings and strict <= 512 token limit
                chunk_enc = tokenizer(
                    chunk_str,
                    return_tensors="pt",
                    max_length=max_chunk_tokens,
                    truncation=True,
                )
                chunk_input_ids = chunk_enc.input_ids
                chunk_len = chunk_input_ids.size(1)

                if chunk_len <= 1:
                    continue

                if chunk_idx == 0:
                    trg_len = chunk_len
                else:
                    overlap_tokens = min(64, chunk_len // 2)
                    trg_len = max(1, chunk_len - overlap_tokens)

                target_ids = chunk_input_ids.clone()
                # Mask out tokens in overlap region already evaluated
                if trg_len < chunk_len:
                    target_ids[:, :-trg_len] = -100

                outputs = model(chunk_input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = chunk_input_ids[..., 1:].contiguous()

                # Calculate per-token log-rank for the target slice
                slice_offset = chunk_len - trg_len
                for t_idx in range(slice_offset, shift_labels.size(1)):
                    target_token = shift_labels[0, t_idx].item()
                    step_logits = shift_logits[0, t_idx]
                    target_logit = step_logits[target_token].item()
                    rank = (step_logits > target_logit).sum().item() + 1
                    total_log_rank += math.log(rank)
                    count_log_rank += 1

                total_nll += neg_log_likelihood.item() * trg_len
                total_evaluated_tokens += trg_len

                del chunk_enc, chunk_input_ids, target_ids, outputs, logits, shift_logits, shift_labels
                if (chunk_idx + 1) % 5 == 0:
                    gc.collect()

                if w_end >= total_words:
                    break

            doc_ppl = (
                math.exp(min(20.0, total_nll / max(1, total_evaluated_tokens)))
                if total_evaluated_tokens > 0
                else 1000.0
            )
            mean_log_rank = (
                total_log_rank / max(1, count_log_rank)
                if count_log_rank > 0
                else 0.0
            )

            # 2. Per-Sentence Perplexity and Burstiness
            s_list = sentences or [text]
            sentence_perplexities: List[float] = []

            for s_idx, s in enumerate(s_list):
                if cancel_check and cancel_check():
                    break
                s_strip = s.strip()
                if len(s_strip.split()) < 3:
                    continue
                s_enc = tokenizer(
                    s_strip,
                    return_tensors="pt",
                    max_length=max_chunk_tokens,
                    truncation=True,
                )
                s_ids = s_enc.input_ids
                if s_ids.size(1) <= 1:
                    continue
                s_out = model(s_ids, labels=s_ids)
                s_loss = s_out.loss.item()
                s_ppl = math.exp(min(20.0, s_loss))
                sentence_perplexities.append(s_ppl)

                del s_enc, s_ids, s_out
                if (s_idx + 1) % 50 == 0:
                    gc.collect()

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
