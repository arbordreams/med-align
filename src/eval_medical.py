"""
Evaluation hooks for the medical TokAlign pipeline.

The CLI can run perplexity-style evaluations on Hugging Face datasets or emit a
placeholder log indicating which datasets need to be provided.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Sequence, Dict, Any, Tuple, Optional
import statistics as _stats

import datasets
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATASETS: List[str] = ["uiyunkim-hub/pubmed-abstract:train"]

logger = logging.getLogger(__name__)


def parse_dataset_spec(spec: str) -> Tuple[str, Optional[str], str]:
    """
    Parse dataset strings of the form "dataset[config]:split".
    The config segment is optional; split defaults to "test".
    """
    dataset_part = spec
    split = "test"
    if ":" in spec:
        dataset_part, split = spec.split(":", maxsplit=1)
    dataset_part = dataset_part.strip()
    split = split.strip() or "test"

    config: Optional[str] = None
    if "[" in dataset_part and dataset_part.endswith("]"):
        name_part, config_part = dataset_part[:-1].split("[", maxsplit=1)
        dataset_part = name_part.strip()
        config = config_part.strip()

    if not dataset_part:
        raise ValueError(f"Invalid dataset specification '{spec}' (missing name).")
    return dataset_part, (config or None), split


def evaluate_perplexity(
    model_path: str,
    tokenizer_path: str,
    dataset_name: str,
    split: str,
    dataset_config: Optional[str] = None,
    max_samples: int | None = None,
    batch_size: int = 32,
    max_length: int = 1024,
) -> float:
    """
    Compute a simple perplexity metric over the requested dataset split.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for eval")

    # Enable TF32 for improved throughput on Hopper (H100)
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    try:
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0

    # Load model with robustness and validate with a dummy forward pass to catch device-side asserts early.
    def _load_and_smoke_test(torch_dtype: torch.dtype, use_flash: bool) -> AutoModelForCausalLM | None:
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if use_flash and hasattr(cfg, "attn_implementation"):
                setattr(cfg, "attn_implementation", "flash_attention_2")
            model_local = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=cfg if use_flash else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            try:
                model_local = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
            except Exception:
                return None
        try:
            model_local.to("cuda").eval()
            # Dummy forward to detect device-side asserts early
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype if torch_dtype != torch.float32 else torch.float32):
                test_ids = tokenizer("hello world", return_tensors="pt", truncation=True, max_length=16)["input_ids"]
                # Map any OOB ids to eos as safety
                embed_size = model_local.get_input_embeddings().num_embeddings
                oob_mask = (test_ids < 0) | (test_ids >= embed_size)
                if oob_mask.any():
                    test_ids = test_ids.masked_fill(oob_mask, eos_id)
                test_ids = test_ids.to("cuda", non_blocking=True)
                _ = model_local(test_ids, labels=test_ids)
            return model_local
        except Exception:
            try:
                del model_local  # type: ignore[unreachable]
            except Exception:
                pass
            torch.cuda.empty_cache()
            return None

    # Try progressively less aggressive configs: bf16+flash_attn2 -> fp16 no-flash -> fp32 no-flash
    # Prefer high-accuracy FP32 first on GH200; fall back to BF16/FP16 only if needed
    model = (
        _load_and_smoke_test(torch.float32, use_flash=False)
        or _load_and_smoke_test(torch.bfloat16, use_flash=True)
        or _load_and_smoke_test(torch.float16, use_flash=False)
    )
    if model is None:
        raise RuntimeError("Failed to load model on CUDA for evaluation after multiple fallbacks.")
    # Pick the active dtype from parameters (first param tensor dtype)
    first_param = next(model.parameters())
    active_dtype = first_param.dtype

    # Use streaming to avoid downloading entire large datasets.
    try:
        ds = datasets.load_dataset(
            dataset_name,
            name=dataset_config,
            split=split,
            streaming=True,
        )
    except Exception as load_exc:
        available = None
        try:
            available = datasets.get_dataset_split_names(dataset_name, dataset_config)
        except Exception:
            pass
        detail = (
            f"Available splits: {available}"
            if available
            else f"Original error: {load_exc}"
        )
        raise RuntimeError(
            f"Unable to load dataset '{dataset_name}' (config={dataset_config or 'default'}) "
            f"split '{split}'. {detail}"
        ) from load_exc

    losses: List[float] = []
    count = 0
    batch_texts: List[str] = []
    # Allow adaptive downshifting on OOM if batch_size <= 0 is provided or OOM occurs.
    current_bs = batch_size if batch_size and batch_size > 0 else 64
    last_oom_error = None  # store last CUDA OOM to re-raise with context when needed

    # Safety: ensure token ids are within embedding size; warn once
    embed_size = model.get_input_embeddings().num_embeddings
    oob_warned = False

    def _process_batch(texts: List[str]) -> bool:
        nonlocal losses, count, current_bs, last_oom_error, oob_warned
        if not texts:
            return True
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        # OOB id guard: map any ids outside embedding range to eos_id
        ids = enc["input_ids"]
        oob_mask = (ids < 0) | (ids >= embed_size)
        if oob_mask.any():
            if not oob_warned:
                logger.warning(
                    "Detected %d out-of-bound token ids; mapping to eos_id=%s. "
                    "Check tokenizer/model vocab alignment.",
                    int(oob_mask.sum().item()),
                    str(eos_id),
                )
                oob_warned = True
            ids = ids.masked_fill(oob_mask, eos_id)
            enc["input_ids"] = ids
        enc = {k: v.to("cuda", non_blocking=True) for k, v in enc.items()}
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=active_dtype if active_dtype != torch.float32 else torch.float32):
                out = model(**enc, labels=enc["input_ids"])
                losses.append(float(out.loss.detach().cpu()))
            count += len(texts)
            return True
        except torch.cuda.OutOfMemoryError as e:
            last_oom_error = e
            torch.cuda.empty_cache()
            # Downshift micro-batch and signal failure to retry with smaller chunk.
            current_bs = max(1, current_bs // 2)
            return False
    for example in ds:
        if max_samples and count >= max_samples:
            break
        text = example.get("text") or example.get("document") or example.get("abstract") or ""
        if not text:
            continue
        batch_texts.append(text)
        if len(batch_texts) >= current_bs:
            # Try to process; if OOM, reduce current_bs and retry with a smaller slice.
            while len(batch_texts) >= current_bs:
                if _process_batch(batch_texts[:current_bs]):
                    batch_texts = batch_texts[current_bs:]
                else:
                    # If we dropped to bs=1 and still OOM, give up early with a clear error.
                    if current_bs <= 1:
                        raise RuntimeError(
                            "CUDA out of memory during evaluation at batch size 1; "
                            "reduce --batch-size or --max-length and retry."
                        ) from last_oom_error

    # Flush remaining
    if batch_texts and (not max_samples or count < max_samples):
        # Process any remaining texts in micro-batches of current_bs
        while batch_texts:
            bs = min(current_bs, len(batch_texts))
            if _process_batch(batch_texts[:bs]):
                batch_texts = batch_texts[bs:]
            else:
                if current_bs <= 1:
                    raise RuntimeError(
                        "CUDA out of memory while flushing remaining texts at batch size 1; "
                        "reduce --batch-size or --max-length and retry."
                    ) from last_oom_error

    if not losses:
        return float("nan")

    avg_loss = sum(losses) / len(losses)
    perplexity = float(torch.exp(torch.tensor(avg_loss)))
    logger.info(
        "Perplexity evaluation finished: dataset=%s config=%s split=%s samples=%d dtype=%s",
        dataset_name,
        dataset_config or "-",
        split,
        count,
        active_dtype,
    )
    return perplexity


def _score_candidates_logprob_sum(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    candidates: Sequence[str],
    dtype: torch.dtype,
    max_length: int = 256,
) -> Dict[str, float]:
    """
    Score each candidate by the (approximate) sum of log-probs over its tokens
    given the prompt. We approximate per-token log-prob by scaling the sequence
    loss with the candidate token count, which is robust and efficient.
    """
    scores: Dict[str, float] = {}
    # Tokenize prompt once
    enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_ids = enc_prompt["input_ids"][0]
    prompt_len = prompt_ids.shape[0]
    for cand in candidates:
        full = tokenizer(prompt + " " + cand, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = full["input_ids"][0]
        labels_ids = input_ids.clone()
        # Ignore prompt tokens; only supervise candidate tokens
        labels_ids[:prompt_len] = -100
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=dtype if dtype != torch.float32 else torch.float32
        ):
            out = model(
                input_ids=full["input_ids"].to("cuda"),
                labels=labels_ids.unsqueeze(0).to("cuda"),
            )
        # loss is mean NLL over candidate tokens; approximate sum by mean * count
        cand_len = int((labels_ids != -100).sum().item()) or 1
        scores[cand] = -float(out.loss.detach().cpu()) * cand_len
    return scores


def _load_model_with_fallbacks(model_path: str, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    """
    Load model with progressive fallbacks to avoid CUDA device-side asserts.
    """
    def _load_and_smoke_test(torch_dtype: torch.dtype, use_flash: bool) -> AutoModelForCausalLM | None:
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if use_flash and hasattr(cfg, "attn_implementation"):
                setattr(cfg, "attn_implementation", "flash_attention_2")
            model_local = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=cfg if use_flash else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            try:
                model_local = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
            except Exception:
                return None
        try:
            model_local.to("cuda").eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype if torch_dtype != torch.float32 else torch.float32):
                _ = model_local(**tokenizer("hello world", return_tensors="pt", truncation=True, max_length=8).to("cuda"))
            return model_local
        except Exception:
            try:
                del model_local  # type: ignore
            except Exception:
                pass
            torch.cuda.empty_cache()
            return None

    model = (
        _load_and_smoke_test(torch.bfloat16, use_flash=True)
        or _load_and_smoke_test(torch.float16, use_flash=False)
        or _load_and_smoke_test(torch.float32, use_flash=False)
    )
    if model is None:
        raise RuntimeError("Failed to load model on CUDA for evaluation after multiple fallbacks.")
    return model


def _extract_medmcqa_fields(example: Dict[str, Any]) -> Tuple[str, List[str], Optional[int]]:
    """
    Normalize MedMCQA entry -> (question, [opts], gold_index)
    """
    question = str(example.get("question") or example.get("Q") or "")
    options = [
        str(example.get("opa") or example.get("A") or ""),
        str(example.get("opb") or example.get("B") or ""),
        str(example.get("opc") or example.get("C") or ""),
        str(example.get("opd") or example.get("D") or ""),
    ]
    answer = example.get("cop") or example.get("answer") or example.get("label")
    idx: Optional[int] = None
    if isinstance(answer, str):
        ans = answer.strip().lower()
        idx = {"a": 0, "b": 1, "c": 2, "d": 3}.get(ans)
    elif isinstance(answer, int):
        idx = answer if 0 <= answer <= 3 else None
    return question, options, idx


def _load_medmcqa(split: str = "validation") -> datasets.Dataset:
    """
    Load the openlifescienceai/medmcqa dataset.
    """
    try:
        ds = datasets.load_dataset("openlifescienceai/medmcqa", split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load openlifescienceai/medmcqa split='{split}'. "
            "Verify network access and dataset availability."
        ) from exc
    logger.info("Loaded openlifescienceai/medmcqa split=%s (%d rows).", split, len(ds))
    return ds


def evaluate_medmcqa(
    *,
    model_path: str,
    tokenizer_path: str,
    split: str = "validation",
    max_samples: int | None = 2000,
) -> dict:
    """
    Multiple-choice (A–D) evaluation on the MedMCQA benchmark.
    Scores each option via log-prob of its text and picks the highest.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for eval")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model_with_fallbacks(model_path, tokenizer)
    dtype = next(model.parameters()).dtype

    ds = _load_medmcqa(split)
    labels = ["A", "B", "C", "D"]
    correct = 0
    total = 0
    per_class = {k: {"correct": 0, "total": 0} for k in labels}
    confusion = [[0, 0, 0, 0] for _ in labels]

    for example in ds:
        if max_samples and total >= max_samples:
            break
        question, options, gold_idx = _extract_medmcqa_fields(example)
        if not question or len(options) != 4 or gold_idx is None or not options[gold_idx]:
            continue
        prompt = (
            f"Question: {question}\nChoices:"
            f" (A) {options[0]} (B) {options[1]} (C) {options[2]} (D) {options[3]}\nAnswer:"
        )
        candidates = [f"{labels[i]}. {options[i]}" for i in range(4)]
        scores = _score_candidates_logprob_sum(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidates=candidates,
            dtype=dtype,
        )
        pred_idx = max(range(4), key=lambda idx: scores[candidates[idx]])
        total += 1
        confusion[gold_idx][pred_idx] += 1
        gold_label = labels[gold_idx]
        per_class[gold_label]["total"] += 1
        if pred_idx == gold_idx:
            correct += 1
            per_class[gold_label]["correct"] += 1
    acc = float(correct) / total if total else float("nan")
    return {
        "accuracy": acc,
        "total": total,
        "split": split,
        "per_class": {
            k: {
                "accuracy": (v["correct"] / v["total"]) if v["total"] else float("nan"),
                "total": v["total"],
            }
            for k, v in per_class.items()
        },
        "labels": labels,
        "confusion_matrix": confusion,
    }



def compute_term_tokenization_coverage(terms_path: str, tokenizer_path: str) -> Dict[str, Any]:
    """
    Compute tokenization coverage for mined medical terms.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lengths: List[int] = []
    total = 0
    with open(terms_path, "r", encoding="utf-8") as fp:
        for line in fp:
            term = line.strip()
            if not term:
                continue
            ids = tok(term, add_special_tokens=False)["input_ids"]
            piece_len = len(ids) if isinstance(ids, list) else int(ids.shape[-1])
            lengths.append(piece_len)
            total += 1
    if not total:
        return {
            "total_terms": 0,
            "single_token_ratio": float("nan"),
            "mean_tokens_per_term": float("nan"),
            "median_tokens_per_term": float("nan"),
            "p95_tokens_per_term": float("nan"),
        }
    single = sum(1 for l in lengths if l == 1)
    return {
        "total_terms": total,
        "single_token_ratio": single / total,
        "mean_tokens_per_term": float(_stats.mean(lengths)),
        "median_tokens_per_term": float(_stats.median(lengths)),
        "p95_tokens_per_term": float(_stats.quantiles(lengths, n=20)[-1]) if len(lengths) >= 20 else max(lengths),
    }


def compute_alignment_coverage(
    *,
    vocab_mapping_path: str,
    target_tokenizer_path: str,
) -> Dict[str, Any]:
    """
    Estimate alignment coverage using vocab_mapping.json (target→source coverage).
    """
    tok = AutoTokenizer.from_pretrained(target_tokenizer_path, trust_remote_code=True)
    target_vocab = int(getattr(tok, "vocab_size", None) or len(tok))
    mapped = 0
    try:
        with open(vocab_mapping_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            if "mapping" in data and isinstance(data["mapping"], dict):
                mapped = len([k for k, v in data["mapping"].items() if v is not None])
            elif "target_to_source" in data and isinstance(data["target_to_source"], dict):
                mapped = len([k for k, v in data["target_to_source"].items() if v is not None])
            else:
                mapped = len([k for k, v in data.items() if str(k).isdigit() and v is not None])
        elif isinstance(data, list):
            mapped = len(data)
    except Exception as e:
        logger.warning("Failed to parse vocab_mapping.json (%s); alignment coverage unknown.", e)
        return {"target_vocab": target_vocab, "mapped": None, "coverage": None}
    coverage = mapped / target_vocab if target_vocab > 0 else None
    return {"target_vocab": target_vocab, "mapped": mapped, "coverage": coverage}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate medical TokAlign models.")
    parser.add_argument("--model", required=True, help="Path to adapted model directory.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path used during evaluation.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Hugging Face dataset name (optionally dataset:split). Can be supplied multiple times.",
    )
    parser.add_argument("--output", required=True, help="Path to JSON file with evaluation results.")
    # Perplexity/general knobs
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate per dataset (0 = all samples).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of texts per forward pass.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization.",
    )
    # PubMedQA options
    parser.add_argument(
        "--run-medmcqa",
        action="store_true",
        help="Run MedMCQA accuracy in addition to perplexity.",
    )
    parser.add_argument(
        "--baseline-model",
        default="mistralai/Mistral-7B-v0.3",
        help="Baseline model for MedMCQA (only used if --run-medmcqa).",
    )
    parser.add_argument(
        "--medmcqa-split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Split to use from openlifescienceai/medmcqa when running --run-medmcqa.",
    )
    parser.add_argument(
        "--log-missing",
        action="store_true",
        help="If set and no dataset is provided, write a placeholder log describing required evaluations.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    os.makedirs(Path(args.output).parent, exist_ok=True)

    datasets_to_evaluate: List[str] = args.dataset if args.dataset else DEFAULT_DATASETS

    if not datasets_to_evaluate:
        placeholder = {
            "status": "missing_datasets",
            "message": (
                "No medical evaluation datasets were provided and no defaults are configured. "
                "Supply --dataset flags with Hugging Face IDs."
            ),
        }
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(placeholder, fp, indent=2)
        if args.log_missing:
            logger.warning(placeholder["message"])
        return

    results: Dict[str, Any] = {}
    for dataset_item in datasets_to_evaluate:
        dataset_name, dataset_config, split = parse_dataset_spec(dataset_item)
        label = dataset_name if not dataset_config else f"{dataset_name}[{dataset_config}]"
        logger.info(
            "Evaluating dataset %s split %s (config=%s).",
            dataset_name,
            split,
            dataset_config or "-",
        )
        perplexity = evaluate_perplexity(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            dataset_name=dataset_name,
            split=split,
            dataset_config=dataset_config,
            max_samples=None if args.max_samples <= 0 else args.max_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        results[f"{label}:{split}"] = {"perplexity": perplexity}

    if args.run_medmcqa:
        try:
            baseline = evaluate_medmcqa(
                model_path=args.baseline_model,
                tokenizer_path=args.baseline_model,
                split=args.medmcqa_split,
                max_samples=None if args.max_samples <= 0 else args.max_samples,
            )
            adapted = evaluate_medmcqa(
                model_path=args.model,
                tokenizer_path=args.tokenizer,
                split=args.medmcqa_split,
                max_samples=None if args.max_samples <= 0 else args.max_samples,
            )
            results["medmcqa"] = {
                "baseline": baseline,
                "adapted": adapted,
                "delta": {"accuracy": adapted["accuracy"] - baseline["accuracy"]},
            }
        except Exception as qa_exc:
            logger.warning("MedMCQA evaluation failed: %s", qa_exc)
            results["medmcqa"] = {"status": "error", "message": str(qa_exc)}

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Evaluation results written to %s.", args.output)


if __name__ == "__main__":
    main()

