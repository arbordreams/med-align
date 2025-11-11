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
from typing import List, Sequence

import datasets
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATASETS: List[str] = [
    "uiyunkim-hub/pubmed-abstract:train",
]

logger = logging.getLogger(__name__)


def evaluate_perplexity(
    model_path: str,
    tokenizer_path: str,
    dataset_name: str,
    split: str,
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

    dtype = torch.bfloat16
    model = None
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(cfg, "attn_implementation"):
            setattr(cfg, "attn_implementation", "flash_attention_2")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except Exception:
        # Fallback if config tweak is not supported
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    model.to("cuda").eval()

    # Use streaming to avoid downloading entire large datasets. Fall back if split is missing.
    def _try_load(name: str, sp: str):
        return datasets.load_dataset(name, split=sp, streaming=True)

    ds = None
    try:
        ds = _try_load(dataset_name, split)
    except Exception:
        # Fallback order: validation -> train
        for fallback_split in ("validation", "train"):
            try:
                logger.warning(
                    "Requested split '%s' unavailable for %s; falling back to '%s'.",
                    split,
                    dataset_name,
                    fallback_split,
                )
                ds = _try_load(dataset_name, fallback_split)
                split = fallback_split
                break
            except Exception:
                continue
        if ds is None:
            raise

    losses: List[float] = []
    count = 0
    batch_texts: List[str] = []
    # Allow adaptive downshifting on OOM if batch_size <= 0 is provided or OOM occurs.
    current_bs = batch_size if batch_size and batch_size > 0 else 64
    last_oom_error = None  # store last CUDA OOM to re-raise with context when needed

    def _process_batch(texts: List[str]) -> bool:
        nonlocal losses, count, current_bs, last_oom_error
        if not texts:
            return True
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to("cuda", non_blocking=True) for k, v in enc.items()}
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
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
    return perplexity


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

    results = {}
    for dataset_item in datasets_to_evaluate:
        if ":" in dataset_item:
            dataset_name, split = dataset_item.split(":", maxsplit=1)
        else:
            dataset_name, split = dataset_item, "test"
        logger.info("Evaluating dataset %s split %s.", dataset_name, split)
        perplexity = evaluate_perplexity(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            dataset_name=dataset_name,
            split=split,
            max_samples=None if args.max_samples <= 0 else args.max_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        results[f"{dataset_name}:{split}"] = {"perplexity": perplexity}

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Evaluation results written to %s.", args.output)


if __name__ == "__main__":
    main()

