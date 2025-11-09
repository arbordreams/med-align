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
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def evaluate_perplexity(
    model_path: str,
    tokenizer_path: str,
    dataset_name: str,
    split: str,
    max_samples: int | None = None,
) -> float:
    """
    Compute a simple perplexity metric over the requested dataset split.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()

    ds = datasets.load_dataset(dataset_name, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    losses: List[float] = []
    for example in ds:
        text = example.get("text") or example.get("document") or ""
        if not text:
            continue
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs, labels=inputs["input_ids"])
            loss = float(output.loss.detach().cpu())
            losses.append(loss)

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
        default=128,
        help="Maximum number of samples to evaluate per dataset (0 = all samples).",
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

    if not args.dataset:
        placeholder = {
            "status": "missing_datasets",
            "message": "No medical evaluation datasets were provided. Supply --dataset flags with Hugging Face IDs.",
        }
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(placeholder, fp, indent=2)
        if args.log_missing:
            logger.warning(placeholder["message"])
        return

    results = {}
    for dataset_item in args.dataset:
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
        )
        results[f"{dataset_name}:{split}"] = {"perplexity": perplexity}

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Evaluation results written to %s.", args.output)


if __name__ == "__main__":
    main()

