#!/usr/bin/env python3
"""
End-to-end runner for the TokAlign medical adaptation pipeline.

Stages:
1. Prepare directories and aggregate the medical JSONL corpus.
2. Mine medical terms and augment both source/target tokenizers.
3. Tokenize the corpus with augmented tokenizers and extract embedding corpora.
4. Train embeddings (GloVe or FastText) and compute token alignment.
5. Apply the alignment to the base model.
6. Optionally evaluate on medical benchmarks.

Each stage includes retry logic with exponential backoff. Logs are written both
to stdout and to `runs/logs/<timestamp>.log`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

# Always prefer the local repository's src/ over any installed package named 'src'
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src import medical_corpus, medical_pipeline, medical_terms  # type: ignore

LOGGER = logging.getLogger("medical_runner")


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _setup_logging(run_dir: Path) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "medical_pipeline.log"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )
    LOGGER.info("Logging initialised. Detailed logs at %s.", log_file)


def _retry(stage_name: str, max_retries: int, backoff: float, fn: Callable[[], Dict[str, str]]) -> Dict[str, str]:
    attempt = 0
    while True:
        try:
            LOGGER.info("Starting stage: %s (attempt %s)", stage_name, attempt + 1)
            result = fn()
            LOGGER.info("Stage %s completed.", stage_name)
            return result
        except Exception as exc:  # pragma: no cover - runtime guard
            attempt += 1
            if attempt > max_retries:
                LOGGER.exception("Stage %s failed after %s attempts.", stage_name, attempt)
                raise
            sleep_time = backoff * (2 ** (attempt - 1))
            LOGGER.warning(
                "Stage %s failed with %s. Retrying in %.2f seconds (%s/%s).",
                stage_name,
                exc,
                sleep_time,
                attempt,
                max_retries,
            )
            time.sleep(sleep_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical TokAlign pipeline end-to-end.")
    parser.add_argument("--input", action="append", required=True, help="Medical corpus JSONL shard or directory.")
    parser.add_argument("--source-tokenizer", required=True, help="Source tokenizer/model identifier.")
    parser.add_argument("--target-tokenizer", required=True, help="Target tokenizer/model identifier.")
    parser.add_argument("--source-model", required=True, help="Base model checkpoint for alignment.")
    parser.add_argument(
        "--source-model-fallback",
        help="Fallback model path if augmented source tokenizer lacks config (default: use --source-model).",
    )
    parser.add_argument(
        "--target-model-fallback",
        help="Fallback model path if augmented target tokenizer lacks config (default: use --target-tokenizer).",
    )
    parser.add_argument("--run-root", default="runs/tokenizer_adapt", help="Root directory for run artefacts.")
    parser.add_argument("--run-id", help="Optional run ID (timestamp used if omitted).")
    parser.add_argument("--byte-budget", type=int, default=0, help="Maximum bytes to ingest (0 = unlimited).")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication during ingestion.")
    parser.add_argument("--hash-name", default="sha256", help="Hash algorithm for deduplication.")
    parser.add_argument("--min-term-frequency", type=int, default=5, help="Minimum term frequency.")
    parser.add_argument("--term-top-k", type=int, default=500, help="Number of terms to keep.")
    parser.add_argument("--use-tfidf", action="store_true", help="Use TF-IDF weighting when mining terms.")
    parser.add_argument(
        "--embedding-backend",
        choices=["glove", "fasttext"],
        default="fasttext",
        help="Embedding backend (FastText relies on the prebuilt fasttext-wheel package).",
    )
    parser.add_argument("--pivot-count", type=int, default=300, help="Number of pivot tokens during alignment.")
    parser.add_argument("--tokenizer-workers", type=int, default=8, help="Number of preprocessing workers.")
    parser.add_argument("--tokenizer-cache", help="Cache directory passed to Hugging Face dataset loader.")
    parser.add_argument("--evaluation-dataset", action="append", help="Optional HF dataset names for evaluation.")
    parser.add_argument("--max-eval-samples", type=int, default=128, help="Samples evaluated per dataset.")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries per stage.")
    parser.add_argument("--retry-backoff", type=float, default=5.0, help="Initial backoff (seconds).")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation stage when set.")
    parser.add_argument("--qa", action="store_true", help="Run PubMedQA evaluation during the evaluation stage.")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation even if --evaluate is set (useful for quick starts).",
    )
    parser.add_argument(
        "--eval-tokenizer",
        help="Tokenizer path used for evaluation (defaults to run tokenizer target directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = medical_pipeline.create_run_dir(Path(args.run_root), args.run_id)
    _setup_logging(run_dir)

    source_model_fallback = args.source_model_fallback or args.source_model
    target_model_fallback = args.target_model_fallback or args.target_tokenizer

    LOGGER.info("Run directory: %s", run_dir)

    stage_outputs: Dict[str, Dict[str, str]] = {}

    stage_outputs["data_prep"] = _retry(
        "aggregate_corpus",
        args.max_retries,
        args.retry_backoff,
        lambda: medical_pipeline.data_prep(
            corpus_inputs=args.input,
            run_dir=run_dir,
            byte_budget=None if args.byte_budget <= 0 else args.byte_budget,
            deduplicate=not args.no_dedup,
            hash_name=args.hash_name,
        ),
    )

    aggregated_jsonl = stage_outputs["data_prep"]["aggregated_jsonl"]

    def _mine_terms() -> Dict[str, str]:
        terms = medical_terms.mine_terms(
            corpus_path=aggregated_jsonl,
            top_k=args.term_top_k,
            min_count=args.min_term_frequency,
            use_tfidf=args.use_tfidf,
        )
        terms_path = run_dir / "corpus" / "medical_terms.txt"
        with open(terms_path, "w", encoding="utf-8") as fp:
            for term in terms:
                fp.write(term + "\n")
        LOGGER.info("Selected %s medical terms.", len(terms))

        augmented_dir = run_dir / "tokenizers"
        source_aug_dir = augmented_dir / "source"
        target_aug_dir = augmented_dir / "target"
        added_source, skipped_source = medical_terms.augment_tokenizer(
            tokenizer_path=args.source_tokenizer,
            terms=terms,
            output_dir=str(source_aug_dir),
        )
        added_target, skipped_target = medical_terms.augment_tokenizer(
            tokenizer_path=args.target_tokenizer,
            terms=terms,
            output_dir=str(target_aug_dir),
        )
        summary = {
            "terms_path": str(terms_path),
            "source_tokenizer": str(source_aug_dir),
            "target_tokenizer": str(target_aug_dir),
            "added_source": len(added_source),
            "added_target": len(added_target),
            "skipped_source": len(skipped_source),
            "skipped_target": len(skipped_target),
        }
        report_path = run_dir / "tokenizers" / "augmentation_report.json"
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        summary["augmentation_report"] = str(report_path)
        return summary

    stage_outputs["term_mining"] = _retry(
        "mine_terms_and_augment_tokenizers",
        args.max_retries,
        args.retry_backoff,
        _mine_terms,
    )

    augmented_source = stage_outputs["term_mining"]["source_tokenizer"]
    augmented_target = stage_outputs["term_mining"]["target_tokenizer"]

    stage_outputs["tokenize"] = _retry(
        "tokenize_and_extract",
        args.max_retries,
        args.retry_backoff,
        lambda: medical_pipeline.tokenize_corpus(
            run_dir=run_dir,
            aggregated_jsonl=aggregated_jsonl,
            tokenizer_source=augmented_source,
            tokenizer_target=augmented_target,
            source_model_fallback=source_model_fallback,
            target_model_fallback=target_model_fallback,
            tokenizer_workers=args.tokenizer_workers,
            tokenizer_cache_dir=args.tokenizer_cache,
        ),
    )

    stage_outputs["train_align"] = _retry(
        "train_embeddings_and_align",
        args.max_retries,
        args.retry_backoff,
        lambda: medical_pipeline.train_embeddings_and_align(
            run_dir=run_dir,
            source_glove_path=stage_outputs["tokenize"]["source_glove"],
            target_glove_path=stage_outputs["tokenize"]["target_glove"],
            tokenizer_source=augmented_source,
            tokenizer_target=augmented_target,
            embedding_backend=args.embedding_backend,
            pivot_count=args.pivot_count,
        ),
    )

    # Optional BLEU evaluation on alignment matrices if a reference pairs file is provided
    try:
        matrix_ref = os.getenv("TOKALIGN_MATRIX_EVAL_FILE")
        medical_pipeline.evaluate_alignment_bleu(
            run_dir=run_dir,
            align_matrix=stage_outputs["train_align"]["align_matrix"],
            eval_pairs_path=matrix_ref,
            ngram=int(os.getenv("TOKALIGN_MATRIX_BLEU_N", "1")),
        )
    except Exception as _bleu_exc:  # pragma: no cover - do not fail pipeline
        LOGGER.warning("Matrix BLEU evaluation skipped/failed: %s", _bleu_exc)
    stage_outputs["apply"] = _retry(
        "apply_alignment",
        args.max_retries,
        args.retry_backoff,
        lambda: {
            "model_dir": medical_pipeline.apply_alignment(
                run_dir=run_dir,
                align_matrix=stage_outputs["train_align"]["align_matrix"],
                source_model=args.source_model,
                target_tokenizer=augmented_target,
            )
        },
    )

    if args.evaluate and not args.skip_eval and args.evaluation_dataset:
        from src import eval_medical

        eval_output = run_dir / "evaluation.json"
        eval_tokenizer = args.eval_tokenizer or augmented_target

        def _run_evaluation() -> Dict[str, str]:
            results = {}
            # Read recommended knobs from environment to avoid expanding CLI surface.
            eval_batch = int(os.getenv("TOKALIGN_EVAL_BATCH", "32"))
            eval_maxlen = int(os.getenv("TOKALIGN_EVAL_MAXLEN", "1024"))
            for dataset_item in args.evaluation_dataset:
                if ":" in dataset_item:
                    dataset_name, split = dataset_item.split(":", maxsplit=1)
                else:
                    dataset_name, split = dataset_item, "test"
                LOGGER.info("Evaluating %s:%s", dataset_name, split)
                results[f"{dataset_name}:{split}"] = {
                    "perplexity": eval_medical.evaluate_perplexity(
                        model_path=stage_outputs["apply"]["model_dir"],
                        tokenizer_path=eval_tokenizer,
                        dataset_name=dataset_name,
                        split=split,
                        max_samples=None if args.max_eval_samples <= 0 else args.max_eval_samples,
                        batch_size=eval_batch,
                        max_length=eval_maxlen,
                    )
                }
            # Optional PubMedQA
            if args.qa or os.getenv("MEDICAL_EVAL_QA", "") == "1":
                qa_out = run_dir / "eval" / "pubmedqa.json"
                os.makedirs(qa_out.parent, exist_ok=True)
                try:
                    qa_res = eval_medical.evaluate_pubmedqa(
                        model_path=stage_outputs["apply"]["model_dir"],
                        tokenizer_path=eval_tokenizer,
                        subset=os.getenv("TOKALIGN_PUBMEDQA_SUBSET", "pqa_labeled"),
                        split=os.getenv("TOKALIGN_PUBMEDQA_SPLIT", "test"),
                        max_samples=int(os.getenv("TOKALIGN_PUBMEDQA_MAX", "200")),
                    )
                    with open(qa_out, "w", encoding="utf-8") as fp:
                        json.dump(qa_res, fp, indent=2)
                    results["pubmedqa"] = {"results_path": str(qa_out), **qa_res}
                except Exception as qa_exc:  # pragma: no cover
                    LOGGER.warning("PubMedQA evaluation skipped/failed: %s", qa_exc)
                    results["pubmedqa"] = {"status": "error", "message": str(qa_exc)}
            with open(eval_output, "w", encoding="utf-8") as fp:
                json.dump(results, fp, indent=2)
            return {
                "evaluation_output": str(eval_output),
                "tokenizer": eval_tokenizer,
                "model": stage_outputs["apply"]["model_dir"],
                "results": results,
            }

        stage_outputs["evaluation"] = _retry(
            "evaluation",
            args.max_retries,
            args.retry_backoff,
            _run_evaluation,
        )

    summary_path = run_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(stage_outputs, fp, indent=2)
    LOGGER.info("Pipeline completed. Summary written to %s.", summary_path)


if __name__ == "__main__":
    main()

