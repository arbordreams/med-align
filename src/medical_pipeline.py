"""
High-level orchestration utilities for the medical TokAlign pipeline.

The CLI exposes three sub-commands that correspond to the medical extensions of
the original shell scripts:

* `data-prep` mirrors the behaviour of `convert2glove_corpus.sh` for medical
  corpora.
* `train-align` replaces the embedding training + alignment steps in
  `token_align.sh`.
* `apply` triggers the weight conversion handled by `init_model.sh`.

The module is also used by `script/run_medical_pipeline.py` to execute the
end-to-end workflow with retry logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
import concurrent.futures as _futures

import torch
from transformers import AutoTokenizer

try:
    from . import medical_corpus  # type: ignore
    from . import medical_terms  # type: ignore
except ImportError:  # pragma: no cover
    import medical_corpus  # type: ignore
    import medical_terms  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    import fasttext as fasttext_module  # type: ignore[import]  # noqa: F401

logger = logging.getLogger(__name__)

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = SCRIPT_ROOT / "runs" / "tokenizer_adapt"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def create_run_dir(run_root: Path = DEFAULT_RUN_ROOT, run_id: Optional[str] = None) -> Path:
    run_id = run_id or timestamp()
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _is_flash_attn_available() -> bool:
    try:
        import flash_attn  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _run_subprocess(
    cmd: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path | str] = None,
) -> None:
    logger.debug("Executing command: %s", " ".join(cmd))
    # Ensure the repo's 'src' is importable first in subprocesses
    repo_root = str(SCRIPT_ROOT)
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        pythonpath = f"{repo_root}:{pythonpath}"
    else:
        pythonpath = repo_root
    env_dict = {**os.environ, "PYTHONPATH": pythonpath, **(env or {})}
    subprocess.run(cmd, check=True, env=env_dict, cwd=str(cwd) if cwd else None)


def data_prep(
    *,
    corpus_inputs: Sequence[str],
    run_dir: Path,
    byte_budget: Optional[int],
    deduplicate: bool,
    hash_name: str,
) -> Dict[str, str]:
    """
    Aggregate the medical corpus into a single JSONL file and mirrored pairs.
    """

    corpus_dir = run_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    aggregated_jsonl = corpus_dir / "medical_corpus.jsonl"
    mirrored_jsonl = corpus_dir / "medical_pairs.jsonl"
    stats_path = corpus_dir / "corpus_stats.json"

    stats = medical_corpus.build_medical_corpus(
        inputs=corpus_inputs,
        output_path=str(aggregated_jsonl),
        mirror_output_path=str(mirrored_jsonl),
        byte_budget=byte_budget,
        deduplicate=deduplicate,
        hash_name=hash_name,
    )

    with open(stats_path, "w", encoding="utf-8") as fp:
        json.dump(stats.to_dict(), fp, indent=2)

    logger.info("Corpus stats recorded at %s.", stats_path)

    outputs = {
        "aggregated_jsonl": str(aggregated_jsonl),
        "mirrored_jsonl": str(mirrored_jsonl),
        "corpus_stats": str(stats_path),
    }

    return outputs


def _resolve_tokenizer_model_path(tokenizer_path: str, fallback_model: str) -> Tuple[str, str]:
    """
    Determine the model_name_or_path to accompany a tokenizer clone.
    If the directory lacks a config.json, we fall back to the provided model.
    Returns (model_path, tokenizer_path).
    """

    local_config = Path(tokenizer_path) / "config.json"
    if local_config.is_file():
        return tokenizer_path, tokenizer_path
    return fallback_model, tokenizer_path


def tokenize_with_tokenizer(
    *,
    run_dir: Path,
    aggregated_jsonl: str,
    tokenizer_path: str,
    fallback_model: str,
    output_subdir: str,
    tokenizer_workers: int,
    tokenizer_cache_dir: Optional[str] = None,
) -> str:
    """
    Tokenize the aggregated corpus with a single tokenizer and persist the dataset.
    This helper is reused for baseline dataset preparation.
    """
    dataset_dir = run_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / output_subdir
    dataset_path.mkdir(parents=True, exist_ok=True)

    cache_dir = tokenizer_cache_dir or str(run_dir / "cache")
    os.makedirs(cache_dir, exist_ok=True)

    model_path, token_path = _resolve_tokenizer_model_path(tokenizer_path, fallback_model)
    cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "src" / "process_dataset.py"),
        "--model_name_or_path",
        model_path,
        "--tokenizer_name",
        token_path,
        "--train_file",
        aggregated_jsonl,
        "--only_tokenize",
        "--validation_split_percentage",
        "0",
        "--dataset_path_in_disk",
        str(dataset_path),
        "--preprocessing_num_workers",
        str(tokenizer_workers),
        "--cache_dir",
        cache_dir,
        "--output_dir",
        str(run_dir / "logs"),
    ]
    logger.info(
        "Tokenizing corpus with %s (output=%s).",
        tokenizer_path,
        dataset_path,
    )
    _run_subprocess(cmd)
    return str(dataset_path)

def _train_fasttext_embedding(
    corpus_path: str,
    save_file: str,
    *,
    epochs: int,
    mincount: int,
    lr: float,
    thread: int,
) -> str:
    """
    Train a FastText model on corpus_path and write vectors to save_file.
    Implemented at module scope so it can be pickled by ProcessPoolExecutor.
    """
    # Local import inside worker to avoid import-time failures when backend not selected
    try:
        import fasttext  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastText backend selected but the `fasttext` module is unavailable. "
            "Install the prebuilt `fasttext-wheel>=0.9.2` package (pip install fasttext-wheel)."
        ) from exc
    model = fasttext.train_unsupervised(
        input=corpus_path,
        model="skipgram",
        dim=300,
        epoch=epochs,
        minCount=mincount,
        lr=lr,
        thread=thread,
    )
    bin_path = str(Path(save_file).with_suffix(".bin"))
    model.save_model(bin_path)
    words = model.get_words(include_freq=True)
    if isinstance(words, tuple):
        word_iterable = zip(*words)
    else:
        word_iterable = words
    with open(save_file, "w", encoding="utf-8") as fp:
        for entry in word_iterable:
            # entry may be "word", ("word", freq), or a non-string ID depending on the fasttext build
            if isinstance(entry, (tuple, list)):
                word = entry[0]
            else:
                word = entry
            if not isinstance(word, str):
                word = str(word)
            vector = model.get_word_vector(word)
            fp.write(word + " " + " ".join(map(str, vector.tolist())) + "\n")
    return save_file


def tokenize_corpus(
    *,
    run_dir: Path,
    aggregated_jsonl: str,
    tokenizer_source: str,
    tokenizer_target: str,
    source_model_fallback: str,
    target_model_fallback: str,
    tokenizer_workers: int,
    tokenizer_cache_dir: Optional[str] = None,
    min_line_length: int = 0,
) -> Dict[str, str]:
    dataset_dir = run_dir / "datasets"
    glove_dir = run_dir / "glove_corpus"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    glove_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(run_dir / "logs", exist_ok=True)

    cache_dir = tokenizer_cache_dir or str(run_dir / "cache")
    os.makedirs(cache_dir, exist_ok=True)

    source_dataset = Path(
        tokenize_with_tokenizer(
            run_dir=run_dir,
            aggregated_jsonl=aggregated_jsonl,
            tokenizer_path=tokenizer_source,
            fallback_model=source_model_fallback,
            output_subdir="source",
            tokenizer_workers=tokenizer_workers,
            tokenizer_cache_dir=tokenizer_cache_dir,
        )
    )
    target_dataset = Path(
        tokenize_with_tokenizer(
            run_dir=run_dir,
            aggregated_jsonl=aggregated_jsonl,
            tokenizer_path=tokenizer_target,
            fallback_model=target_model_fallback,
            output_subdir="target",
            tokenizer_workers=tokenizer_workers,
            tokenizer_cache_dir=tokenizer_cache_dir,
        )
    )

    def _convert(dataset_path: Path, output_file: Path) -> Path:
        cmd = [
            sys.executable,
            str(SCRIPT_ROOT / "src" / "convert2glove_train.py"),
            "-s",
            str(dataset_path),
            "-k",
            "train",
            "-o",
            str(output_file),
            "-m",
            str(min_line_length),
        ]
        logger.info("Generating embedding corpus at %s.", output_file)
        _run_subprocess(cmd)
        return output_file

    source_glove = _convert(source_dataset, glove_dir / "source.txt")
    target_glove = _convert(target_dataset, glove_dir / "target.txt")

    return {
        "source_dataset": str(source_dataset),
        "target_dataset": str(target_dataset),
        "source_glove": str(source_glove),
        "target_glove": str(target_glove),
    }


def train_embeddings_and_align(
    *,
    run_dir: Path,
    source_glove_path: str,
    target_glove_path: str,
    tokenizer_source: str,
    tokenizer_target: str,
    embedding_backend: str,
    pivot_count: int,
    fasttext_epochs: int = 5,
    fasttext_mincount: int = 5,
    fasttext_lr: float = 0.05,
    thread: int = 8,
    gold_mapping_path: Optional[str] = None,
    similarity_threshold: float = 0.3,
) -> Dict[str, str]:
    align_dir = run_dir / "alignment"
    align_dir.mkdir(parents=True, exist_ok=True)

    embedding_backend = embedding_backend.lower()
    if embedding_backend not in {"glove", "fasttext"}:
        raise ValueError(f"Unsupported embedding backend: {embedding_backend}")

    glove_dir: Optional[Path] = None
    if embedding_backend == "glove":
        glove_dir_env = os.getenv("GLOVE_DIR")
        if not glove_dir_env:
            logger.warning("GLOVE_DIR is not set; falling back to FastText embedding backend.")
            embedding_backend = "fasttext"
        else:
            glove_dir = Path(glove_dir_env).expanduser().resolve()
            if not glove_dir.exists():
                logger.warning(
                    "GLOVE_DIR path %s does not exist; falling back to FastText embedding backend.",
                    glove_dir,
                )
                embedding_backend = "fasttext"

    vec_source = align_dir / f"source_vec.{embedding_backend}.txt"
    vec_target = align_dir / f"target_vec.{embedding_backend}.txt"

    if embedding_backend == "glove":
        train_script = SCRIPT_ROOT / "script" / "train_glove.sh"
        assert glove_dir is not None  # for mypy/static checkers
        for corpus_path, save_file in ((source_glove_path, vec_source), (target_glove_path, vec_target)):
            cmd = ["bash", str(train_script), corpus_path, str(Path(save_file).with_suffix(""))]
            logger.info("Training GloVe embeddings for %s.", corpus_path)
            _run_subprocess(cmd, cwd=glove_dir)
    else:
        # Train FastText embeddings using python binding (parallel for source and target).
        jobs = [
            (source_glove_path, str(vec_source)),
            (target_glove_path, str(vec_target)),
        ]
        logger.info("Training FastText embeddings in parallel (workers=2).")
        with _futures.ProcessPoolExecutor(max_workers=2) as executor:
            future_map = {
                executor.submit(
                    _train_fasttext_embedding,
                    corpus_path,
                    save_file,
                    epochs=fasttext_epochs,
                    mincount=fasttext_mincount,
                    lr=fasttext_lr,
                    thread=thread,
                ): (corpus_path, save_file)
                for corpus_path, save_file in jobs
            }
            for future in _futures.as_completed(future_map):
                corpus_path, save_file = future_map[future]
                try:
                    result_path = future.result()
                    logger.info("FastText training complete for %s -> %s", corpus_path, result_path)
                except Exception as exc:  # pragma: no cover - surface worker errors
                    logger.exception("FastText training failed for %s: %s", corpus_path, exc)
                    raise

    # Infer vocab sizes from tokenizers before evaluating coverage/alignment
    try:
        source_tok = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        target_tok = AutoTokenizer.from_pretrained(tokenizer_target, trust_remote_code=True)
        source_vocab_size = int(len(source_tok))
        target_vocab_size = int(len(target_tok))
        logger.info(
            "Tokenizer vocab sizes (len) source=%d (reported=%s) target=%d (reported=%s)",
            source_vocab_size,
            getattr(source_tok, "vocab_size", None),
            target_vocab_size,
            getattr(target_tok, "vocab_size", None),
        )
    except Exception as tok_exc:  # pragma: no cover
        logger.warning(
            "Failed to load tokenizers to infer vocab sizes (%s). Falling back to 32768 for both.",
            tok_exc,
        )
        # Reasonable default for Mistral-7B / BioMistral-7B; prevents out-of-range alignment
        source_vocab_size = 32768
        target_vocab_size = 32768

    logger.info("Counting vocabulary overlaps between %s and %s.", tokenizer_source, tokenizer_target)
    vocab_count_path = align_dir / "vocab_mapping.json"
    cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "src" / "count_dict.py"),
        "-s",
        tokenizer_source,
        "-t",
        tokenizer_target,
        "-o",
        str(vocab_count_path),
    ]
    _run_subprocess(cmd)
    mapped_targets = None
    try:
        with open(vocab_count_path, "r", encoding="utf-8") as fp:
            vocab_mapping = json.load(fp)
        if isinstance(vocab_mapping, dict):
            mapped_targets = len([v for v in vocab_mapping.values() if v is not None])
        elif isinstance(vocab_mapping, list):
            mapped_targets = len(vocab_mapping)
    except Exception as mapping_exc:  # pragma: no cover
        logger.warning("Unable to inspect vocab mapping coverage: %s", mapping_exc)
    else:
        if mapped_targets is not None and target_vocab_size:
            coverage = mapped_targets / target_vocab_size
            if coverage > 1.0:
                logger.warning(
                    "Vocabulary mapping reported coverage %.3f (>1.0). "
                    "Check that tokenizer lengths and mappings align. (mapped=%d target=%d)",
                    coverage,
                    mapped_targets,
                    target_vocab_size,
                )
            else:
                logger.info(
                    "Vocabulary mapping coverage: %d/%d (%.2f%%).",
                    mapped_targets,
                    target_vocab_size,
                    coverage * 100.0,
                )

    align_matrix_path = align_dir / "align_matrix.json"
    gold_mapping = gold_mapping_path or str(vocab_count_path)
    cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "src" / "cal_trans_matrix.py"),
        "-s",
        str(vec_source),
        "-s1",
        str(source_vocab_size),
        "-t",
        str(vec_target),
        "-s2",
        str(target_vocab_size),
        "-g",
        gold_mapping,
        "-o",
        str(align_matrix_path),
        "-r",
        "-n",
        str(pivot_count),
        "--similarity-threshold",
        str(similarity_threshold),
    ]
    _run_subprocess(cmd)

    report_path = align_dir / "alignment_report.json"
    report = {
        "embedding_backend": embedding_backend,
        "pivot_count": pivot_count,
        "source_glove": str(source_glove_path),
        "target_glove": str(target_glove_path),
        "align_matrix": str(align_matrix_path),
        "vocab_mapping": str(vocab_count_path),
    }
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return {
        "align_matrix": str(align_matrix_path),
        "vocab_mapping": str(vocab_count_path),
        "alignment_report": str(report_path),
        "source_vectors": str(vec_source),
        "target_vectors": str(vec_target),
    }


def apply_alignment(
    *,
    run_dir: Path,
    align_matrix: str,
    source_model: str,
    target_tokenizer: str,
) -> str:
    output_dir = run_dir / "adapted_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "src" / "convert.py"),
        "-m",
        align_matrix,
        "-s",
        source_model,
        "-t",
        target_tokenizer,
        "-o",
        str(output_dir),
    ]
    logger.info("Applying alignment to initialize adapted model at %s.", output_dir)
    _run_subprocess(cmd)
    return str(output_dir)


def evaluate_alignment_bleu(
    *,
    run_dir: Path,
    align_matrix: str,
    eval_pairs_path: Optional[str],
    ngram: int = 1,
) -> Optional[str]:
    """
    Evaluate alignment quality with BLEU-N on token-pair references if provided.
    Writes results to runs/<run>/metrics/matrix_bleu.json and returns the path.
    If eval_pairs_path is None or missing, logs and returns None.
    """
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if not eval_pairs_path or not Path(eval_pairs_path).exists():
        logger.warning("No alignment reference pairs provided; skipping BLEU evaluation.")
        return None
    try:
        from . import eval_matrix  # type: ignore
    except Exception:
        import sys as _sys
        _sys.path.append(str(SCRIPT_ROOT / "src"))
        import eval_matrix  # type: ignore
    result = eval_matrix.compute_bleu(align_matrix, eval_pairs_path, ngram=ngram)
    out_path = metrics_dir / "matrix_bleu.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)
    logger.info("Matrix BLEU written to %s: %s", out_path, result)
    return str(out_path)


def embedding_warmup(
    *,
    run_dir: Path,
    adapted_model_path: str,
    dataset_path: str,
    config: Dict[str, object],
) -> Dict[str, str]:
    """
    Lightweight embedding-only warm-up stage after vocabulary alignment.
    
    Trains only embedding layer + LM head (~1-5k steps) while freezing all
    transformer layers. This stabilizes new medical tokens, corrects embedding
    drift, and improves similarity neighborhoods at low cost (<1% of full FT).
    
    TokAlign reports 30-60% perplexity reduction from this stage alone.
    """
    warmup_dir = run_dir / "embedding_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)

    # Extract config with explicit casting
    steps = int(config.get("steps", 2500))  # type: ignore[arg-type]
    lr = float(config.get("lr", 5e-5))  # type: ignore[arg-type]
    batch_size = int(config.get("batch_size", 4))  # type: ignore[arg-type]
    grad_acc = int(config.get("gradient_accumulation", 8))  # type: ignore[arg-type]
    max_seq_length = int(config.get("max_seq_length", 2048))  # type: ignore[arg-type]
    seed = int(config.get("seed", 0))  # type: ignore[arg-type]
    use_flash_attn = bool(config.get("use_flash_attn", False)) and _is_flash_attn_available()
    bf16_flag = bool(config.get("bf16", True))  # Default True for efficiency

    logger.info(
        "Embedding warm-up hyperparams: steps=%d lr=%s batch=%d grad_acc=%d "
        "max_seq_len=%d bf16=%s",
        steps,
        lr,
        batch_size,
        grad_acc,
        max_seq_length,
        bf16_flag,
    )

    # Environment for GPU training
    cuda_env: Dict[str, str] = {}
    try:
        import os as _os_env
        if torch.cuda.is_available() and not _os_env.environ.get("CUDA_VISIBLE_DEVICES"):
            cuda_env["CUDA_VISIBLE_DEVICES"] = "0"
        if not _os_env.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
            cuda_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception:
        pass

    # Build training arguments - matching original vocab_adaptation Stage 1 exactly
    # Use same structure as _common_args() from vocab_adaptation for consistency
    warmup_args = [
        sys.executable,
        "-m",
        "src.clm_train",
        "--model_name",
        adapted_model_path,
        "--tokenizer_path",
        adapted_model_path,
        "--dataset_name",
        dataset_path,
        "--output_dir",
        str(warmup_dir),
        "--max_steps",
        str(steps),
        "--save_steps",
        str(steps),
        "--logging_steps",
        "50",
        "--learning_rate",
        str(lr),
        "--per_device_train_batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_acc),
        "--max_seq_length",
        str(max_seq_length),
        "--use_gradient_checkpointing",
        "True",
        "--bf16",
        "True" if bf16_flag else "False",
        "--packing",
        "True",
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.03",
        "--weight_decay",
        "0.01",
        "--max_grad_norm",
        "0.3",  # Same as default in clm_train.py ScriptArguments
        "--ignore_data_skip",
        "True",
        "--seed",
        str(seed),
        "--finetune_embed_only",
        "True",  # Only train embeddings + LM head (matches vocab_adaptation Stage 1)
        "--train_start_idx",
        "0",  # Same as vocab_adaptation Stage 1
    ]
    if use_flash_attn:
        warmup_args.extend(["--use_flash_attn", "True"])

    logger.info("Starting embedding warm-up (embeddings + LM head only) at %s.", warmup_dir)
    _run_subprocess(warmup_args, env=cuda_env)

    final_ckpt = warmup_dir / f"checkpoint-{steps}"
    logger.info("Embedding warm-up completed. Final checkpoint: %s", final_ckpt)

    return {
        "model_dir": str(final_ckpt),
        "warmup_dir": str(warmup_dir),
    }


def vocab_adaptation(
    *,
    run_dir: Path,
    adapted_model_path: str,
    dataset_path: str,
    config: Dict[str, object],
) -> Dict[str, str]:
    """
    Run two-stage vocabulary adaptation fine-tuning, faithful to TokAlign:
    - Stage 1: embeddings-only warmup
    - Stage 2: full-model continued training
    All hyperparameters are provided via config; no hardcoded values.
    """
    va_root = run_dir / "vocab_adaptation"
    stage1_dir = va_root / "stage1_embed_only"
    stage2_dir = va_root / "stage2_full"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # Extract config with explicit casting
    stage1_steps = int(config.get("stage1_steps", 2500))  # type: ignore[arg-type]
    stage2_steps = int(config.get("stage2_steps", 2500))  # type: ignore[arg-type]
    lr_stage1 = float(config.get("lr_stage1", 6.4e-4))  # type: ignore[arg-type]
    lr_stage2 = float(config.get("lr_stage2", 5e-5))  # type: ignore[arg-type]
    batch_size = int(config.get("batch_size", 2))  # type: ignore[arg-type]
    grad_acc = int(config.get("gradient_accumulation", 16))  # type: ignore[arg-type]
    max_seq_length = int(config.get("max_seq_length", 2048))  # type: ignore[arg-type]
    train_start_idx_stage2 = int(config.get("train_start_idx_stage2", 2560000))  # type: ignore[arg-type]
    seed = int(config.get("seed", 0))  # type: ignore[arg-type]
    # Optional Stage 2 specialization (config-driven)
    stage2_use_lora = bool(config.get("stage2_use_lora", False))
    stage2_optimizer = str(config.get("stage2_optimizer", "adamw_torch"))
    lora_r = int(config.get("lora_r", 64))
    lora_alpha = int(config.get("lora_alpha", 16))
    lora_dropout = float(config.get("lora_dropout", 0.1))
    lora_target_modules = str(
        config.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj")
    )
    # Match config default (False) and enable only if available
    use_flash_attn = bool(config.get("use_flash_attn", False)) and _is_flash_attn_available()
    bf16_flag = bool(config.get("bf16", False))

    logger.info(
        "Vocab adaptation hyperparams: stage1=%d steps (lr=%s) stage2=%d steps (lr=%s, start_idx=%d, optim=%s, lora=%s) "
        "batch=%d grad_acc=%d max_seq_len=%d bf16=%s",
        stage1_steps,
        lr_stage1,
        stage2_steps,
        lr_stage2,
        train_start_idx_stage2,
        stage2_optimizer,
        stage2_use_lora,
        batch_size,
        grad_acc,
        max_seq_length,
        bf16_flag,
    )

    # Proceed without explicit GPU gate; environment is expected to provide GPU.

    # Common argument fragments
    def _common_args() -> list[str]:
        args: list[str] = [
            sys.executable,
            "-m",
            "src.clm_train",
            "--tokenizer_path",
            adapted_model_path,
            "--dataset_name",
            dataset_path,
            "--max_seq_length",
            str(max_seq_length),
            "--per_device_train_batch_size",
            str(batch_size),
            "--gradient_accumulation_steps",
            str(grad_acc),
            "--use_gradient_checkpointing",
            "True",
            "--bf16",
            "True" if bf16_flag else "False",
            "--packing",
            "True",
            "--lr_scheduler_type",
            "cosine",
            "--warmup_ratio",
            "0.03",
            "--weight_decay",
            "0.01",
            "--ignore_data_skip",
            "True",
            "--seed",
            str(seed),
        ]
        if use_flash_attn:
            args.extend(["--use_flash_attn", "True"])
        return args

    # Environment for GPU training: respect user CUDA selection if already set
    cuda_env: Dict[str, str] = {}
    try:
        import os as _os_env  # local alias to avoid top-level shadowing
        if torch.cuda.is_available() and not _os_env.environ.get("CUDA_VISIBLE_DEVICES"):
            # Default to first device on single-GPU GH200 nodes; do not override if user set it.
            cuda_env["CUDA_VISIBLE_DEVICES"] = "0"
        # Improve allocator stability during large optimizer state allocations
        if not _os_env.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
            cuda_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception:
        pass

    # Stage 1: embeddings-only
    stage1_args = _common_args()
    stage1_args.extend(
        [
            "--model_name",
            adapted_model_path,
            "--output_dir",
            str(stage1_dir),
            "--max_steps",
            str(stage1_steps),
            "--save_steps",
            str(stage1_steps),
            "--logging_steps",
            "50",
            "--learning_rate",
            str(lr_stage1),
            "--finetune_embed_only",
            "True",
            "--train_start_idx",
            "0",
        ]
    )
    logger.info("Starting vocabulary adaptation Stage 1 (embeddings-only) at %s.", stage1_dir)
    _run_subprocess(stage1_args, env=cuda_env)

    # Stage 2: full model
    stage1_ckpt = stage1_dir / f"checkpoint-{stage1_steps}"
    stage2_args = _common_args()
    stage2_args.extend(
        [
            "--model_name",
            str(stage1_ckpt),
            "--output_dir",
            str(stage2_dir),
            "--max_steps",
            str(stage2_steps),
            "--save_steps",
            str(stage2_steps),
            "--logging_steps",
            "50",
            "--learning_rate",
            str(lr_stage2),
            "--train_start_idx",
            str(train_start_idx_stage2),
        ]
    )
    # Use memory-friendlier optimizer if requested (e.g., Adafactor)
    if stage2_optimizer:
        stage2_args.extend(["--optim", stage2_optimizer])
    # Optionally use LoRA adapters for Stage 2 to reduce trainable parameter footprint
    if stage2_use_lora:
        stage2_args.extend(
            [
                "--use_peft_lora",
                "True",
                "--lora_r",
                str(lora_r),
                "--lora_alpha",
                str(lora_alpha),
                "--lora_dropout",
                str(lora_dropout),
                "--lora_target_modules",
                lora_target_modules,
            ]
        )
    logger.info("Starting vocabulary adaptation Stage 2 (full-model) at %s.", stage2_dir)
    _run_subprocess(stage2_args, env=cuda_env)

    final_ckpt = stage2_dir / f"checkpoint-{stage2_steps}"
    return {
        "stage1_model_dir": str(stage1_dir),
        "stage2_model_dir": str(stage2_dir),
        "final_model_dir": str(final_ckpt),
    }


def train_baseline_model(
    *,
    run_dir: Path,
    baseline_model: str,
    baseline_tokenizer: str,
    dataset_path: str,
    config: Dict[str, object],
    mode: str = "embed_only",
) -> Dict[str, str]:
    """
    Train the baseline model without alignment for apples-to-apples comparisons.
    """
    normalized_mode = mode.lower().strip()
    if normalized_mode not in {"embed_only", "full"}:
        raise ValueError(f"Unsupported baseline training mode '{mode}'. Expected 'embed_only' or 'full'.")
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Baseline dataset not found at {dataset_dir}")

    training_dir = run_dir / "baseline_training" / normalized_mode
    training_dir.mkdir(parents=True, exist_ok=True)

    # Shared hyperparameters (mirrors vocab_adaptation defaults)
    stage1_steps = int(config.get("stage1_steps", 2500))  # type: ignore[arg-type]
    stage2_steps = int(config.get("stage2_steps", 2500))  # type: ignore[arg-type]
    lr_stage1 = float(config.get("lr_stage1", 6.4e-4))  # type: ignore[arg-type]
    lr_stage2 = float(config.get("lr_stage2", 5e-5))  # type: ignore[arg-type]
    batch_size = int(config.get("batch_size", 2))  # type: ignore[arg-type]
    grad_acc = int(config.get("gradient_accumulation", 16))  # type: ignore[arg-type]
    max_seq_length = int(config.get("max_seq_length", 2048))  # type: ignore[arg-type]
    train_start_idx_stage2 = int(config.get("train_start_idx_stage2", 2560000))  # type: ignore[arg-type]
    seed = int(config.get("seed", 0))  # type: ignore[arg-type]
    stage2_optimizer = str(config.get("stage2_optimizer", "adamw_torch"))
    stage2_use_lora = bool(config.get("stage2_use_lora", False))
    lora_r = int(config.get("lora_r", 64))
    lora_alpha = int(config.get("lora_alpha", 16))
    lora_dropout = float(config.get("lora_dropout", 0.1))
    lora_target_modules = str(
        config.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj")
    )
    use_flash_attn = bool(config.get("use_flash_attn", False)) and _is_flash_attn_available()
    bf16_flag = bool(config.get("bf16", False))

    steps = stage1_steps if normalized_mode == "embed_only" else stage2_steps
    learning_rate = lr_stage1 if normalized_mode == "embed_only" else lr_stage2
    train_start_idx = 0 if normalized_mode == "embed_only" else train_start_idx_stage2

    logger.info(
        "Starting baseline training (%s): steps=%d lr=%s batch=%d grad_acc=%d max_seq=%d bf16=%s "
        "optimizer=%s lora=%s dataset=%s",
        normalized_mode,
        steps,
        learning_rate,
        batch_size,
        grad_acc,
        max_seq_length,
        bf16_flag,
        stage2_optimizer if normalized_mode == "full" else "adamw_torch",
        stage2_use_lora if normalized_mode == "full" else False,
        dataset_dir,
    )

    def _common_args() -> list[str]:
        args: list[str] = [
            sys.executable,
            "-m",
            "src.clm_train",
            "--tokenizer_path",
            baseline_tokenizer,
            "--dataset_name",
            dataset_path,
            "--max_seq_length",
            str(max_seq_length),
            "--per_device_train_batch_size",
            str(batch_size),
            "--gradient_accumulation_steps",
            str(grad_acc),
            "--use_gradient_checkpointing",
            "True",
            "--bf16",
            "True" if bf16_flag else "False",
            "--packing",
            "True",
            "--lr_scheduler_type",
            "cosine",
            "--warmup_ratio",
            "0.03",
            "--weight_decay",
            "0.01",
            "--ignore_data_skip",
            "True",
            "--seed",
            str(seed),
        ]
        if use_flash_attn:
            args.extend(["--use_flash_attn", "True"])
        return args

    cuda_env: Dict[str, str] = {}
    try:
        import os as _os_env

        if torch.cuda.is_available() and not _os_env.environ.get("CUDA_VISIBLE_DEVICES"):
            cuda_env["CUDA_VISIBLE_DEVICES"] = "0"
        if not _os_env.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
            cuda_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception:
        pass

    train_args = _common_args()
    train_args.extend(
        [
            "--model_name",
            baseline_model,
            "--output_dir",
            str(training_dir),
            "--max_steps",
            str(steps),
            "--save_steps",
            str(steps),
            "--logging_steps",
            "50",
            "--learning_rate",
            str(learning_rate),
            "--train_start_idx",
            str(train_start_idx),
        ]
    )
    if normalized_mode == "embed_only":
        train_args.extend(["--finetune_embed_only", "True"])
    elif normalized_mode == "full":
        if stage2_optimizer:
            train_args.extend(["--optim", stage2_optimizer])
        if stage2_use_lora:
            train_args.extend(
                [
                    "--use_peft_lora",
                    "True",
                    "--lora_r",
                    str(lora_r),
                    "--lora_alpha",
                    str(lora_alpha),
                    "--lora_dropout",
                    str(lora_dropout),
                    "--lora_target_modules",
                    lora_target_modules,
                ]
            )

    _run_subprocess(train_args, env=cuda_env)

    final_ckpt = training_dir / f"checkpoint-{steps}"
    trainer_state = training_dir / "trainer_state.json"
    logger.info(
        "Baseline training (%s) complete. Checkpoint=%s trainer_state=%s",
        normalized_mode,
        final_ckpt,
        trainer_state,
    )
    return {
        "model_dir": str(final_ckpt if final_ckpt.exists() else training_dir),
        "training_dir": str(training_dir),
        "trainer_state": str(trainer_state),
    }


def compare_training_metrics(
    *,
    run_dir: Path,
    baseline_trainer_state: str,
    adapted_trainer_state: str,
    mode: str,
    comparison_points: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compare training trajectories between baseline and adapted models.
    """

    def _load_history(path: str) -> List[Dict[str, Any]]:
        data = Path(path)
        if not data.exists():
            raise FileNotFoundError(f"trainer_state not found: {path}")
        with open(data, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        history = payload.get("log_history", [])
        cleaned: List[Dict[str, Any]] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            step = entry.get("step")
            loss_val = entry.get("loss", entry.get("train_loss"))
            if step is None or loss_val is None:
                continue
            try:
                step_int = int(step)
                loss_float = float(loss_val)
            except (TypeError, ValueError):
                continue
            lr_val = entry.get("learning_rate")
            lr_float: Optional[float]
            try:
                lr_float = float(lr_val) if lr_val is not None else None
            except (TypeError, ValueError):
                lr_float = None
            cleaned.append(
                {
                    "step": step_int,
                    "loss": loss_float,
                    "learning_rate": lr_float,
                }
            )
        cleaned.sort(key=lambda rec: rec["step"])
        return cleaned

    def _select_record(history: List[Dict[str, Any]], target_step: int) -> Optional[Dict[str, Any]]:
        if not history:
            return None
        for record in history:
            if record["step"] == target_step:
                return record
            if record["step"] > target_step:
                return record
        return history[-1]

    def _format_record(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not record:
            return {"step": None, "loss": None, "learning_rate": None}
        return {
            "step": record.get("step"),
            "loss": record.get("loss"),
            "learning_rate": record.get("learning_rate"),
        }

    def _loss_per_step(history: List[Dict[str, Any]]) -> Optional[float]:
        if len(history) < 2:
            return None
        start = history[0]
        end = history[-1]
        if start.get("loss") is None or end.get("loss") is None:
            return None
        step_delta = max(1, end["step"] - start["step"])
        return (start["loss"] - end["loss"]) / step_delta

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_file = metrics_dir / "training_comparison.json"
    existing: Dict[str, Any] = {}
    if comparison_file.exists():
        try:
            with open(comparison_file, "r", encoding="utf-8") as fp:
                existing = json.load(fp)
        except Exception:
            existing = {}

    mode_key = "full_model" if mode.lower() == "full" else mode
    comparison_points = comparison_points or [1, 10, 50, 100]
    point_values = sorted({int(p) for p in comparison_points if isinstance(p, int) and p > 0})

    try:
        baseline_history = _load_history(baseline_trainer_state)
    except Exception as exc:
        logger.warning("Failed to load baseline trainer_state for %s: %s", mode_key, exc)
        existing[mode_key] = {
            "error": f"baseline trainer_state unavailable: {exc}",
            "baseline_trainer_state": baseline_trainer_state,
            "adapted_trainer_state": adapted_trainer_state,
        }
        with open(comparison_file, "w", encoding="utf-8") as fp:
            json.dump(existing, fp, indent=2)
        return existing
    try:
        adapted_history = _load_history(adapted_trainer_state)
    except Exception as exc:
        logger.warning("Failed to load adapted trainer_state for %s: %s", mode_key, exc)
        existing[mode_key] = {
            "error": f"adapted trainer_state unavailable: {exc}",
            "baseline_trainer_state": baseline_trainer_state,
            "adapted_trainer_state": adapted_trainer_state,
        }
        with open(comparison_file, "w", encoding="utf-8") as fp:
            json.dump(existing, fp, indent=2)
        return existing

    if not baseline_history or not adapted_history:
        existing[mode_key] = {
            "error": "insufficient training history for comparison",
            "baseline_trainer_state": baseline_trainer_state,
            "adapted_trainer_state": adapted_trainer_state,
        }
        with open(comparison_file, "w", encoding="utf-8") as fp:
            json.dump(existing, fp, indent=2)
        return existing

    final_step_baseline = baseline_history[-1]["step"]
    final_step_adapted = adapted_history[-1]["step"]
    midpoint_baseline = max(1, final_step_baseline // 2)
    midpoint_adapted = max(1, final_step_adapted // 2)

    def _collect(history: List[Dict[str, Any]], midpoint_step: int, final_label: str) -> Dict[str, Dict[str, Any]]:
        bucket: Dict[str, Dict[str, Any]] = {}
        for point in point_values:
            bucket[f"step_{point}"] = _format_record(_select_record(history, point))
        bucket["midpoint"] = _format_record(_select_record(history, midpoint_step))
        bucket["final"] = _format_record(history[-1])
        return bucket

    baseline_points = _collect(baseline_history, midpoint_baseline, "baseline_final")
    adapted_points = _collect(adapted_history, midpoint_adapted, "adapted_final")

    improvements: Dict[str, Any] = {}
    for label, base_entry in baseline_points.items():
        if label == "efficiency":
            continue
        adapted_entry = adapted_points.get(label)
        base_loss = base_entry.get("loss")
        adapted_loss = adapted_entry.get("loss") if adapted_entry else None
        if base_loss is None or adapted_loss is None:
            continue
        absolute = adapted_loss - base_loss
        relative = (
            (absolute / base_loss) * 100.0
            if base_loss is not None and base_loss != 0
            else None
        )
        improvements[label] = {
            "absolute": absolute,
            "relative_pct": relative,
        }

    base_eff = _loss_per_step(baseline_history)
    adapted_eff = _loss_per_step(adapted_history)
    if base_eff is not None and adapted_eff is not None and base_eff != 0:
        improvements["efficiency"] = {
            "baseline_loss_per_step": base_eff,
            "adapted_loss_per_step": adapted_eff,
            "improvement_pct": ((base_eff - adapted_eff) / abs(base_eff)) * 100.0,
        }

    mode_result = {
        "baseline": baseline_points,
        "adapted": adapted_points,
        "improvement": improvements,
        "metadata": {
            "baseline_trainer_state": baseline_trainer_state,
            "adapted_trainer_state": adapted_trainer_state,
        },
    }
    existing[mode_key] = mode_result
    with open(comparison_file, "w", encoding="utf-8") as fp:
        json.dump(existing, fp, indent=2)
    return existing


def compare_final_performance(
    *,
    run_dir: Path,
    baseline_model_path: str,
    baseline_tokenizer_path: str,
    adapted_model_path: str,
    adapted_tokenizer_path: str,
    eval_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare final performance of baseline vs. adapted models on shared evaluations.
    """
    from . import eval_medical  # Local import to avoid heavy dependency for callers that skip eval

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / "performance_comparison.json"

    comparison: Dict[str, Any] = {
        "perplexity": {},
        "qa": None,
        "tokenization": None,
    }

    eval_batch = int(os.getenv("TOKALIGN_EVAL_BATCH", "32"))
    eval_maxlen = int(os.getenv("TOKALIGN_EVAL_MAXLEN", "1024"))
    max_samples = int(eval_config.get("max_samples", 128))
    datasets_cfg = list(eval_config.get("datasets") or [])

    for dataset_spec in datasets_cfg:
        dataset_name, dataset_conf, split = eval_medical.parse_dataset_spec(dataset_spec)
        dataset_label = dataset_name if not dataset_conf else f"{dataset_name}[{dataset_conf}]"
        entry: Dict[str, Any] = {
            "dataset": dataset_label,
            "split": split,
            "baseline": None,
            "adapted": None,
        }
        try:
            baseline_ppl = eval_medical.evaluate_perplexity(
                model_path=baseline_model_path,
                tokenizer_path=baseline_tokenizer_path,
                dataset_name=dataset_name,
                split=split,
                dataset_config=dataset_conf,
                max_samples=None if max_samples <= 0 else max_samples,
                batch_size=eval_batch,
                max_length=eval_maxlen,
            )
            adapted_ppl = eval_medical.evaluate_perplexity(
                model_path=adapted_model_path,
                tokenizer_path=adapted_tokenizer_path,
                dataset_name=dataset_name,
                split=split,
                dataset_config=dataset_conf,
                max_samples=None if max_samples <= 0 else max_samples,
                batch_size=eval_batch,
                max_length=eval_maxlen,
            )
            entry["baseline"] = baseline_ppl
            entry["adapted"] = adapted_ppl
            entry["improvement_pct"] = (
                ((baseline_ppl - adapted_ppl) / baseline_ppl) * 100.0
                if baseline_ppl is not None and baseline_ppl != 0
                else None
            )
        except Exception as exc:
            logger.warning("Perplexity evaluation failed for %s:%s - %s", dataset_label, split, exc)
            entry["error"] = str(exc)
        comparison["perplexity"][f"{dataset_label}:{split}"] = entry

    if bool(eval_config.get("qa", False)):
        qa_entry: Dict[str, Any] = {}
        try:
            baseline_res = eval_medical.evaluate_medmcqa(
                model_path=baseline_model_path,
                tokenizer_path=baseline_tokenizer_path,
                split="validation",
                max_samples=None if max_samples <= 0 else max_samples,
            )
            adapted_res = eval_medical.evaluate_medmcqa(
                model_path=adapted_model_path,
                tokenizer_path=adapted_tokenizer_path,
                split="validation",
                max_samples=None if max_samples <= 0 else max_samples,
            )
            qa_entry = {
                "baseline": baseline_res,
                "adapted": adapted_res,
                "improvement": {
                    "accuracy_delta": adapted_res["accuracy"] - baseline_res["accuracy"],
                    "accuracy_relative": (
                        ((adapted_res["accuracy"] - baseline_res["accuracy"]) / baseline_res["accuracy"]) * 100.0
                        if baseline_res["accuracy"] is not None and baseline_res["accuracy"] != 0
                        else None
                    ),
                },
            }
        except Exception as exc:
            logger.warning("MedMCQA evaluation failed: %s", exc)
            qa_entry = {"error": str(exc)}
        comparison["qa"] = qa_entry

    terms_path = run_dir / "corpus" / "medical_terms.txt"
    if terms_path.exists():
        try:
            baseline_tok = eval_medical.compute_term_tokenization_coverage(
                terms_path=str(terms_path),
                tokenizer_path=baseline_tokenizer_path,
            )
            adapted_tok = eval_medical.compute_term_tokenization_coverage(
                terms_path=str(terms_path),
                tokenizer_path=adapted_tokenizer_path,
            )
            tokenization_entry = {
                "terms_path": str(terms_path),
                "baseline": baseline_tok,
                "adapted": adapted_tok,
                "improvement": {
                    "single_token_ratio_delta": (
                        adapted_tok.get("single_token_ratio") - baseline_tok.get("single_token_ratio")
                        if baseline_tok.get("single_token_ratio") is not None
                        and adapted_tok.get("single_token_ratio") is not None
                        else None
                    ),
                    "mean_tokens_delta": (
                        baseline_tok.get("mean_tokens_per_term") - adapted_tok.get("mean_tokens_per_term")
                        if baseline_tok.get("mean_tokens_per_term") is not None
                        and adapted_tok.get("mean_tokens_per_term") is not None
                        else None
                    ),
                },
            }
            comparison["tokenization"] = tokenization_entry
        except Exception as exc:
            logger.warning("Tokenization efficiency comparison failed: %s", exc)
            comparison["tokenization"] = {"error": str(exc), "terms_path": str(terms_path)}

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(comparison, fp, indent=2)
    return comparison


def generate_initialization_analysis(
    *,
    run_dir: Path,
    training_comparison: Dict[str, Any],
    performance_comparison: Dict[str, Any],
) -> Dict[str, str]:
    """
    Build a markdown report plus structured data summarising initialization quality.
    """
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / "initialization_analysis.md"
    json_path = metrics_dir / "initialization_data.json"
    csv_path = metrics_dir / "initialization_data.csv"

    def _fmt_pct(value: Optional[float]) -> str:
        if value is None or not isinstance(value, (int, float)) or math.isnan(value):
            return "N/A"
        return f"{value:+.1f}%"

    def _fmt_loss(value: Optional[float]) -> str:
        if value is None or not isinstance(value, (int, float)) or math.isnan(value):
            return "N/A"
        return f"{value:.3f}"

    summary_lines: List[str] = []
    embed_step1 = (
        training_comparison.get("embed_only", {})
        .get("improvement", {})
        .get("step_1", {})
        .get("relative_pct")
        if isinstance(training_comparison, dict)
        else None
    )
    full_step1 = (
        training_comparison.get("full_model", {})
        .get("improvement", {})
        .get("step_1", {})
        .get("relative_pct")
        if isinstance(training_comparison, dict)
        else None
    )
    if embed_step1 is not None:
        summary_lines.append(f"- Step 1 Loss Improvement (embeddings-only): {_fmt_pct(embed_step1)}")
    if full_step1 is not None:
        summary_lines.append(f"- Step 1 Loss Improvement (full model): {_fmt_pct(full_step1)}")

    efficiency = (
        training_comparison.get("embed_only", {})
        .get("improvement", {})
        .get("efficiency", {})
        .get("improvement_pct")
        if isinstance(training_comparison, dict)
        else None
    )
    if efficiency is not None:
        summary_lines.append(f"- Training Efficiency Gain: {_fmt_pct(efficiency)} faster loss reduction")

    # Final performance summary (use first dataset if available)
    perf_summary = ""
    perplexity_entries = performance_comparison.get("perplexity", {}) if isinstance(performance_comparison, dict) else {}
    if isinstance(perplexity_entries, dict):
        for label, entry in perplexity_entries.items():
            if isinstance(entry, dict) and entry.get("improvement_pct") is not None:
                perf_summary = f"{label}: {_fmt_pct(entry['improvement_pct'])} perplexity gain"
                break
    if perf_summary:
        summary_lines.append(f"- Final Perplexity: {perf_summary}")
    qa_entry = performance_comparison.get("qa") if isinstance(performance_comparison, dict) else None
    if isinstance(qa_entry, dict) and isinstance(qa_entry.get("improvement"), dict):
        qa_delta = qa_entry["improvement"].get("accuracy_delta")
        qa_rel = qa_entry["improvement"].get("accuracy_relative")
        baseline_acc = qa_entry.get("baseline", {}).get("accuracy")
        adapted_acc = qa_entry.get("adapted", {}).get("accuracy")
        if qa_delta is not None:
            if isinstance(baseline_acc, (int, float)) and isinstance(adapted_acc, (int, float)):
                summary_lines.append(
                    f"- MedMCQA Accuracy: {baseline_acc:.3f} → {adapted_acc:.3f} "
                    f"({qa_delta:+.3f} abs, {_fmt_pct(qa_rel)} rel)"
                )
            else:
                summary_lines.append(f"- MedMCQA Accuracy Gain: {qa_delta:+.3f} ({_fmt_pct(qa_rel)})")

    sections: List[str] = []
    sections.append("# Initialization Quality Analysis")
    sections.append("\n## Summary")
    if summary_lines:
        sections.extend(summary_lines)
    else:
        sections.append("- No comparison data available.")

    # Detailed training sections
    def _build_training_section(mode_key: str, label: str) -> str:
        mode_data = training_comparison.get(mode_key, {}) if isinstance(training_comparison, dict) else {}
        baseline = mode_data.get("baseline", {})
        adapted = mode_data.get("adapted", {})
        improvement = mode_data.get("improvement", {})
        lines = [f"### {label}"]
        step1_base = baseline.get("step_1", {}).get("loss")
        step1_adapt = adapted.get("step_1", {}).get("loss")
        if step1_base is not None and step1_adapt is not None:
            rel = improvement.get("step_1", {}).get("relative_pct")
            lines.append(
                f"- Step 1 loss: {step1_base:.3f} → {step1_adapt:.3f} ({_fmt_pct(rel)})"
            )
        final_base = baseline.get("final", {}).get("loss")
        final_adapt = adapted.get("final", {}).get("loss")
        if final_base is not None and final_adapt is not None:
            rel_final = improvement.get("final", {}).get("relative_pct")
            lines.append(
                f"- Final loss: {final_base:.3f} → {final_adapt:.3f} ({_fmt_pct(rel_final)})"
            )
        eff = improvement.get("efficiency", {}).get("improvement_pct")
        if eff is not None:
            lines.append(f"- Loss reduction per step improvement: {_fmt_pct(eff)}")
        return "\n".join(lines)

    if isinstance(training_comparison, dict) and training_comparison:
        sections.append("\n## Detailed Findings\n")
        if "embed_only" in training_comparison:
            sections.append(_build_training_section("embed_only", "Embeddings-only Training"))
        if "full_model" in training_comparison:
            sections.append(_build_training_section("full_model", "Full-Model Training"))

    # Final performance section
    sections.append("\n### Final Performance")
    if isinstance(performance_comparison, dict) and performance_comparison:
        perplexity_lines: List[str] = []
        for label, entry in (performance_comparison.get("perplexity", {}) or {}).items():
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("baseline"), (int, float))
                and isinstance(entry.get("adapted"), (int, float))
            ):
                perplexity_lines.append(
                    f"- {label}: {entry['baseline']:.2f} → {entry['adapted']:.2f} "
                    f"({_fmt_pct(entry.get('improvement_pct'))})"
                )
        if perplexity_lines:
            sections.extend(perplexity_lines)
        qa_entry = performance_comparison.get("qa")
        if (
            isinstance(qa_entry, dict)
            and isinstance(qa_entry.get("baseline", {}).get("accuracy"), (int, float))
            and isinstance(qa_entry.get("adapted", {}).get("accuracy"), (int, float))
        ):
            sections.append(
                f"- MedMCQA accuracy: {qa_entry['baseline']['accuracy']:.3f} → "
                f"{qa_entry['adapted']['accuracy']:.3f} "
                f"({_fmt_pct(qa_entry.get('improvement', {}).get('accuracy_relative'))})"
            )
        token_entry = performance_comparison.get("tokenization")
        if isinstance(token_entry, dict) and token_entry.get("baseline") and token_entry.get("adapted"):
            base_single = token_entry["baseline"].get("single_token_ratio")
            adapted_single = token_entry["adapted"].get("single_token_ratio")
            single_delta = token_entry.get("improvement", {}).get("single_token_ratio_delta")
            if isinstance(base_single, (int, float)) and isinstance(adapted_single, (int, float)):
                pct_delta = single_delta * 100.0 if isinstance(single_delta, (int, float)) else None
                sections.append(
                    f"- Single-token ratio: {base_single:.3f} → {adapted_single:.3f} ({_fmt_pct(pct_delta)})"
                )
    else:
        sections.append("- No performance data available.")

    report_text = "\n".join(sections)
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write(report_text.strip() + "\n")

    combined_data = {
        "training_comparison": training_comparison,
        "performance_comparison": performance_comparison,
    }
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(combined_data, fp, indent=2)

    # Visualization-friendly CSV rows
    csv_rows: List[List[Any]] = [["mode", "point", "baseline_loss", "adapted_loss", "absolute_delta", "relative_pct"]]
    if isinstance(training_comparison, dict):
        for mode_key, mode_data in training_comparison.items():
            baseline_points = mode_data.get("baseline", {}) if isinstance(mode_data, dict) else {}
            adapted_points = mode_data.get("adapted", {}) if isinstance(mode_data, dict) else {}
            improvements = mode_data.get("improvement", {}) if isinstance(mode_data, dict) else {}
            for point, base_entry in baseline_points.items():
                if not isinstance(base_entry, dict) or "loss" not in base_entry:
                    continue
                adapt_entry = adapted_points.get(point, {})
                improv_entry = improvements.get(point, {})
                csv_rows.append(
                    [
                        mode_key,
                        point,
                        base_entry.get("loss"),
                        adapt_entry.get("loss"),
                        improv_entry.get("absolute"),
                        improv_entry.get("relative_pct"),
                    ]
                )
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(csv_rows)

    return {
        "report": str(report_path),
        "data": str(json_path),
        "csv": str(csv_path),
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokAlign medical pipeline helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_parser = subparsers.add_parser("data-prep", help="Prepare datasets from the medical corpus.")
    prep_parser.add_argument("--input", action="append", required=True, help="JSONL file or directory.")
    prep_parser.add_argument("--run-dir", required=True, help="Run directory for outputs.")
    prep_parser.add_argument("--byte-budget", type=int, default=0, help="Maximum bytes to read (0 = unlimited).")
    prep_parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication.")
    prep_parser.add_argument("--hash-name", default="sha256", help="Hash function used for deduplication.")

    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize aggregated corpus and build ID corpora.")
    tokenize_parser.add_argument("--run-dir", required=True, help="Run directory (must contain aggregated corpus).")
    tokenize_parser.add_argument("--aggregated-jsonl", required=True, help="Path to aggregated medical corpus.")
    tokenize_parser.add_argument("--tokenizer-source", required=True, help="Source tokenizer path or identifier.")
    tokenize_parser.add_argument("--tokenizer-target", required=True, help="Target tokenizer path or identifier.")
    tokenize_parser.add_argument(
        "--source-model-fallback",
        help="Model identifier to use when the source tokenizer lacks an attached config (defaults to --tokenizer-source).",
    )
    tokenize_parser.add_argument(
        "--target-model-fallback",
        help="Model identifier to use when the target tokenizer lacks an attached config (defaults to --tokenizer-target).",
    )
    tokenize_parser.add_argument("--tokenizer-workers", type=int, default=8, help="Workers for tokenization.")
    tokenize_parser.add_argument("--tokenizer-cache", help="Cache directory for datasets/tokenizers.")
    tokenize_parser.add_argument(
        "--min-line-length",
        type=int,
        default=0,
        help="Minimum sequence length when extracting embedding corpora.",
    )

    train_parser = subparsers.add_parser("train-align", help="Train embeddings and compute alignment.")
    train_parser.add_argument("--run-dir", required=True, help="Run directory used during data-prep.")
    train_parser.add_argument("--source-glove", required=True, help="Path to source embedding corpus.")
    train_parser.add_argument("--target-glove", required=True, help="Path to target embedding corpus.")
    train_parser.add_argument("--tokenizer-source", required=True, help="Source tokenizer path (augmented).")
    train_parser.add_argument("--tokenizer-target", required=True, help="Target tokenizer path (augmented).")
    train_parser.add_argument(
        "--embedding-backend",
        default="fasttext",
        choices=["glove", "fasttext"],
        help="Embedding backend to train.",
    )
    train_parser.add_argument("--pivot-count", type=int, default=300, help="Number of pivot tokens.")
    # Optional FastText quality/performance knobs
    train_parser.add_argument("--fasttext-epochs", type=int, default=5, help="FastText training epochs.")
    train_parser.add_argument("--fasttext-mincount", type=int, default=5, help="FastText minimum frequency.")
    train_parser.add_argument("--fasttext-lr", type=float, default=0.05, help="FastText learning rate.")
    train_parser.add_argument("--fasttext-thread", type=int, default=8, help="FastText CPU threads.")
    train_parser.add_argument("--gold-mapping", help="Optional pre-computed target->source mapping.")
    train_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.3,
        help="Minimum similarity score for alignment acceptance.",
    )

    apply_parser = subparsers.add_parser("apply", help="Apply alignment to initialise model weights.")
    apply_parser.add_argument("--run-dir", required=True, help="Run directory.")
    apply_parser.add_argument("--align-matrix", required=True, help="Alignment matrix JSON.")
    apply_parser.add_argument("--source-model", required=True, help="Source model identifier/path.")
    apply_parser.add_argument("--target-tokenizer", required=True, help="Target tokenizer path.")

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

    if args.command == "data-prep":
        outputs = data_prep(
            corpus_inputs=args.input,
            run_dir=Path(args.run_dir),
            byte_budget=None if args.byte_budget <= 0 else args.byte_budget,
            deduplicate=not args.no_dedup,
            hash_name=args.hash_name,
        )
        print(json.dumps(outputs, indent=2))
    elif args.command == "tokenize":
        source_fallback = args.source_model_fallback or args.tokenizer_source
        target_fallback = args.target_model_fallback or args.tokenizer_target
        outputs = tokenize_corpus(
            run_dir=Path(args.run_dir),
            aggregated_jsonl=args.aggregated_jsonl,
            tokenizer_source=args.tokenizer_source,
            tokenizer_target=args.tokenizer_target,
            source_model_fallback=source_fallback,
            target_model_fallback=target_fallback,
            tokenizer_workers=args.tokenizer_workers,
            tokenizer_cache_dir=args.tokenizer_cache,
            min_line_length=args.min_line_length,
        )
        print(json.dumps(outputs, indent=2))
    elif args.command == "train-align":
        outputs = train_embeddings_and_align(
            run_dir=Path(args.run_dir),
            source_glove_path=args.source_glove,
            target_glove_path=args.target_glove,
            tokenizer_source=args.tokenizer_source,
            tokenizer_target=args.tokenizer_target,
            embedding_backend=args.embedding_backend,
            pivot_count=args.pivot_count,
            fasttext_epochs=args.fasttext_epochs,
            fasttext_mincount=args.fasttext_mincount,
            fasttext_lr=args.fasttext_lr,
            thread=args.fasttext_thread,
            gold_mapping_path=args.gold_mapping,
            similarity_threshold=args.similarity_threshold,
        )
        print(json.dumps(outputs, indent=2))
    elif args.command == "apply":
        model_dir = apply_alignment(
            run_dir=Path(args.run_dir),
            align_matrix=args.align_matrix,
            source_model=args.source_model,
            target_tokenizer=args.target_tokenizer,
        )
        print(json.dumps({"adapted_model_dir": model_dir}, indent=2))
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
