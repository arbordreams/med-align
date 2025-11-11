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
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

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
    return datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")


def create_run_dir(run_root: Path = DEFAULT_RUN_ROOT, run_id: Optional[str] = None) -> Path:
    run_id = run_id or timestamp()
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_subprocess(
    cmd: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path | str] = None,
) -> None:
    logger.debug("Executing command: %s", " ".join(cmd))
    # Ensure user-installed packages take precedence over system packages
    user_site = os.path.expanduser("~/.local/lib/python3.12/site-packages")
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        pythonpath = f"{user_site}:{pythonpath}"
    else:
        pythonpath = user_site
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

    def _tokenize(tokenizer_path: str, output_subdir: str, fallback_model: str) -> Path:
        dataset_path = dataset_dir / output_subdir
        dataset_path.mkdir(parents=True, exist_ok=True)
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
            "--dataset_path_in_disk",
            str(dataset_path),
            "--preprocessing_num_workers",
            str(tokenizer_workers),
            "--cache_dir",
            cache_dir,
            "--output_dir",
            str(run_dir / "logs"),
        ]
        logger.info("Tokenizing corpus with %s.", tokenizer_path)
        _run_subprocess(cmd)
        return dataset_path

    source_dataset = _tokenize(tokenizer_source, "source", source_model_fallback)
    target_dataset = _tokenize(tokenizer_target, "target", target_model_fallback)

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
    gold_mapping_path: Optional[str] = None,
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
        # Train FastText embeddings using python binding.
        try:
            import fasttext  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "FastText backend selected but the `fasttext` module is unavailable. "
                "Install the prebuilt `fasttext-wheel>=0.9.2` package (pip install fasttext-wheel)."
            ) from exc

        for corpus_path, save_file in ((source_glove_path, vec_source), (target_glove_path, vec_target)):
            logger.info("Training FastText embeddings for %s.", corpus_path)
            model = fasttext.train_unsupervised(
                input=corpus_path,
                model="skipgram",
                dim=300,
                epoch=5,
            )
            model.save_model(str(save_file.with_suffix(".bin")))
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

    align_matrix_path = align_dir / "align_matrix.json"
    gold_mapping = gold_mapping_path or str(vocab_count_path)
    cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "src" / "cal_trans_matrix.py"),
        "-s",
        str(vec_source),
        "-t",
        str(vec_target),
        "-g",
        gold_mapping,
        "-o",
        str(align_matrix_path),
        "-r",
        "-n",
        str(pivot_count),
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
    train_parser.add_argument("--gold-mapping", help="Optional pre-computed target->source mapping.")

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
            gold_mapping_path=args.gold_mapping,
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

