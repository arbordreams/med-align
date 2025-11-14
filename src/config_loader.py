from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'PyYAML'. Please add 'pyyaml>=6.0.1' to requirements and install."
    ) from exc


def _expand_env_in_obj(obj: Any) -> Any:
    """
    Recursively expand environment variables in all strings within obj.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_in_obj(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def get_default_config() -> Dict[str, Any]:
    """
    Defaults that mirror argparse defaults in script/run_medical_pipeline.py.
    This ensures behavior is unchanged when --config is not provided.
    """
    return {
        "models": {
            "source_tokenizer": "BioMistral/BioMistral-7B",
            "target_tokenizer": "mistralai/Mistral-7B-v0.3",
            "source_model": "BioMistral/BioMistral-7B",
        },
        "corpus": {
            "size_gb": 1.0,
            "byte_budget": 0,
            "deduplicate": True,
            "hash_name": "sha256",
        },
        "term_mining": {
            "top_k": 500,
            "min_frequency": 5,
            "use_tfidf": False,
        },
        "embedding": {
            "backend": "fasttext",
            "fasttext": {
                "epochs": 5,
                "mincount": 5,
                "lr": 0.05,
                "thread": 8,
            },
        },
        "alignment": {
            "pivot_count": 300,
            "similarity_threshold": 0.3,
        },
        "tokenization": {
            "workers": 8,
            "cache_dir": None,
            "min_line_length": 0,
        },
        "evaluation": {
            "enabled": False,
            "datasets": [],
            "max_samples": 128,
            "qa": False,
            "baseline_model": "mistralai/Mistral-7B-v0.3",
        },
        "pipeline": {
            "run_root": "runs/tokenizer_adapt",
            "max_retries": 1,
            "retry_backoff": 5.0,
        },
        "embedding_warmup": {
            "enabled": False,
            "steps": 2500,
            "lr": 5e-5,
            "batch_size": 4,
            "gradient_accumulation": 8,
            "max_seq_length": 2048,
            "seed": 0,
            "use_flash_attn": False,
            "bf16": True,
        },
        "vocab_adaptation": {
            "enabled": True,
            "stage1_steps": 2500,
            "stage2_steps": 2500,
            "lr_stage1": 6.4e-4,
            "lr_stage2": 5e-5,
            "batch_size": 2,
            "gradient_accumulation": 16,
            "max_seq_length": 2048,
            "train_start_idx_stage2": 2560000,
            "seed": 0,
            "use_flash_attn": False,
            "bf16": False,
        },
    }


def _validate_config_structure(cfg: Dict[str, Any]) -> None:
    """
    Shallow validation to provide helpful error messages when config is malformed.
    We validate presence of top-level sections and the expected key types where practical.
    """
    def _coerce_to_float_if_numeric_str(value: Any) -> Any:
        """
        Best-effort coercion used for tolerant config loading:
        - If value is a string that represents a number (including scientific notation), return float(value).
        - Otherwise, return the original value unchanged.
        """
        if isinstance(value, str):
            stripped = value.strip()
            try:
                return float(stripped)
            except (ValueError, TypeError):
                return value
        return value
    required_sections = [
        "models",
        "corpus",
        "term_mining",
        "embedding",
        "alignment",
        "tokenization",
        "evaluation",
        "pipeline",
        "vocab_adaptation",
    ]
    for section in required_sections:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Config validation error: missing or invalid section '{section}'.")

    # Spot-check a few types for early failure with clear messages
    if not isinstance(cfg["models"].get("source_tokenizer", ""), str):
        raise ValueError("models.source_tokenizer must be a string")
    if not isinstance(cfg["models"].get("target_tokenizer", ""), str):
        raise ValueError("models.target_tokenizer must be a string")
    if not isinstance(cfg["models"].get("source_model", ""), str):
        raise ValueError("models.source_model must be a string")

    if not isinstance(cfg["corpus"].get("size_gb", 0.0), (int, float)):
        raise ValueError("corpus.size_gb must be a number")
    if not isinstance(cfg["corpus"].get("byte_budget", 0), int):
        raise ValueError("corpus.byte_budget must be an integer")
    if not isinstance(cfg["corpus"].get("deduplicate", True), bool):
        raise ValueError("corpus.deduplicate must be a boolean")

    if not isinstance(cfg["term_mining"].get("top_k", 0), int):
        raise ValueError("term_mining.top_k must be an integer")
    if not isinstance(cfg["term_mining"].get("min_frequency", 0), int):
        raise ValueError("term_mining.min_frequency must be an integer")
    if not isinstance(cfg["term_mining"].get("use_tfidf", False), bool):
        raise ValueError("term_mining.use_tfidf must be a boolean")

    if not isinstance(cfg["embedding"].get("backend", ""), str):
        raise ValueError("embedding.backend must be a string")
    ft = cfg["embedding"].get("fasttext", {})
    if not isinstance(ft, dict):
        raise ValueError("embedding.fasttext must be a mapping")
    if not isinstance(ft.get("epochs", 0), int):
        raise ValueError("embedding.fasttext.epochs must be an integer")
    if not isinstance(ft.get("mincount", 0), int):
        raise ValueError("embedding.fasttext.mincount must be an integer")
    if not isinstance(ft.get("lr", 0.0), (int, float)):
        raise ValueError("embedding.fasttext.lr must be a number")
    if ft.get("thread", 0) is not None and not isinstance(ft.get("thread", 0), int):
        raise ValueError("embedding.fasttext.thread must be an integer or null")

    if not isinstance(cfg["alignment"].get("pivot_count", 0), int):
        raise ValueError("alignment.pivot_count must be an integer")
    if not isinstance(cfg["alignment"].get("similarity_threshold", 0.0), (int, float)):
        raise ValueError("alignment.similarity_threshold must be a number")

    if cfg["tokenization"].get("workers", 0) is not None and not isinstance(cfg["tokenization"].get("workers", 0), int):
        raise ValueError("tokenization.workers must be an integer or null")
    if cfg["tokenization"].get("cache_dir", None) is not None and not isinstance(cfg["tokenization"].get("cache_dir", None), str):
        raise ValueError("tokenization.cache_dir must be a string or null")
    if not isinstance(cfg["tokenization"].get("min_line_length", 0), int):
        raise ValueError("tokenization.min_line_length must be an integer")

    if not isinstance(cfg["evaluation"].get("enabled", False), bool):
        raise ValueError("evaluation.enabled must be a boolean")
    if not isinstance(cfg["evaluation"].get("datasets", []), list):
        raise ValueError("evaluation.datasets must be a list of strings")
    if not isinstance(cfg["evaluation"].get("max_samples", 0), int):
        raise ValueError("evaluation.max_samples must be an integer")
    if not isinstance(cfg["evaluation"].get("qa", False), bool):
        raise ValueError("evaluation.qa must be a boolean")
    if not isinstance(cfg["evaluation"].get("baseline_model", ""), str):
        raise ValueError("evaluation.baseline_model must be a string")
    if not isinstance(cfg["pipeline"].get("run_root", ""), str):
        raise ValueError("pipeline.run_root must be a string")
    if not isinstance(cfg["pipeline"].get("max_retries", 0), int):
        raise ValueError("pipeline.max_retries must be an integer")
    if not isinstance(cfg["pipeline"].get("retry_backoff", 0.0), (int, float)):
        raise ValueError("pipeline.retry_backoff must be a number")

    # embedding_warmup
    ew = cfg.get("embedding_warmup", {})
    if not isinstance(ew.get("enabled", False), bool):
        raise ValueError("embedding_warmup.enabled must be a boolean")
    if ew.get("enabled", False):
        if not isinstance(ew.get("steps", 0), int):
            raise ValueError("embedding_warmup.steps must be an integer")
        if not isinstance(ew.get("lr", 0.0), (int, float)):
            raise ValueError("embedding_warmup.lr must be a number")
        if not isinstance(ew.get("batch_size", 0), int):
            raise ValueError("embedding_warmup.batch_size must be an integer")
        if not isinstance(ew.get("gradient_accumulation", 0), int):
            raise ValueError("embedding_warmup.gradient_accumulation must be an integer")
        if not isinstance(ew.get("max_seq_length", 0), int):
            raise ValueError("embedding_warmup.max_seq_length must be an integer")
        if not isinstance(ew.get("seed", 0), int):
            raise ValueError("embedding_warmup.seed must be an integer")
        if not isinstance(ew.get("use_flash_attn", False), bool):
            raise ValueError("embedding_warmup.use_flash_attn must be a boolean")
        if not isinstance(ew.get("bf16", True), bool):
            raise ValueError("embedding_warmup.bf16 must be a boolean")

    # vocab_adaptation
    va = cfg["vocab_adaptation"]
    # Tolerate scientific-notation strings in learning rates by coercing them to float
    va["lr_stage1"] = _coerce_to_float_if_numeric_str(va.get("lr_stage1", 0.0))
    va["lr_stage2"] = _coerce_to_float_if_numeric_str(va.get("lr_stage2", 0.0))
    if not isinstance(va.get("enabled", True), bool):
        raise ValueError("vocab_adaptation.enabled must be a boolean")
    if not isinstance(va.get("stage1_steps", 0), int):
        raise ValueError("vocab_adaptation.stage1_steps must be an integer")
    if not isinstance(va.get("stage2_steps", 0), int):
        raise ValueError("vocab_adaptation.stage2_steps must be an integer")
    if not isinstance(va.get("lr_stage1", 0.0), (int, float)):
        raise ValueError("vocab_adaptation.lr_stage1 must be a number")
    if not isinstance(va.get("lr_stage2", 0.0), (int, float)):
        raise ValueError("vocab_adaptation.lr_stage2 must be a number")
    if not isinstance(va.get("batch_size", 0), int):
        raise ValueError("vocab_adaptation.batch_size must be an integer")
    if not isinstance(va.get("gradient_accumulation", 0), int):
        raise ValueError("vocab_adaptation.gradient_accumulation must be an integer")
    if not isinstance(va.get("max_seq_length", 0), int):
        raise ValueError("vocab_adaptation.max_seq_length must be an integer")
    if not isinstance(va.get("train_start_idx_stage2", 0), int):
        raise ValueError("vocab_adaptation.train_start_idx_stage2 must be an integer")
    if not isinstance(va.get("seed", 0), int):
        raise ValueError("vocab_adaptation.seed must be an integer")
    if not isinstance(va.get("use_flash_attn", True), bool):
        raise ValueError("vocab_adaptation.use_flash_attn must be a boolean")
    if not isinstance(va.get("bf16", False), bool):
        raise ValueError("vocab_adaptation.bf16 must be a boolean")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override into base (immutable). Dicts are merged recursively; other types are replaced.
    """
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)  # type: ignore[index]
        else:
            result[k] = v
    return result


def load_config(config_path: Optional[str]) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Load a YAML config file. Returns (config_dict, resolved_path) where resolved_path is None when config_path is None.
    Applies environment variable substitution.
    """
    if not config_path:
        return get_default_config(), None
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML must be a mapping/object.")
    expanded = _expand_env_in_obj(raw)
    merged = _deep_merge(get_default_config(), expanded)
    _validate_config_structure(merged)
    return merged, path


def merge_config_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Apply CLI overrides to config. Only overrides when args differ from argparse defaults.
    Booleans are only applied when True/explicitly set (preserves YAML intent).
    """
    final_cfg = copy.deepcopy(config)

    # models
    if getattr(args, "source_tokenizer", None) not in (None, ""):
        final_cfg["models"]["source_tokenizer"] = args.source_tokenizer
    if getattr(args, "target_tokenizer", None) not in (None, ""):
        final_cfg["models"]["target_tokenizer"] = args.target_tokenizer
    if getattr(args, "source_model", None) not in (None, ""):
        final_cfg["models"]["source_model"] = args.source_model

    # pipeline root / retries
    if getattr(args, "run_root", None):
        final_cfg["pipeline"]["run_root"] = args.run_root
    if getattr(args, "max_retries", None) is not None:
        final_cfg["pipeline"]["max_retries"] = args.max_retries
    if getattr(args, "retry_backoff", None) is not None:
        final_cfg["pipeline"]["retry_backoff"] = args.retry_backoff

    # corpus
    if getattr(args, "byte_budget", None) is not None:
        final_cfg["corpus"]["byte_budget"] = max(int(args.byte_budget), 0)
    if getattr(args, "corpus_size_gb", None) is not None:
        final_cfg["corpus"]["size_gb"] = max(float(args.corpus_size_gb), 0.0)
    if getattr(args, "no_dedup", False):
        final_cfg["corpus"]["deduplicate"] = False
    if getattr(args, "hash_name", None):
        final_cfg["corpus"]["hash_name"] = args.hash_name

    # term mining
    if getattr(args, "term_top_k", None) is not None and int(args.term_top_k) > 0:
        final_cfg["term_mining"]["top_k"] = int(args.term_top_k)
    if getattr(args, "min_term_frequency", None) is not None and int(args.min_term_frequency) >= 0:
        final_cfg["term_mining"]["min_frequency"] = int(args.min_term_frequency)
    if getattr(args, "use_tfidf", False):
        final_cfg["term_mining"]["use_tfidf"] = True

    # tokenization
    if getattr(args, "tokenizer_workers", None) is not None and int(args.tokenizer_workers) > 0:
        final_cfg["tokenization"]["workers"] = int(args.tokenizer_workers)
    if getattr(args, "tokenizer_cache", None):
        final_cfg["tokenization"]["cache_dir"] = args.tokenizer_cache

    # embedding
    if getattr(args, "embedding_backend", None):
        final_cfg["embedding"]["backend"] = args.embedding_backend
    # fasttext sub-keys
    if getattr(args, "fasttext_epochs", None) is not None and int(args.fasttext_epochs) > 0:
        final_cfg["embedding"]["fasttext"]["epochs"] = int(args.fasttext_epochs)
    if getattr(args, "fasttext_mincount", None) is not None and int(args.fasttext_mincount) >= 0:
        final_cfg["embedding"]["fasttext"]["mincount"] = int(args.fasttext_mincount)
    if getattr(args, "fasttext_lr", None) is not None and float(args.fasttext_lr) > 0:
        final_cfg["embedding"]["fasttext"]["lr"] = float(args.fasttext_lr)
    if getattr(args, "fasttext_thread", None) is not None and int(args.fasttext_thread) > 0:
        final_cfg["embedding"]["fasttext"]["thread"] = int(args.fasttext_thread)

    # alignment
    if getattr(args, "pivot_count", None) is not None and int(args.pivot_count) > 0:
        final_cfg["alignment"]["pivot_count"] = int(args.pivot_count)
    if getattr(args, "similarity_threshold", None) is not None and float(args.similarity_threshold) >= 0:
        final_cfg["alignment"]["similarity_threshold"] = float(args.similarity_threshold)

    # evaluation
    # evaluate/skip-eval are special; CLI takes precedence when explicitly set
    if getattr(args, "evaluate", False):
        final_cfg["evaluation"]["enabled"] = True
    if getattr(args, "skip_eval", False):
        final_cfg["evaluation"]["enabled"] = False
    if getattr(args, "evaluation_dataset", None):
        final_cfg["evaluation"]["datasets"] = list(args.evaluation_dataset)
    if getattr(args, "max_eval_samples", None) is not None and int(args.max_eval_samples) >= 0:
        final_cfg["evaluation"]["max_samples"] = int(args.max_eval_samples)
    if getattr(args, "qa", False):
        final_cfg["evaluation"]["qa"] = True

    return final_cfg


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)


