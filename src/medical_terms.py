"""
Mining high-frequency medical terms and augmenting tokenizers for TokAlign.

This module provides two entry points:
* `mine_terms` scans a JSONL corpus and selects salient tokens via simple
  frequency thresholds or TF-IDF weighting.
* `augment_tokenizer` injects those tokens into an existing tokenizer while
  avoiding collisions, returning the list of newly added tokens and saving the
  augmented tokenizer to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
try:  # optional stopwords without hard dependency
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _STOPWORDS  # type: ignore
except Exception:  # pragma: no cover
    _STOPWORDS = set()  # minimal fallback
from transformers import AutoTokenizer, AddedToken

logger = logging.getLogger(__name__)


def _read_corpus(path: str, max_docs: int | None = None) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if max_docs is not None and idx >= max_docs:
                break
            record = json.loads(line)
            text = record.get("text")
            if not text:
                continue
            texts.append(text)
    return texts


def _tokenize(text: str) -> List[str]:
    # Basic whitespace split paired with medical-friendly lowercasing and conservative filtering.
    toks = [tok for tok in text.strip().lower().split() if tok]
    filtered: List[str] = []
    for tok in toks:
        if len(tok) < 3:
            continue
        if tok.isdigit():
            continue
        if tok in _STOPWORDS:
            continue
        filtered.append(tok)
    return filtered


def _is_medical_term_quality(
    term: str,
    min_length: int = 3,
    max_length: int = 50,
    require_alpha: bool = True,
    allow_hyphens: bool = True,
    medical_patterns: Optional[Sequence[re.Pattern[str]]] = None,
) -> bool:
    """
    Heuristic quality checks for candidate medical terms.
    """
    stripped = term.strip()
    if not stripped:
        return False
    length = len(stripped)
    if length < min_length or length > max_length:
        return False
    if require_alpha and not any(ch.isalpha() for ch in stripped):
        return False
    if not allow_hyphens and "-" in stripped:
        return False
    if medical_patterns:
        if not any(p.search(stripped) for p in medical_patterns):
            return False
    return True


def _compute_term_quality_score(
    term: str,
    frequency: int,
    tfidf_score: Optional[float],
    corpus_stats: Dict[str, Any],
) -> float:
    """
    Compute a composite quality score in [0, 1] combining frequency,
    TF-IDF, and token characteristics.
    """
    freq = max(int(frequency), 0)
    max_freq = max(int(corpus_stats.get("max_frequency", 0)), 0)
    if max_freq > 0 and freq > 0:
        freq_component = math.log(freq + 1.0) / math.log(max_freq + 1.0)
    else:
        freq_component = 0.0

    max_tfidf = float(corpus_stats.get("max_tfidf", 0.0) or 0.0)
    tfidf_val = float(tfidf_score or 0.0)
    if max_tfidf > 0.0 and tfidf_val > 0.0:
        tfidf_component = min(max(tfidf_val / max_tfidf, 0.0), 1.0)
    else:
        tfidf_component = 0.0

    term_str = term.strip()
    term_len = len(term_str)
    min_length = int(corpus_stats.get("min_length", 3))
    max_length = int(corpus_stats.get("max_length", 50))
    if term_len <= 0 or term_len < min_length or term_len > max_length:
        char_component = 0.0
    else:
        ideal_length = float(corpus_stats.get("ideal_length", (min_length + max_length) / 2.0))
        max_dev = max(ideal_length - min_length, max_length - ideal_length)
        if max_dev <= 0:
            length_component = 1.0
        else:
            length_component = 1.0 - min(abs(term_len - ideal_length) / max_dev, 1.0)

        alpha_count = sum(1 for ch in term_str if ch.isalpha())
        alpha_ratio = alpha_count / max(term_len, 1)
        char_component = 0.7 * length_component + 0.3 * alpha_ratio
        char_component = max(0.0, min(char_component, 1.0))

    score = 0.4 * freq_component + 0.4 * tfidf_component + 0.2 * char_component
    return max(0.0, min(score, 1.0))


def _compute_adaptive_thresholds(
    counter: Counter[str],
    corpus_size: int,
    target_terms: int = 500,
) -> Dict[str, Any]:
    """
    Compute adaptive frequency thresholds based on corpus statistics.
    """
    thresholds: Dict[str, Any] = {
        "enabled": False,
        "percentile": None,
        "statistical": None,
        "size_adaptive": None,
        "chosen": None,
        "target_terms": int(target_terms),
        "corpus_size": int(corpus_size),
    }
    if not counter:
        return thresholds

    values = np.array(list(counter.values()), dtype=float)
    if values.size == 0:
        return thresholds

    percentile_value = int(np.percentile(values, 90))
    if percentile_value < 1:
        percentile_value = 1

    mean = float(values.mean())
    std = float(values.std())
    stat_value = int(round(mean + std))
    if stat_value < 1:
        stat_value = 1

    if corpus_size <= 0:
        size_value = 1
    else:
        size_value = max(1, int(round(math.log10(corpus_size))))

    def _count_at(thresh: int) -> int:
        return int((values >= thresh).sum())

    candidates: Dict[str, Dict[str, Any]] = {
        "percentile": {"value": int(percentile_value), "estimated_terms": _count_at(percentile_value)},
        "statistical": {"value": int(stat_value), "estimated_terms": _count_at(stat_value)},
        "size_adaptive": {"value": int(size_value), "estimated_terms": _count_at(size_value)},
    }
    chosen_name = min(
        candidates.keys(),
        key=lambda name: abs(candidates[name]["estimated_terms"] - max(target_terms, 1)),
    )
    chosen = {"method": chosen_name, **candidates[chosen_name]}

    thresholds.update(candidates)
    thresholds["chosen"] = chosen
    thresholds["enabled"] = True
    return thresholds


def mine_terms(
    corpus_path: str,
    *,
    top_k: int = 500,
    min_count: int = 5,
    use_tfidf: bool = False,
    max_docs: int | None = None,
    use_adaptive_thresholds: bool = True,
    quality_filter: bool = True,
    min_quality_score: float = 0.3,
    medical_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract salient medical terms from the monolingual corpus.

    Parameters
    ----------
    corpus_path:
        JSONL file produced by `medical_corpus.build_medical_corpus`.
    top_k:
        Maximum number of terms to return.
    min_count:
        Minimum frequency for a token to be considered.
    use_tfidf:
        If `True`, rank by TF-IDF score; otherwise simply count token frequency.
    max_docs:
        Optional cap on the number of documents to inspect.
    use_adaptive_thresholds:
        If `True`, compute an adaptive frequency threshold close to target term count.
    quality_filter:
        If `True`, apply heuristic filtering and quality scoring.
    min_quality_score:
        Minimum composite quality score in [0, 1] for a term to be kept.
    medical_patterns:
        Optional regex patterns that candidate terms must match at least one of.

    Returns
    -------
    terms, metadata
    """

    compiled_patterns: Optional[List[re.Pattern[str]]] = None
    if medical_patterns:
        compiled_patterns = []
        for pattern in medical_patterns:
            if isinstance(pattern, str):
                try:
                    compiled_patterns.append(re.compile(pattern))
                except re.error:
                    logger.warning("Invalid medical pattern %r; skipping.", pattern)
            else:
                compiled_patterns.append(pattern)
        if not compiled_patterns:
            compiled_patterns = None

    metadata: Dict[str, Any] = {
        "total_docs": 0,
        "total_terms": 0,
        "filtered_by_quality": 0,
        "filtered_by_threshold": 0,
        "adaptive_thresholds": {"enabled": False},
        "final_term_count": 0,
        "quality_score_stats": {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        },
        "max_tfidf": 0.0,
    }

    def _summarize_scores(scores: Sequence[float]) -> Dict[str, float]:
        if not scores:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        arr = np.array(scores, dtype=float)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    # TF-IDF requires the full text list; frequency mode can stream to avoid OOM on large corpora.
    if use_tfidf:
        texts = _read_corpus(corpus_path, max_docs=max_docs)
        metadata["total_docs"] = len(texts)
        if not texts:
            logger.warning("No texts were found in %s; returning an empty term list.", corpus_path)
            return [], metadata

        vectorizer = TfidfVectorizer(tokenizer=_tokenize, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(texts)
        scores = tfidf_matrix.max(axis=0).toarray().flatten()
        vocab = np.array(vectorizer.get_feature_names_out())
        max_tfidf = float(scores.max()) if scores.size else 0.0
        metadata["max_tfidf"] = max_tfidf

        # Build frequency counter for adaptive thresholds and quality scoring.
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(_tokenize(text))
        metadata["total_terms"] = len(counter)
        corpus_tokens = int(sum(counter.values()))
        max_freq = max(counter.values()) if counter else 0

        adaptive_thresholds: Dict[str, Any] = {"enabled": False}
        if use_adaptive_thresholds and counter:
            adaptive_thresholds = _compute_adaptive_thresholds(
                counter=counter,
                corpus_size=corpus_tokens,
                target_terms=top_k,
            )
        metadata["adaptive_thresholds"] = adaptive_thresholds
        chosen_value = 0
        chosen_info = adaptive_thresholds.get("chosen") or {}
        if adaptive_thresholds.get("enabled") and isinstance(chosen_info.get("value"), int):
            chosen_value = int(chosen_info["value"])

        effective_min_count = max(int(min_count), 1, chosen_value)
        if use_adaptive_thresholds and max_freq > 0:
            effective_min_count = min(effective_min_count, max_freq)
        corpus_stats = {
            "max_frequency": max_freq,
            "max_tfidf": max_tfidf,
            "min_length": 3,
            "max_length": 50,
            "ideal_length": 12,
        }

        score_by_term = {term: float(score) for term, score in zip(vocab, scores)}

        threshold_rejected = 0
        quality_rejected = 0
        candidates: List[Tuple[str, float, int, float]] = []

        for term in vocab:
            freq = int(counter.get(term, 0))
            if freq < effective_min_count:
                threshold_rejected += 1
                continue
            tfidf_val = score_by_term.get(term, 0.0)

            passes_quality = _is_medical_term_quality(
                term,
                medical_patterns=compiled_patterns,
            )
            score = _compute_term_quality_score(
                term=term,
                frequency=freq,
                tfidf_score=tfidf_val,
                corpus_stats=corpus_stats,
            )

            if quality_filter and not passes_quality:
                quality_rejected += 1
                continue
            if score < float(min_quality_score):
                quality_rejected += 1
                continue

            candidates.append((term, score, freq, tfidf_val))

        metadata["filtered_by_threshold"] = threshold_rejected
        metadata["filtered_by_quality"] = quality_rejected

        if not candidates:
            logger.warning("All candidate terms were filtered; returning an empty term list.")
            return [], metadata

        candidates.sort(key=lambda item: (-item[1], -item[2], item[0]))
        selected = candidates[:top_k]
        terms = [term for term, _, _, _ in selected]
        scores_final = [score for _, score, _, _ in selected]
        metadata["final_term_count"] = len(terms)
        metadata["quality_score_stats"] = _summarize_scores(scores_final)
        return terms, metadata

    counter: Counter[str] = Counter()
    total_docs = 0
    with open(corpus_path, "r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if max_docs is not None and idx >= max_docs:
                break
            try:
                record = json.loads(line)
            except Exception:
                continue
            text = record.get("text")
            if not text:
                continue
            total_docs += 1
            counter.update(_tokenize(text))

    metadata["total_docs"] = total_docs
    metadata["total_terms"] = len(counter)

    if not counter:
        logger.warning("No tokens were mined from %s; returning an empty term list.", corpus_path)
        return [], metadata

    corpus_tokens = int(sum(counter.values()))
    max_freq = max(counter.values()) if counter else 0
    adaptive_thresholds = {"enabled": False}
    if use_adaptive_thresholds and counter:
        adaptive_thresholds = _compute_adaptive_thresholds(
            counter=counter,
            corpus_size=corpus_tokens,
            target_terms=top_k,
        )
    metadata["adaptive_thresholds"] = adaptive_thresholds
    chosen_value = 0
    chosen_info = adaptive_thresholds.get("chosen") or {}
    if adaptive_thresholds.get("enabled") and isinstance(chosen_info.get("value"), int):
        chosen_value = int(chosen_info["value"])

    effective_min_count = max(int(min_count), 1, chosen_value)
    if use_adaptive_thresholds and max_freq > 0:
        effective_min_count = min(effective_min_count, max_freq)

    threshold_filtered: List[Tuple[str, int]] = []
    threshold_rejected = 0
    for term, count in counter.items():
        if count >= effective_min_count:
            threshold_filtered.append((term, count))
        else:
            threshold_rejected += 1
    metadata["filtered_by_threshold"] = threshold_rejected

    corpus_stats = {
        "max_frequency": max_freq,
        "max_tfidf": 0.0,
        "min_length": 3,
        "max_length": 50,
        "ideal_length": 12,
    }

    quality_rejected = 0
    candidates_freq: List[Tuple[str, float, int]] = []

    for term, count in threshold_filtered:
        passes_quality = _is_medical_term_quality(
            term,
            medical_patterns=compiled_patterns,
        )
        score = _compute_term_quality_score(
            term=term,
            frequency=count,
            tfidf_score=None,
            corpus_stats=corpus_stats,
        )
        if quality_filter and not passes_quality:
            quality_rejected += 1
            continue
        if score < float(min_quality_score):
            quality_rejected += 1
            continue
        candidates_freq.append((term, score, count))

    metadata["filtered_by_quality"] = quality_rejected

    if not candidates_freq:
        logger.warning("All candidate terms were filtered; returning an empty term list.")
        return [], metadata

    candidates_freq.sort(key=lambda item: (-item[1], -item[2], item[0]))
    selected_freq = candidates_freq[:top_k]
    terms = [term for term, _, _ in selected_freq]
    scores_final = [score for _, score, _ in selected_freq]
    metadata["final_term_count"] = len(terms)
    metadata["quality_score_stats"] = _summarize_scores(scores_final)
    return terms, metadata


def augment_tokenizer(
    tokenizer_path: str,
    terms: Sequence[str],
    output_dir: str,
) -> Tuple[List[str], List[str]]:
    """
    Insert medical terms into a tokenizer and save the augmented version.

    Returns
    -------
    added_tokens, skipped_tokens
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    existing_vocab = set(tokenizer.get_vocab().keys())
    special_tokens = set(tokenizer.all_special_tokens or [])

    sanitized_terms: List[str] = []
    skipped_terms: List[str] = []
    for term in terms:
        stripped = term.strip()
        if not stripped:
            continue
        if stripped in special_tokens:
            skipped_terms.append(stripped)
            continue
        if stripped in existing_vocab or stripped in sanitized_terms:
            skipped_terms.append(stripped)
            continue
        sanitized_terms.append(stripped)

    # SentencePiece / LLaMA / Mistral tokenizers rely on an implicit whitespace prefix.
    # Instead of injecting a literal ▁ (U+2581) which never matches raw text, rely on
    # AddedToken with lstrip=True so occurrences of " term" map to the new entry.
    sp_like = any(k in tokenizer.__class__.__name__.lower() for k in ("llama", "mistral"))
    added_entries: List[str | AddedToken] = []

    if sp_like:
        for term in sanitized_terms:
            added_entries.append(
                AddedToken(term, lstrip=True, rstrip=False, normalized=False)
            )
    else:
        added_entries = sanitized_terms.copy()

    if added_entries:
        original_size = len(tokenizer)
        tokenizer.add_tokens(added_entries)
        new_size = len(tokenizer)
        logger.info(
            "Tokenizer vocabulary grew from %d to %d (+%d).",
            original_size,
            new_size,
            new_size - original_size,
        )
    else:
        logger.info("No new terms to add; tokenizer vocabulary remains unchanged.")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    logger.info(
        "Tokenizer augmented with %s new terms (skipped %s duplicates).",
        len(sanitized_terms),
        len(skipped_terms),
    )
    return sanitized_terms, skipped_terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract medical terms and augment tokenizers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine_parser = subparsers.add_parser("mine", help="Mine high-frequency medical terms.")
    mine_parser.add_argument("--corpus", required=True, help="Path to medical JSONL corpus.")
    mine_parser.add_argument("--output", required=True, help="Destination file (.txt) for selected terms.")
    mine_parser.add_argument("--top-k", type=int, default=500, help="Number of terms to keep.")
    mine_parser.add_argument("--min-count", type=int, default=5, help="Minimum frequency threshold.")
    mine_parser.add_argument(
        "--use-tfidf",
        action="store_true",
        help="Rank tokens by TF-IDF instead of raw frequency.",
    )
    mine_parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional cap on documents inspected while mining (0 = no limit).",
    )
    mine_parser.add_argument(
        "--use-adaptive-thresholds",
        action="store_true",
        default=None,
        help="Enable adaptive thresholds when selecting terms (default: enabled).",
    )
    mine_parser.add_argument(
        "--quality-filter",
        action="store_true",
        default=None,
        help="Enable heuristic quality filtering of candidate terms (default: enabled).",
    )
    mine_parser.add_argument(
        "--min-quality-score",
        type=float,
        default=None,
        help="Minimum composite quality score in [0,1] for inclusion (default: 0.3).",
    )
    mine_parser.add_argument(
        "--medical-patterns",
        action="append",
        default=None,
        help="Optional regex pattern(s) that candidate terms must match; can be passed multiple times.",
    )

    augment_parser = subparsers.add_parser("augment", help="Augment a tokenizer with mined terms.")
    augment_parser.add_argument("--tokenizer", required=True, help="Tokenizer path or identifier.")
    augment_parser.add_argument("--terms", required=True, help="Path to newline-delimited term list.")
    augment_parser.add_argument("--output", required=True, help="Directory where the augmented tokenizer is saved.")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _run_mine(args: argparse.Namespace) -> None:
    use_adaptive = args.use_adaptive_thresholds
    if use_adaptive is None:
        use_adaptive = True
    quality_filter = args.quality_filter
    if quality_filter is None:
        quality_filter = True
    min_quality = args.min_quality_score if args.min_quality_score is not None else 0.3

    terms, metadata = mine_terms(
        corpus_path=args.corpus,
        top_k=args.top_k,
        min_count=args.min_count,
        use_tfidf=args.use_tfidf,
        max_docs=None if args.max_docs <= 0 else args.max_docs,
        use_adaptive_thresholds=use_adaptive,
        quality_filter=quality_filter,
        min_quality_score=min_quality,
        medical_patterns=args.medical_patterns,
    )
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        for term in terms:
            fp.write(term + "\n")
    metadata_path = str(args.output) + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    logger.info("Saved %s medical terms to %s (metadata: %s).", len(terms), args.output, metadata_path)


def _run_augment(args: argparse.Namespace) -> None:
    with open(args.terms, "r", encoding="utf-8") as fp:
        terms = [line.strip() for line in fp if line.strip()]
    added, skipped = augment_tokenizer(
        tokenizer_path=args.tokenizer,
        terms=terms,
        output_dir=args.output,
    )
    summary = {
        "added": added,
        "skipped": skipped,
    }
    report_path = os.path.join(args.output, "medical_term_augmented.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    logger.info("Tokenizer augmentation report written to %s.", report_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    if args.command == "mine":
        _run_mine(args)
    elif args.command == "augment":
        _run_augment(args)
    else:
        raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
