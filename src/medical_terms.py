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
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
try:  # optional stopwords without hard dependency
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _STOPWORDS  # type: ignore
except Exception:  # pragma: no cover
    _STOPWORDS = set()  # minimal fallback
from transformers import AutoTokenizer

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


def mine_terms(
    corpus_path: str,
    *,
    top_k: int = 500,
    min_count: int = 5,
    use_tfidf: bool = False,
    max_docs: int | None = None,
) -> List[str]:
    """
    Extract salient medical terms from the monolingual corpus.

    Parameters
    ----------
    corpus_path:
        JSONL file produced by `medical_corpus.build_medical_corpus`.
    top_k:
        Maximum number of terms to return.
    min_count:
        Minimum frequency for a token to be considered (ignored for TF-IDF if < 1).
    use_tfidf:
        If `True`, rank by TF-IDF score; otherwise simply count token frequency.
    max_docs:
        Optional cap on the number of documents to inspect.
    """

    # TF-IDF requires the full text list; frequency mode can stream to avoid OOM on large corpora.
    if use_tfidf:
        texts = _read_corpus(corpus_path, max_docs=max_docs)
        if not texts:
            logger.warning("No texts were found in %s; returning an empty term list.", corpus_path)
            return []
        vectorizer = TfidfVectorizer(tokenizer=_tokenize, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(texts)
        scores = tfidf_matrix.max(axis=0).toarray().flatten()
        vocab = np.array(vectorizer.get_feature_names_out())
        top_indices = np.argsort(scores)[::-1]
        terms: List[str] = []
        for idx in top_indices:
            term = vocab[idx]
            if term.strip() and term not in terms:
                terms.append(term)
            if len(terms) >= top_k:
                break
        return terms

    counter: Counter[str] = Counter()
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
            counter.update(_tokenize(text))

    filtered = [(term, count) for term, count in counter.items() if count >= min_count]
    filtered.sort(key=lambda item: item[1], reverse=True)
    return [term for term, _ in filtered[:top_k]]


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

    sanitized_terms: List[str] = []
    skipped_terms: List[str] = []
    for term in terms:
        stripped = term.strip()
        if not stripped:
            continue
        if stripped in existing_vocab or stripped in sanitized_terms:
            skipped_terms.append(stripped)
            continue
        sanitized_terms.append(stripped)

    # SentencePiece-friendly: prefix tokens with ▁ to increase match rate on spm tokenizers.
    sp_like = any(k in tokenizer.__class__.__name__.lower() for k in ("llama", "mistral"))
    sp_terms: List[str] = []
    for term in sanitized_terms:
        if sp_like and not term.startswith("▁"):
            sp_terms.append("▁" + term)
        else:
            sp_terms.append(term)
    if sp_terms:
        tokenizer.add_tokens(sp_terms)
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
    terms = mine_terms(
        corpus_path=args.corpus,
        top_k=args.top_k,
        min_count=args.min_count,
        use_tfidf=args.use_tfidf,
        max_docs=None if args.max_docs <= 0 else args.max_docs,
    )
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        for term in terms:
            fp.write(term + "\n")
    logger.info("Saved %s medical terms to %s.", len(terms), args.output)


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

