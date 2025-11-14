#!/usr/bin/env python3
"""
Lightweight helper to inspect how medical terms are tokenized before/after
augmentation.

Example:
    python script/tokenizer_term_diagnostics.py \
        --baseline mistralai/Mistral-7B-v0.3 \
        --candidate runs/tokenizer_adapt/.../tokenizers/target \
        --terms runs/tokenizer_adapt/.../corpus/medical_terms.txt \
        --limit 20
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from transformers import AutoTokenizer


def _load_terms(path: Path, limit: int | None) -> List[str]:
    terms: List[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            term = line.strip()
            if not term:
                continue
            terms.append(term)
            if limit is not None and len(terms) >= limit:
                break
    return terms


def _tokenize(tokenizer, term: str) -> Tuple[List[int], List[str]]:
    enc = tokenizer(term, add_special_tokens=False)
    ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return ids, tokens


def _summarize(lengths: Sequence[int]) -> str:
    if not lengths:
        return "n/a"
    single_ratio = sum(1 for l in lengths if l == 1) / len(lengths)
    return (
        f"avg={statistics.mean(lengths):.3f} "
        f"median={statistics.median(lengths):.3f} "
        f"single-token={single_ratio:.2%}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs augmented tokenizer behaviour.")
    parser.add_argument("--baseline", required=True, help="Baseline tokenizer/model identifier (e.g., mistralai/Mistral-7B-v0.3).")
    parser.add_argument("--candidate", required=True, help="Path or identifier for the augmented tokenizer.")
    parser.add_argument("--terms", required=True, help="Path to newline-delimited medical terms.")
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of terms to inspect (default: 25).")
    args = parser.parse_args(argv)

    terms = _load_terms(Path(args.terms), args.limit)
    if not terms:
        raise SystemExit("No terms loaded; ensure the terms file is non-empty.")

    baseline_tok = AutoTokenizer.from_pretrained(args.baseline, trust_remote_code=True)
    candidate_tok = AutoTokenizer.from_pretrained(args.candidate, trust_remote_code=True)

    baseline_lengths: List[int] = []
    candidate_lengths: List[int] = []

    print(f"Inspecting {len(terms)} terms\n")
    for term in terms:
        base_ids, base_tokens = _tokenize(baseline_tok, term)
        cand_ids, cand_tokens = _tokenize(candidate_tok, term)
        baseline_lengths.append(len(base_ids))
        candidate_lengths.append(len(cand_ids))

        print(f"Term: {term}")
        print(f"  baseline ({len(base_ids)} tokens): {base_tokens}")
        print(f"  candidate ({len(cand_ids)} tokens): {cand_tokens}")
        print()

    print("Summary:")
    print(f"  Baseline : {_summarize(baseline_lengths)}")
    print(f"  Candidate: {_summarize(candidate_lengths)}")


if __name__ == "__main__":
    main()

