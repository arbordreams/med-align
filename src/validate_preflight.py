from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from transformers import AutoTokenizer


def print_tokenizer_info(name: str, tok_id: str) -> None:
    tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    info = {
        "name": name,
        "identifier": tok_id,
        "vocab_size": len(tok),
        "pad_token": tok.pad_token,
        "eos_token": tok.eos_token,
        "bos_token": tok.bos_token,
        "unk_token": tok.unk_token,
        "special_tokens_map": tok.special_tokens_map,
        "chat_template_present": hasattr(tok, "chat_template") and tok.chat_template is not None,
    }
    print(json.dumps(info, indent=2))


def check_dataset(path: str, text_field_hint: Optional[str]) -> None:
    from datasets import load_from_disk

    ds = load_from_disk(path)
    splits = list(ds.keys())
    print(json.dumps({"dataset_path": path, "splits": splits}, indent=2))
    # Peek at a few rows to infer text field if not provided
    split_name = "train" if "train" in ds else splits[0]
    sample = ds[split_name][0]
    if text_field_hint and text_field_hint in sample:
        field = text_field_hint
    else:
        # pick a plausible text field
        candidates = [k for k, v in sample.items() if isinstance(v, str)]
        field = candidates[0] if candidates else None
    print(json.dumps({"text_field": field}, indent=2))
    if field:
        print(json.dumps({"sample_text_preview": sample[field][:160]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-flight validation for BioMistral → Mistral TokAlign pipeline.")
    ap.add_argument("--source-tokenizer", required=True, help="Source tokenizer id/path (BioMistral).")
    ap.add_argument("--target-tokenizer", required=True, help="Target tokenizer id/path (Mistral).")
    ap.add_argument("--dataset", help="Optional dataset path (load_from_disk) to validate.")
    ap.add_argument("--text-field", help="Optional text field hint.")
    args = ap.parse_args()

    print_tokenizer_info("source", args.source_tokenizer)
    print_tokenizer_info("target", args.target_tokenizer)
    if args.dataset:
        try:
            check_dataset(args.dataset, args.text_field)
        except Exception as exc:
            print(json.dumps({"dataset_check_error": str(exc)}))


if __name__ == "__main__":
    main()


