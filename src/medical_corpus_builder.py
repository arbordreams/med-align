"""
CLI tool that downloads curated medical datasets and normalizes them into
TokAlign-friendly JSONL shards. It relies on the dataset registry to avoid
schema surprises, respects HF_TOKEN for gated sources, and can optionally
aggregate the shards using the existing medical corpus ingest logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset  # type: ignore

from . import medical_corpus
from .medical_corpus_registry import DatasetSpec, resolve_datasets, list_datasets

logger = logging.getLogger(__name__)


def _compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def _write_jsonl(
    spec: DatasetSpec,
    dataset,
    output_path: Path,
    *,
    max_samples: Optional[int] = None,
) -> Tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total_bytes = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for example in dataset:
            text = spec.extract_text(example if isinstance(example, dict) else dict(example))
            if not text:
                continue
            record = {"text": text}
            line = json.dumps(record, ensure_ascii=False)
            fp.write(line + "\n")
            total_bytes += len(line.encode("utf-8")) + 1
            count += 1
            if max_samples and count >= max_samples:
                break
    return count, total_bytes


def build_dataset_shard(
    spec: DatasetSpec,
    *,
    output_dir: Path,
    hf_token: Optional[str],
    max_samples: Optional[int],
) -> Dict[str, object]:
    """
    Download a dataset and normalize it into a JSONL shard.
    Returns a manifest entry describing the outcome.
    """

    manifest_entry: Dict[str, object] = {
        "slug": spec.slug,
        "dataset_id": spec.dataset_id,
        "split": spec.split,
        "requires_auth": spec.requires_auth,
        "license": spec.license,
        "description": spec.description,
        "status": "pending",
    }

    if spec.requires_auth and not hf_token:
        manifest_entry["status"] = "skipped"
        manifest_entry["reason"] = "HF_TOKEN is required but missing."
        logger.warning("Skipping %s because HF_TOKEN is not set.", spec.slug)
        return manifest_entry

    load_kwargs = {
        "split": spec.split,
        "streaming": True,
        "token": hf_token,
    }
    try:
        if spec.config_name:
            dataset = load_dataset(spec.dataset_id, spec.config_name, **load_kwargs)
        else:
            dataset = load_dataset(spec.dataset_id, **load_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        manifest_entry["status"] = "error"
        manifest_entry["error"] = str(exc)
        logger.exception("Failed to load dataset %s: %s", spec.slug, exc)
        return manifest_entry

    shard_path = output_dir / "shards" / f"{spec.slug}.jsonl"

    try:
        max_samples = max_samples or spec.max_samples_hint
        records, bytes_written = _write_jsonl(
            spec,
            dataset,
            shard_path,
            max_samples=max_samples,
        )
    except Exception as exc:  # pragma: no cover - defensive
        manifest_entry["status"] = "error"
        manifest_entry["error"] = str(exc)
        logger.exception("Failed to write shard for %s: %s", spec.slug, exc)
        return manifest_entry

    manifest_entry.update(
        status="success",
        output_path=str(shard_path),
        records=records,
        bytes=bytes_written,
        sha256=_compute_sha256(shard_path) if shard_path.exists() else None,
    )
    return manifest_entry


def aggregate_shards(
    shard_entries: Sequence[Dict[str, object]],
    *,
    output_dir: Path,
    byte_budget: Optional[int],
    deduplicate: bool,
    hash_name: str,
) -> Dict[str, object]:
    shard_paths = [
        entry["output_path"]
        for entry in shard_entries
        if entry.get("status") == "success" and entry.get("output_path")
    ]
    if not shard_paths:
        return {"status": "skipped", "reason": "no successful dataset shards"}

    aggregate_dir = output_dir / "aggregated"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregated_path = aggregate_dir / "medical_corpus.jsonl"
    mirror_path = aggregate_dir / "medical_pairs.jsonl"

    stats = medical_corpus.build_medical_corpus(
        inputs=shard_paths,
        output_path=str(aggregated_path),
        mirror_output_path=str(mirror_path),
        byte_budget=byte_budget,
        deduplicate=deduplicate,
        hash_name=hash_name,
    )

    return {
        "status": "success",
        "aggregated_path": str(aggregated_path),
        "mirror_path": str(mirror_path),
        "sha256": _compute_sha256(aggregated_path),
        "stats": stats.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated medical corpus for TokAlign.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for shards and manifest.")
    parser.add_argument(
        "--include",
        nargs="*",
        help="Dataset slugs to include (default: all).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Dataset slugs to exclude.",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token. Defaults to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional sample cap per dataset (0 = full dataset).",
    )
    parser.add_argument(
        "--byte-budget",
        type=int,
        default=0,
        help="Optional byte budget applied during aggregation (0 = unlimited).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication during aggregation.",
    )
    parser.add_argument(
        "--hash-name",
        default="sha256",
        help="Hash algorithm for deduplication (default: sha256).",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregated corpus creation (only write per-dataset shards).",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List curated dataset slugs and exit.",
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

    if args.list_datasets:
        print("\n".join(list_datasets()))
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    max_samples = args.max_samples if args.max_samples > 0 else None
    byte_budget = args.byte_budget if args.byte_budget > 0 else None
    deduplicate = not args.no_dedup

    specs: List[DatasetSpec] = resolve_datasets(args.include, args.exclude)
    if not specs:
        raise SystemExit("No datasets selected. Use --include with valid slugs.")

    shard_entries: List[Dict[str, object]] = []
    for spec in specs:
        logger.info("Processing dataset %s (%s)", spec.slug, spec.dataset_id)
        entry = build_dataset_shard(
            spec,
            output_dir=output_dir,
            hf_token=hf_token,
            max_samples=max_samples,
        )
        shard_entries.append(entry)

    aggregate_entry: Dict[str, object] = {}
    if not args.skip_aggregate:
        aggregate_entry = aggregate_shards(
            shard_entries,
            output_dir=output_dir,
            byte_budget=byte_budget,
            deduplicate=deduplicate,
            hash_name=args.hash_name,
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "datasets": shard_entries,
        "aggregate": aggregate_entry,
        "byte_budget": byte_budget,
        "deduplicate": deduplicate,
        "hash_name": args.hash_name,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

