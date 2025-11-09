"""
Utilities for preparing a monolingual medical corpus for TokAlign.

This module flattens a directory of JSONL shards, deduplicates examples via
hashing, enforces optional byte budgets, and optionally mirrors each span into
`(source_text, target_text)` pairs so downstream components can re-use the same
content for both tokenizers.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CorpusStats:
    """Light-weight record of how much text was ingested."""

    total_examples: int = 0
    unique_examples: int = 0
    total_bytes: int = 0
    truncated: bool = False

    def to_dict(self) -> Dict[str, int | bool]:
        return {
            "total_examples": self.total_examples,
            "unique_examples": self.unique_examples,
            "total_bytes": self.total_bytes,
            "truncated": self.truncated,
        }


def _iter_jsonl_files(paths: Iterable[str]) -> Iterator[Path]:
    """
    Yield jsonl files by flattening any directories.

    Parameters
    ----------
    paths:
        Collection of files or directories that may contain JSONL shards.
    """

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".jsonl":
            yield path
        elif path.is_dir():
            for candidate in sorted(path.rglob("*.jsonl")):
                if candidate.is_file():
                    yield candidate
        else:
            logger.warning("Skipping path %s because it is not a JSONL file or directory.", path)


def _hash_text(text: str, hash_name: str = "sha256") -> str:
    h = hashlib.new(hash_name)
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def build_medical_corpus(
    inputs: Iterable[str],
    output_path: str,
    *,
    mirror_output_path: Optional[str] = None,
    byte_budget: Optional[int] = None,
    deduplicate: bool = True,
    hash_name: str = "sha256",
) -> CorpusStats:
    """
    Aggregate JSONL shards containing {"text": ...} records into a single JSONL.

    Parameters
    ----------
    inputs:
        An iterable of JSONL files or directories to expand recursively.
    output_path:
        Destination JSONL file where deduplicated samples will be written.
    mirror_output_path:
        Optional JSONL file where mirrored `(source_text, target_text)` objects
        should be saved.
    byte_budget:
        Optional maximum number of bytes (including newlines) that may be
        written to `output_path`. Once the budget is exceeded ingestion stops.
    deduplicate:
        Whether to drop duplicate samples based on `hash_name`.
    hash_name:
        Name of the hashlib algorithm to use when deduplicating.
    """

    os.makedirs(Path(output_path).parent, exist_ok=True)
    if mirror_output_path:
        os.makedirs(Path(mirror_output_path).parent, exist_ok=True)

    seen: Set[str] = set()
    stats = CorpusStats()
    budget = byte_budget if byte_budget and byte_budget > 0 else None

    with open(output_path, "w", encoding="utf-8") as out_fp:
        mirror_fp = open(mirror_output_path, "w", encoding="utf-8") if mirror_output_path else None
        try:
            for shard in _iter_jsonl_files(inputs):
                logger.info("Reading shard %s", shard)
                with shard.open("r", encoding="utf-8") as shard_fp:
                    for line in shard_fp:
                        stats.total_examples += 1

                        line = line.strip()
                        if not line:
                            continue

                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Skipping malformed line in %s", shard)
                            continue

                        if "text" not in payload:
                            logger.warning("Skipping line without 'text' field in %s", shard)
                            continue

                        text = payload["text"]
                        key = _hash_text(text, hash_name) if deduplicate else None
                        if key is not None:
                            if key in seen:
                                continue
                            seen.add(key)

                        serialized = json.dumps({"text": text}, ensure_ascii=False)
                        projected_bytes = stats.total_bytes + len(serialized.encode("utf-8")) + 1
                        if budget is not None and projected_bytes > budget:
                            stats.truncated = True
                            logger.info(
                                "Byte budget reached (%s bytes); stopping ingestion with %s unique examples.",
                                budget,
                                stats.unique_examples,
                            )
                            return stats

                        out_fp.write(serialized + "\n")
                        stats.unique_examples += 1
                        stats.total_bytes += len(serialized.encode("utf-8")) + 1

                        if mirror_fp:
                            mirror_payload = {"source_text": text, "target_text": text}
                            mirror_fp.write(json.dumps(mirror_payload, ensure_ascii=False) + "\n")
        finally:
            if mirror_fp:
                mirror_fp.close()

    logger.info(
        "Finished corpus aggregation: %s total / %s unique examples; %s bytes written.",
        stats.total_examples,
        stats.unique_examples,
        stats.total_bytes,
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a medical JSONL corpus for TokAlign.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input JSONL file or directory. Use multiple --input flags for shards.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file containing deduplicated medical texts (one per line).",
    )
    parser.add_argument(
        "--mirror-output",
        help="Optional JSONL where each line mirrors the text into (source_text,target_text).",
    )
    parser.add_argument(
        "--byte-budget",
        type=int,
        default=0,
        help="Maximum number of bytes to write before halting ingestion. Zero disables the limit.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable hash-based deduplication. Enabled by default.",
    )
    parser.add_argument(
        "--hash-name",
        default="sha256",
        help="Hash algorithm name to use for deduplication (default: sha256).",
    )
    parser.add_argument(
        "--stats-path",
        help="Optional JSON file where ingestion statistics should be recorded.",
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

    stats = build_medical_corpus(
        inputs=args.input,
        output_path=args.output,
        mirror_output_path=args.mirror_output,
        byte_budget=None if args.byte_budget <= 0 else args.byte_budget,
        deduplicate=not args.no_dedup,
        hash_name=args.hash_name,
    )

    if args.stats_path:
        os.makedirs(Path(args.stats_path).parent, exist_ok=True)
        with open(args.stats_path, "w", encoding="utf-8") as fp:
            json.dump(stats.to_dict(), fp, indent=2)


if __name__ == "__main__":
    main()

