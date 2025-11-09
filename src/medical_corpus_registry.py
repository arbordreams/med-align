"""
Registry describing curated medical datasets for TokAlign corpus building.

Each entry specifies a Hugging Face dataset ID, optional config, split,
relevant text fields, license notes, and whether an auth token (HF_TOKEN) is
required. The builder uses this metadata to download and normalize corpora
without brittle per-dataset parsing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence


TextExtractor = Callable[[Dict[str, object]], str]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a curated medical dataset."""

    slug: str
    dataset_id: str
    split: str
    text_fields: Sequence[str] = field(default_factory=lambda: ("text",))
    config_name: Optional[str] = None
    license: str = "unspecified"
    description: str = ""
    requires_auth: bool = False
    max_samples_hint: Optional[int] = None

    def extract_text(self, example: Dict[str, object]) -> str:
        """Compose a text span from the configured fields."""

        parts: List[str] = []
        for field in self.text_fields:
            value = example
            for key in field.split("."):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            if isinstance(value, str):
                parts.append(value.strip())
        return "\n\n".join(part for part in parts if part)


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "pubmed_abstract": DatasetSpec(
        slug="pubmed_abstract",
        dataset_id="uiyunkim-hub/pubmed-abstract",
        config_name=None,
        split="train",
        text_fields=("abstract",),
        license="Public Domain",
        description="PubMed abstracts from uiyunkim-hub collection.",
        max_samples_hint=None,
    ),
}


def list_datasets() -> List[str]:
    """Return sorted slugs for discoverability."""

    return sorted(DATASET_REGISTRY.keys())


def get_dataset(slug: str) -> DatasetSpec:
    """Fetch the dataset specification or raise a KeyError."""

    return DATASET_REGISTRY[slug]


def resolve_datasets(
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[DatasetSpec]:
    """
    Compute the final dataset list given include/exclude filters.

    When `include` is None all datasets are used (minus any excluded ones).
    """

    include_set = set(include) if include else set(DATASET_REGISTRY.keys())
    exclude_set = set(exclude) if exclude else set()
    final = [slug for slug in include_set if slug in DATASET_REGISTRY and slug not in exclude_set]
    return [DATASET_REGISTRY[slug] for slug in sorted(final)]

