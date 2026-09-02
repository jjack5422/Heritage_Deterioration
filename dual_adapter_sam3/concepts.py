"""Versioned fixed-prompt contract for the two deterioration channels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


CHANNEL_ORDER = ("crack_craquelure", "loss")
EXPECTED_PROMPTS = (
    "craquelure, a network of fine cracks in the painted surface",
    "paint loss, a missing area in the paint layer that exposes the substrate",
)
EXPECTED_RAW_LABELS = ((1,), (2,))


class ConceptContractError(ValueError):
    """Raised when the fixed concept registry is incompatible with the experiment."""


@dataclass(frozen=True)
class Concept:
    name: str
    canonical_prompt: str
    aliases: tuple[str, ...]
    raw_labels: tuple[int, ...]


@dataclass(frozen=True)
class ConceptRegistry:
    schema_version: int
    prompt_set_id: str
    concepts: tuple[Concept, ...]
    sha256: str

    @property
    def channel_order(self) -> tuple[str, ...]:
        return tuple(concept.name for concept in self.concepts)

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(concept.canonical_prompt for concept in self.concepts)


def _canonical_payload(schema_version: int, prompt_set_id: str, concepts: tuple[Concept, ...]) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "prompt_set_id": prompt_set_id,
        "channel_order": [concept.name for concept in concepts],
        "concepts": [
            {
                "name": concept.name,
                "canonical_prompt": concept.canonical_prompt,
                "aliases": list(concept.aliases),
                "raw_labels": list(concept.raw_labels),
            }
            for concept in concepts
        ],
    }


def load_concept_registry(path: str | Path) -> ConceptRegistry:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConceptContractError(f"cannot read concept registry {path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ConceptContractError("concept registry schema_version must be 1")
    if raw.get("prompt_set_id") != "monument_deterioration_v1":
        raise ConceptContractError("unexpected prompt_set_id")
    mapping = raw.get("concepts")
    if not isinstance(mapping, dict) or tuple(mapping) != CHANNEL_ORDER:
        raise ConceptContractError(f"concept channel order must be {CHANNEL_ORDER!r}")
    concepts: list[Concept] = []
    for index, name in enumerate(CHANNEL_ORDER):
        item = mapping[name]
        if not isinstance(item, dict):
            raise ConceptContractError(f"concept {name} must be an object")
        prompt = item.get("canonical_prompt")
        aliases = item.get("aliases")
        labels = item.get("raw_labels")
        if prompt != EXPECTED_PROMPTS[index]:
            raise ConceptContractError(f"canonical prompt changed for {name}")
        if not isinstance(aliases, list) or not aliases or not all(isinstance(v, str) and v.strip() for v in aliases):
            raise ConceptContractError(f"concept {name} aliases must be non-empty strings")
        if not isinstance(labels, list) or tuple(labels) != EXPECTED_RAW_LABELS[index]:
            raise ConceptContractError(f"raw labels changed for {name}")
        concepts.append(Concept(name, prompt, tuple(aliases), tuple(labels)))
    immutable = tuple(concepts)
    payload = _canonical_payload(1, str(raw["prompt_set_id"]), immutable)
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ConceptRegistry(1, str(raw["prompt_set_id"]), immutable, digest)


def canonical_prompt_texts(registry: ConceptRegistry) -> tuple[str, str]:
    if registry.channel_order != CHANNEL_ORDER:
        raise ConceptContractError("registry channel order is incompatible")
    return registry.prompts  # type: ignore[return-value]


def concept_contract_record(registry: ConceptRegistry) -> dict[str, object]:
    payload = _canonical_payload(registry.schema_version, registry.prompt_set_id, registry.concepts)
    return {**payload, "sha256": registry.sha256}
