from pathlib import Path

import pytest

from monument_da_sam3.concepts import (
    CHANNEL_ORDER,
    ConceptContractError,
    canonical_prompt_texts,
    load_concept_registry,
)


CONFIG = Path(__file__).parents[1] / "configs" / "concepts.yaml"


def test_approved_registry_is_stable() -> None:
    registry = load_concept_registry(CONFIG)
    assert registry.channel_order == CHANNEL_ORDER
    assert canonical_prompt_texts(registry) == (
        "craquelure, a network of fine cracks in the painted surface",
        "paint loss, a missing area in the paint layer that exposes the substrate",
    )
    assert len(registry.sha256) == 64
    assert "surface crack" not in registry.prompts


def test_changed_prompt_is_rejected(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace("fine cracks", "tiny cracks", 1)
    path = tmp_path / "concepts.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConceptContractError, match="canonical prompt changed"):
        load_concept_registry(path)
