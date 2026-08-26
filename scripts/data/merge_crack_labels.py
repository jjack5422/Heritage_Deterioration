"""Create a clean dataset where Crack and Craquelure share one label.

The source dataset is never modified.  The output keeps the same image names
and five-fold membership while remapping the seven-class label IDs to six
contiguous IDs::

    background=0, craquelure=1, loss=2, shrinkage=3, flaking=4, stain=5

Source Crack (1) and source Craquelure (4) both become Craquelure (1).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SOURCE_CLASS_IDS = {
    "background": 0,
    "crack": 1,
    "loss": 2,
    "shrinkage": 3,
    "craquelure": 4,
    "flaking": 5,
    "stain": 6,
}
MERGED_CLASS_IDS = {
    "background": 0,
    "craquelure": 1,
    "loss": 2,
    "shrinkage": 3,
    "flaking": 4,
    "stain": 5,
}
SOURCE_TO_MERGED = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 4, 6: 5, 255: 255}
IGNORE_VALUE = 255


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_rows(rows: list[tuple[str, str]]) -> str:
    payload = "".join(f"{name}\t{digest}\n" for name, digest in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Merge Crack/Craquelure and return a contiguous uint8 label mask."""

    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must be a two-dimensional label-ID image")
    unknown = set(np.unique(values).tolist()) - set(SOURCE_TO_MERGED)
    if unknown:
        raise ValueError(f"mask contains unsupported source label IDs: {sorted(unknown)}")
    lookup = np.arange(256, dtype=np.uint8)
    for source, target in SOURCE_TO_MERGED.items():
        lookup[source] = target
    return lookup[values.astype(np.uint8, copy=False)]


def _update_split(
    source_split: dict[str, Any],
    *,
    pair_count: int,
    names_hash: str,
    tile_index_hash: str,
    image_hash: str,
    mask_hash: str,
    source_manifest_hash: str,
) -> dict[str, Any]:
    split = copy.deepcopy(source_split)
    split.update(
        {
            "schema_version": "crackseg-data-view-v4",
            "view_role": "clean_v2_merged_craquelure",
            "task": "multiclass_deterioration_merged_craquelure",
            "tiles_root": ".",
        }
    )
    split["data_contract"] = {
        "schema_version": "crackseg-data-contract-v4",
        "task": "multiclass_deterioration_merged_craquelure",
        "task_mode": "multiclass_crack_merged_as_craquelure",
        "task_tiles_root": ".",
        "class_names": list(MERGED_CLASS_IDS),
        "ignore_value": IGNORE_VALUE,
        "label_semantics": {
            "ignore_label": IGNORE_VALUE,
            "allowed_mask_values": [*MERGED_CLASS_IDS.values(), IGNORE_VALUE],
            "class_ids": MERGED_CLASS_IDS,
            "merged_source_labels": ["crack", "craquelure"],
            "merged_target_label": "craquelure",
        },
        "eligible_tile_count": pair_count,
        "eligible_names_sha256": names_hash,
        "task_tile_index_sha256": tile_index_hash,
        "task_image_manifest_sha256": image_hash,
        "task_mask_manifest_sha256": mask_hash,
        "synthetic_data_included": False,
    }
    evaluation = dict(split.get("evaluation_contract") or {})
    evaluation.update(
        {
            "ground_truth_rule": "raw_mask == 1 (merged craquelure)",
            "positive_label": 1,
            "background_labels": [0],
            "scored_background_rule": "raw_mask == 0",
        }
    )
    split["evaluation_contract"] = evaluation
    evidence = dict(split.get("source_evidence") or {})
    evidence.update(
        {
            "source_dataset": "dataset_clean_v2",
            "source_manifest_sha256": source_manifest_hash,
            "label_mapping": {str(key): value for key, value in SOURCE_TO_MERGED.items()},
        }
    )
    split["source_evidence"] = evidence
    split.pop("manifest_sha256", None)
    split["manifest_sha256"] = _json_hash(split)
    return split


def merge_dataset(source_root: str | Path, destination_root: str | Path) -> dict[str, Any]:
    """Write a validated merged copy and return its conversion summary."""

    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists; refusing to overwrite: {destination}")
    if source == destination or source in destination.parents:
        raise ValueError("destination must not be the source or a child of the source dataset")

    source_manifest_path = source / "manifest.json"
    source_index_path = source / "tile_index.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("label_contract", {}).get("class_ids") != SOURCE_CLASS_IDS:
        raise ValueError("source manifest is not the expected clean_v2 seven-class contract")
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    items = source_index.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("source tile_index.json must contain a non-empty items array")
    names = [item.get("tile") for item in items]
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("every tile_index item must contain a tile name")
    if len(names) != len(set(names)):
        raise ValueError("tile_index.json contains duplicate tile names")
    if int(source_manifest.get("pair_count", -1)) != len(names):
        raise ValueError("source manifest pair_count differs from tile_index.json")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        (temporary / "images").mkdir()
        (temporary / "masks").mkdir()
        (temporary / "splits").mkdir()
        source_counts: Counter[int] = Counter()
        merged_counts: Counter[int] = Counter()
        image_rows: list[tuple[str, str]] = []
        mask_rows: list[tuple[str, str]] = []
        for name in sorted(names):
            source_image = source / "images" / name
            source_mask = source / "masks" / name
            if not source_image.is_file() or not source_mask.is_file():
                raise FileNotFoundError(f"missing image/mask pair: {name}")
            with Image.open(source_image) as image, Image.open(source_mask) as mask_image:
                if image.size != mask_image.size:
                    raise ValueError(f"image/mask dimensions differ: {name}")
                raw_mask = np.asarray(mask_image)
            mapped_mask = remap_mask(raw_mask)
            source_unique, source_frequencies = np.unique(raw_mask, return_counts=True)
            merged_unique, merged_frequencies = np.unique(mapped_mask, return_counts=True)
            source_counts.update(dict(zip(source_unique.tolist(), source_frequencies.tolist(), strict=True)))
            merged_counts.update(dict(zip(merged_unique.tolist(), merged_frequencies.tolist(), strict=True)))
            target_image = temporary / "images" / name
            target_mask = temporary / "masks" / name
            shutil.copy2(source_image, target_image)
            Image.fromarray(mapped_mask, mode="L").save(target_mask)
            image_rows.append((name, _sha256_file(target_image)))
            mask_rows.append((name, _sha256_file(target_mask)))

        image_hash = _sha256_rows(image_rows)
        mask_hash = _sha256_rows(mask_rows)
        names_hash = hashlib.sha256("".join(f"{name}\n" for name in sorted(names)).encode("utf-8")).hexdigest()
        source_manifest_hash = _sha256_file(source_manifest_path)
        merged_index = copy.deepcopy(source_index)
        merged_index["summary"] = {
            **dict(merged_index.get("summary") or {}),
            "task": "multiclass_deterioration_merged_craquelure",
            "total_tiles": len(names),
            "source_dataset": str(source),
        }
        _write_json(temporary / "tile_index.json", merged_index)
        tile_index_hash = _sha256_file(temporary / "tile_index.json")

        for split_path in sorted((source / "splits").glob("fold*.json")):
            source_split = json.loads(split_path.read_text(encoding="utf-8"))
            merged_split = _update_split(
                source_split,
                pair_count=len(names),
                names_hash=names_hash,
                tile_index_hash=tile_index_hash,
                image_hash=image_hash,
                mask_hash=mask_hash,
                source_manifest_hash=source_manifest_hash,
            )
            _write_json(temporary / "splits" / split_path.name, merged_split)

        named_source_counts = {
            name: int(source_counts[class_id]) for name, class_id in SOURCE_CLASS_IDS.items()
        }
        named_source_counts["ignore"] = int(source_counts[IGNORE_VALUE])
        named_merged_counts = {
            name: int(merged_counts[class_id]) for name, class_id in MERGED_CLASS_IDS.items()
        }
        named_merged_counts["ignore"] = int(merged_counts[IGNORE_VALUE])
        manifest: dict[str, Any] = {
            "schema_version": "multiclass-pair-authority-v2",
            "authority_id": "clean_v2_merged_craquelure",
            "pair_count": len(names),
            "training_eligible": True,
            "image_manifest_sha256": image_hash,
            "mask_manifest_sha256": mask_hash,
            "names_sha256": names_hash,
            "mask_values": sorted(merged_counts),
            "label_contract": {
                "class_ids": MERGED_CLASS_IDS,
                "ignore_value": IGNORE_VALUE,
                "merge_rule": "source crack and source craquelure become craquelure",
                "source_to_merged_ids": {str(key): value for key, value in SOURCE_TO_MERGED.items()},
            },
            "source_dataset": {
                "path": str(source),
                "manifest_sha256": source_manifest_hash,
                "source_was_modified": False,
            },
            "pixel_counts": {
                "source": named_source_counts,
                "merged": named_merged_counts,
            },
        }
        manifest["manifest_sha256"] = _json_hash(manifest)
        _write_json(temporary / "manifest.json", manifest)
        summary = {
            "dataset_root": str(destination),
            "source_dataset_root": str(source),
            "pair_count": len(names),
            "fold_count": len(list((temporary / "splits").glob("fold*.json"))),
            "class_ids": MERGED_CLASS_IDS,
            "source_pixel_counts": named_source_counts,
            "merged_pixel_counts": named_merged_counts,
            "image_manifest_sha256": image_hash,
            "mask_manifest_sha256": mask_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_was_modified": False,
        }
        _write_json(temporary / "merge_summary.json", summary)
        (temporary / "README.md").write_text(
            "# Clean v2 — Crack merged into Craquelure\n\n"
            "This is a derived copy of `dataset_clean_v2`; the source dataset was not modified.\n\n"
            "Class IDs: background=0, craquelure=1, loss=2, shrinkage=3, "
            "flaking=4, stain=5, ignore=255. Source Crack and Craquelure pixels "
            "are both Craquelure in this dataset. The five outer-fold memberships "
            "are unchanged.\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            merge_dataset(args.source_root, args.output_root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
