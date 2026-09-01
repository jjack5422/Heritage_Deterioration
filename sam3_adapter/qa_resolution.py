"""Create the approved 20-tile native-resolution preprocessing QA contact sheets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from sam2_adapter.data import H0TileDataset, denormalize_image, prepare_data_plan
from sam2_adapter.runtime import _seed_everything
from sam3_adapter.probe_models import Sam2ProbeModel, Sam3ProbeModel


WORKSPACE = Path(__file__).resolve().parent.parent


def _representatives(plan, count: int = 20) -> list[str]:
    candidates = []
    for name in plan.train:
        with Image.open(plan.root / "masks" / name) as mask_image:
            mask = np.asarray(mask_image)
        with Image.open(plan.root / "images" / name) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
        foreground = int((mask == 1).sum())
        if foreground:
            candidates.append((str(plan.index[name]["source_group"]), foreground, float(gray.std()), name))
    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    selected: list[str] = []
    seen: set[str] = set()
    for group, _, _, name in candidates:
        if group not in seen:
            selected.append(name); seen.add(group)
        if len(selected) == count:
            return selected
    for _, _, _, name in candidates:
        if name not in selected:
            selected.append(name)
        if len(selected) == count:
            break
    return selected


def _panel(image: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor, model_size: int) -> np.ndarray:
    rgb = denormalize_image(image).unsqueeze(0)
    resized = F.interpolate(rgb, size=(model_size, model_size), mode="bilinear", align_corners=False, antialias=True)
    roundtrip = F.interpolate(resized, size=(512, 512), mode="bilinear", align_corners=False, antialias=True)[0]
    source = rgb[0].permute(1, 2, 0).mul(255).round().byte().numpy()
    restored = roundtrip.permute(1, 2, 0).mul(255).round().byte().numpy()
    overlay = source.astype(np.float32)
    gt = target.numpy() == 1
    pred = prediction.numpy()
    overlay[gt] = 0.55 * overlay[gt] + 0.45 * np.array((244, 63, 94))
    overlay[pred] = 0.55 * overlay[pred] + 0.45 * np.array((6, 182, 212))
    return np.concatenate((source, restored, np.clip(overlay, 0, 255).astype(np.uint8)), axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("sam2_probe", "sam3_probe"), required=True)
    args = parser.parse_args()
    _seed_everything(42)
    plan = prepare_data_plan(WORKSPACE / "datasets" / "dataset_clean_v2_merged_craquelure", outer_fold=0)
    names = _representatives(plan)
    dataset = H0TileDataset(plan, names, train_augmentation=False)
    checkpoint = WORKSPACE / ("segment-anything-2/checkpoints/sam2.1_hiera_large.pt" if args.group == "sam2_probe" else "segment-anything-3/checkpoints/sam3.pt")
    model = (Sam2ProbeModel if args.group == "sam2_probe" else Sam3ProbeModel)(checkpoint).eval()
    destination = WORKSPACE / "sam3_adapter" / "runs" / "preflight" / "resolution_qa" / args.group
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(dataset), 4):
        items = [dataset[index] for index in range(start, min(start + 4, len(dataset)))]
        images = torch.stack([item["image"] for item in items]).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            probabilities = torch.sigmoid(model(images)).cpu()
        for item, probability in zip(items, probabilities, strict=True):
            name = str(item["name"])
            output = destination / f"{Path(name).stem}.png"
            Image.fromarray(_panel(item["image"], item["mask"], probability[0] >= 0.5, model.model_input_size)).save(output)
            rows.append((name, item["source_group"], model.model_input_size, output.name))
    with (destination / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("image", "source_group", "model_input_size", "panel")); writer.writerows(rows)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
