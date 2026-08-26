from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
IMAGES = [
    ROOT / "_data/image/01_門神部分(必要標註).jpg",
    ROOT / "_data/image/KJTHT-SC-L-1RB1-1.jpg",
    ROOT / "_data/image/KJTHT-SC-M-A4-8.jpg",
    ROOT / "_data/image/KJTHT-SC-R-A4-3.jpg",
]

CRAQ_PROB_DIR = ROOT / "unet/runs/predict-panels5-2026-06-10/prob"
CRACK_PROB_DIR = ROOT / "unet/runs/prelabel-four-crack-2026-06-14/prob"
OUT_DIR = ROOT / "prelabel_outputs/four_images_2026-06-14"

CRACK_COLOR = np.array([255, 24, 3], dtype=np.uint8)
CRAQ_COLOR = np.array([102, 255, 102], dtype=np.uint8)


def load_prob(prob_path: Path) -> np.ndarray:
    arr = np.load(prob_path)
    if arr.ndim == 3:
        return arr[1].astype(np.float32)
    return arr.astype(np.float32)


def filter_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0 or not mask.any():
        return mask
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def colorize(label: np.ndarray) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    out[label == 1] = CRACK_COLOR
    out[label == 2] = CRAQ_COLOR
    return out


def overlay(img: np.ndarray, label: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    colors = colorize(label)
    fg = label > 0
    out = img.copy()
    out[fg] = (alpha * colors[fg] + (1.0 - alpha) * img[fg]).astype(np.uint8)
    return out


def save_preview(path: Path, img: np.ndarray, max_side: int = 1800) -> None:
    im = Image.fromarray(img)
    scale = min(1.0, max_side / max(im.size))
    if scale < 1.0:
        new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
    im.save(path, quality=92)


def sliding_coords(h: int, w: int, tile: int = 512, stride: int = 384):
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if h < tile or (h - tile) % stride != 0:
        ys.append(max(0, h - tile))
    if w < tile or (w - tile) % stride != 0:
        xs.append(max(0, w - tile))
    seen = set()
    for y in ys:
        for x in xs:
            if (y, x) not in seen:
                seen.add((y, x))
                yield y, x


def write_tiles(stem: str, img: np.ndarray, label: np.ndarray, crack_prob: np.ndarray, craq_prob: np.ndarray):
    tile_dir = OUT_DIR / "top_tile_overlays" / stem
    tile_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    h, w = label.shape
    for y, x in sliding_coords(h, w):
        lab = label[y : y + 512, x : x + 512]
        crack_px = int((lab == 1).sum())
        craq_px = int((lab == 2).sum())
        fg_px = crack_px + craq_px
        if fg_px == 0:
            continue
        cp = crack_prob[y : y + 512, x : x + 512]
        qp = craq_prob[y : y + 512, x : x + 512]
        score = float(fg_px / lab.size + 0.25 * max(cp.max(initial=0), qp.max(initial=0)))
        rows.append(
            {
                "image": stem,
                "x": x,
                "y": y,
                "w": lab.shape[1],
                "h": lab.shape[0],
                "foreground_px": fg_px,
                "crack_px": crack_px,
                "craquelure_px": craq_px,
                "foreground_ratio": fg_px / lab.size,
                "crack_max_prob": float(cp.max(initial=0)),
                "craquelure_max_prob": float(qp.max(initial=0)),
                "score": score,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    for rank, row in enumerate(rows[:20], 1):
        x, y = int(row["x"]), int(row["y"])
        crop = img[y : y + 512, x : x + 512]
        lab = label[y : y + 512, x : x + 512]
        ov = overlay(crop, lab)
        Image.fromarray(ov).save(tile_dir / f"rank{rank:02d}_x{x}_y{y}.jpg", quality=92)
    return rows


def main() -> None:
    for sub in ("semantic_mask", "color_mask", "overlay", "preview", "top_tile_overlays"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    all_tile_rows = []
    summary = []
    for image_path in IMAGES:
        stem = image_path.stem
        img = np.array(Image.open(image_path).convert("RGB"))
        h, w = img.shape[:2]
        crack_prob = load_prob(CRACK_PROB_DIR / f"{stem}.npy")
        craq_prob = load_prob(CRAQ_PROB_DIR / f"{stem}.npy")
        if crack_prob.shape != (h, w) or craq_prob.shape != (h, w):
            raise ValueError(
                f"shape mismatch for {stem}: image={(h, w)} "
                f"crack={crack_prob.shape} craq={craq_prob.shape}"
            )

        # Conservative pre-label: these are candidates for human correction, not GT.
        crack = filter_components(crack_prob >= 0.60, min_area=20)
        craq = filter_components(craq_prob >= 0.70, min_area=40)

        label = np.zeros((h, w), dtype=np.uint8)
        label[craq] = 2
        label[crack] = 1  # crack has display/export priority on overlap.

        Image.fromarray(label).save(OUT_DIR / "semantic_mask" / f"{stem}.png")
        Image.fromarray(colorize(label)).save(OUT_DIR / "color_mask" / f"{stem}.png")
        ov = overlay(img, label)
        Image.fromarray(ov).save(OUT_DIR / "overlay" / f"{stem}.jpg", quality=92)
        save_preview(OUT_DIR / "preview" / f"{stem}_preview.jpg", ov)

        rows = write_tiles(stem, img, label, crack_prob, craq_prob)
        all_tile_rows.extend(rows)
        summary.append(
            {
                "image": stem,
                "width": w,
                "height": h,
                "crack_pixels": int((label == 1).sum()),
                "craquelure_pixels": int((label == 2).sum()),
                "crack_ratio": float((label == 1).mean()),
                "craquelure_ratio": float((label == 2).mean()),
                "tile_candidates": len(rows),
                "thresholds": {"crack": 0.60, "craquelure": 0.70},
            }
        )

    keys = [
        "image",
        "x",
        "y",
        "w",
        "h",
        "foreground_px",
        "crack_px",
        "craquelure_px",
        "foreground_ratio",
        "crack_max_prob",
        "craquelure_max_prob",
        "score",
    ]
    with (OUT_DIR / "tile_candidates_512_stride384.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_tile_rows)
    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(
            {
                "source_notion_page": "Label Overlay Visualization — Craquelure / Crack / Shrinkage",
                "labelmap": {
                    "0": "background",
                    "1": "crack, rgb=(255,24,3)",
                    "2": "craquelure, rgb=(102,255,102)",
                },
                "note": "Pre-label candidates only. Thresholded from local crack/craquelure model probabilities.",
                "summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
