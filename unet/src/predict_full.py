"""ResUNet (smp) sliding-window 推論, 與 sam2 版的 predict_full.py 介面對齊。"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import crackseg_common.dataset as _dataset
from crackseg_common.augment import IMAGENET_MEAN, IMAGENET_STD
from crackseg_common.dataset import set_class_names
from crackseg_common.metrics import ConfusionMeter, format_metrics
from PIL import Image
from tqdm import tqdm

from crackseg_common.thresholding import ThresholdPolicy, apply_threshold
from unet_model import build_resunet

_USE_CLAHE = False  # set by --clahe; matches train aug A.CLAHE(LAB L-channel, grid 8x8)


def clahe_tile(tile_uint8, clip=2.0, grid=(8, 8)):
    lab = cv2.cvtColor(tile_uint8, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


_DEFAULT_PALETTE = {
    "background": (0, 0, 0),
    "crack":      (255, 0, 0),
    "loss":       (0, 255, 255),
    "shrinkage":  (255, 255, 0),
    "craquelure": (255, 0, 255),
}
_FALLBACK = [(0, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 255, 255)]


def build_class_rgb(class_names):
    rgb = []
    fb = iter(_FALLBACK)
    for n in class_names:
        rgb.append(_DEFAULT_PALETTE.get(n, next(fb, (200, 200, 200))))
    return np.array(rgb, dtype=np.uint8)


CLASS_NAMES = _dataset.CLASS_NAMES
NUM_CLASSES = _dataset.NUM_CLASSES
CLASS_RGB = build_class_rgb(CLASS_NAMES)


Image.MAX_IMAGE_PIXELS = None  # heritage scans can exceed PIL's bomb limit
def load_image_rgb(path): return np.array(Image.open(path).convert("RGB"))
def load_label(path):
    arr = np.array(Image.open(path))
    return (arr[..., 0] if arr.ndim == 3 else arr).astype(np.uint8)


def resolve_threshold_policy(
    expert: str,
    foreground_id: int = 1,
    threshold: float | None = None,
    policy_path: str | Path | None = None,
) -> ThresholdPolicy:
    policy = (
        ThresholdPolicy.load(policy_path)
        if policy_path is not None
        else ThresholdPolicy.default(expert, foreground_id)
    )
    if policy.expert != expert or policy.foreground_id != foreground_id:
        raise ValueError(
            f"threshold policy expert 不符: policy={policy.expert}/{policy.foreground_id}, "
            f"checkpoint={expert}/{foreground_id}"
        )
    return policy.with_manual_threshold(threshold) if threshold is not None else policy


def discover_threshold_policy(checkpoint: str | Path) -> Path | None:
    """Find ``foldN/config/threshold_policy.json`` beside a standard best.pt."""

    path = Path(checkpoint)
    if len(path.parents) < 3:
        return None
    candidate = path.parents[2] / "config" / "threshold_policy.json"
    return candidate if candidate.is_file() else None


def resolve_inference_sources(
    checkpoint: str | None,
    image: str | None,
    image_dir: str | None,
    preview_threshold: bool,
    chooser=None,
) -> tuple[str, str | None, str | None]:
    """Use desktop file pickers only for missing single-image preview inputs."""

    if preview_threshold and image_dir is None and (checkpoint is None or image is None):
        if chooser is None:
            from threshold_preview import choose_preview_sources

            chooser = choose_preview_sources
        checkpoint, image = chooser(checkpoint, image)
    if checkpoint is None:
        raise ValueError("請指定 checkpoint，或使用 --preview-threshold 開啟選擇視窗")
    if (image is None) == (image_dir is None):
        raise ValueError("請指定 image 或 image_dir 其中之一")
    return str(checkpoint), image, image_dir


def files_from_split_plan(
    image_dir: str | Path,
    split_plan_path: str | Path,
    split: str,
) -> list[str]:
    plan = json.loads(Path(split_plan_path).read_text(encoding="utf-8"))
    if split not in {"train", "val", "test"} or not isinstance(plan.get(split), list):
        raise ValueError(f"split_plan 沒有有效的 {split!r} 清單")
    paths = [Path(image_dir) / str(name) for name in plan[split]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"split 缺少影像: {missing[0]}")
    return [str(path) for path in paths]


def gaussian_window(tile_size, sigma_ratio=0.125):
    sigma = max(1.0, tile_size * sigma_ratio)
    coords = np.arange(tile_size) - (tile_size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    w = np.outer(g, g).astype(np.float32)
    w /= w.max()
    return w


def sliding_coords(H, W, tile, stride):
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if (H - tile) % stride != 0 or H < tile:
        ys.append(max(0, H - tile))
    if (W - tile) % stride != 0 or W < tile:
        xs.append(max(0, W - tile))
    coords, seen = [], set()
    for y in ys:
        for x in xs:
            if (y, x) not in seen:
                seen.add((y, x))
                coords.append((y, x))
    return coords


def pad_to_min(img, tile):
    h, w = img.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    if img.ndim == 3:
        out = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    else:
        out = np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=0)
    return out, (pad_h, pad_w)


def normalize_tile(tile_uint8, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    x = torch.from_numpy(tile_uint8).float().div_(255.0).permute(2, 0, 1)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def predict_full(model, img, device, tile=512, stride=384,
                 batch_size=4, tta_flip=False, use_amp=True):
    H0, W0 = img.shape[:2]
    img_p, _ = pad_to_min(img, tile)
    H, W = img_p.shape[:2]
    coords = sliding_coords(H, W, tile, stride)
    win = gaussian_window(tile)
    prob_canvas = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    weight_canvas = np.zeros((H, W), dtype=np.float32)
    buffer_tiles, buffer_pos = [], []

    def flush():
        if not buffer_tiles:
            return
        x = torch.stack(buffer_tiles, dim=0).to(device, non_blocking=True)
        if use_amp and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                if tta_flip:
                    logits = logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])
                    logits = logits + torch.flip(model(torch.flip(x, dims=[-2])), dims=[-2])
                    logits = logits / 3.0
        else:
            logits = model(x)
            if tta_flip:
                logits = logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])
                logits = logits + torch.flip(model(torch.flip(x, dims=[-2])), dims=[-2])
                logits = logits / 3.0
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        for p, (y, x_) in zip(probs, buffer_pos):
            prob_canvas[:, y:y + tile, x_:x_ + tile] += p * win[None, :, :]
            weight_canvas[y:y + tile, x_:x_ + tile] += win
        buffer_tiles.clear()
        buffer_pos.clear()

    for (y, x) in coords:
        t = img_p[y:y + tile, x:x + tile]
        if _USE_CLAHE:
            t = clahe_tile(t)
        buffer_tiles.append(normalize_tile(t))
        buffer_pos.append((y, x))
        if len(buffer_tiles) >= batch_size:
            flush()
    flush()

    weight_canvas = np.maximum(weight_canvas, 1e-6)
    prob_canvas /= weight_canvas[None, :, :]
    return prob_canvas[:, :H0, :W0]


def colorize_label(label):
    return CLASS_RGB[label.clip(0, NUM_CLASSES - 1)]


def overlay(img, label, alpha=0.5):
    color = colorize_label(label)
    fg = label > 0
    out = img.copy()
    out[fg] = (alpha * color[fg] + (1 - alpha) * img[fg]).astype(np.uint8)
    return out


def load_model_from_ckpt(ckpt_path, device):
    global CLASS_NAMES, NUM_CLASSES, CLASS_RGB
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = payload.get("args", {})
    encoder = args.get("encoder", "resnet50")
    cls_str = args.get("class_names", "background,craquelure")
    names = [s.strip() for s in cls_str.split(",") if s.strip()]
    set_class_names(names)
    CLASS_NAMES = _dataset.CLASS_NAMES
    NUM_CLASSES = _dataset.NUM_CLASSES
    CLASS_RGB = build_class_rgb(CLASS_NAMES)
    print(f"ckpt encoder={encoder} class_names={CLASS_NAMES}")
    model = build_resunet(encoder=encoder, encoder_weights=None,
                          num_classes=NUM_CLASSES).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "predict"),
    )
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tta_flip", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_prob", action="store_true")
    parser.add_argument("--threshold", type=float, default=None,
                        help="手動 foreground threshold；未指定時讀 policy 或使用 0.50")
    parser.add_argument("--threshold-policy", default=None,
                        help="validation 校正產生的 threshold_policy.json")
    parser.add_argument("--split-plan", default=None,
                        help="只推論 split_plan.json 指定的清單")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--preview-threshold", action="store_true",
                        help="開啟本地 threshold 拉桿；未指定路徑時使用桌面選擇視窗")
    parser.add_argument("--clahe", action="store_true",
                        help="test-time CLAHE per tile (matches train aug)")
    args = parser.parse_args()
    global _USE_CLAHE
    _USE_CLAHE = args.clahe

    try:
        args.ckpt, args.image, args.image_dir = resolve_inference_sources(
            args.ckpt,
            args.image,
            args.image_dir,
            args.preview_threshold,
        )
    except ValueError as error:
        parser.error(str(error))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    for sub in ("label", "color", "overlay"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    if args.save_prob:
        (out_dir / "prob").mkdir(parents=True, exist_ok=True)

    model, payload = load_model_from_ckpt(args.ckpt, device)
    if NUM_CLASSES != 2:
        raise SystemExit("threshold policy 只支援 background + 單一劣化的二元 expert checkpoint")
    checkpoint_expert = payload.get("expert", {})
    expert = str(checkpoint_expert.get("name", CLASS_NAMES[1]))
    foreground_id = int(checkpoint_expert.get("source_class_id", 1))
    policy_path = args.threshold_policy or discover_threshold_policy(args.ckpt)
    policy = resolve_threshold_policy(
        expert,
        foreground_id,
        threshold=args.threshold,
        policy_path=policy_path,
    )
    policy.save(out_dir / "threshold_policy.json")
    print(f"loaded ckpt={args.ckpt} epoch={payload.get('epoch')} val={payload.get('val', {}).get('miou')}")

    if args.image:
        items = [args.image]
    elif args.split_plan:
        items = files_from_split_plan(args.image_dir, args.split_plan, args.split)
    else:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        items = sorted([os.path.join(args.image_dir, f)
                        for f in os.listdir(args.image_dir)
                        if f.lower().endswith(exts)])
    if args.preview_threshold and len(items) != 1:
        raise SystemExit("--preview-threshold 目前只支援單一 --image")
    print(
        f"images={len(items)} tile={args.tile} stride={args.stride} "
        f"tta_flip={args.tta_flip} threshold={policy.threshold:.2f} source={policy.source}"
    )

    overall = ConfusionMeter(NUM_CLASSES)
    per_image_rows = []

    for path in tqdm(items):
        stem = Path(path).stem
        img = load_image_rgb(path)
        prob = predict_full(model, img, device,
                            tile=args.tile, stride=args.stride,
                            batch_size=args.batch_size, tta_flip=args.tta_flip,
                            use_amp=not args.no_amp)
        label = apply_threshold(prob, policy.threshold)

        Image.fromarray(label).save(out_dir / "label" / f"{stem}.png")
        Image.fromarray(colorize_label(label)).save(out_dir / "color" / f"{stem}.png")
        Image.fromarray(overlay(img, label)).save(out_dir / "overlay" / f"{stem}.png")
        if args.save_prob:
            np.save(out_dir / "prob" / f"{stem}.npy", prob[1].astype(np.float32))

        if args.mask_dir is not None:
            gt_path = os.path.join(args.mask_dir, stem + ".png")
            if os.path.isfile(gt_path):
                raw_gt = load_label(gt_path)
                gt = np.where(raw_gt == 255, 255, raw_gt == foreground_id).astype(np.uint8)
                if gt.shape == label.shape:
                    per_meter = ConfusionMeter(NUM_CLASSES)
                    per_meter.update(label, gt)
                    overall.update(label, gt)
                    res = per_meter.compute(class_names=CLASS_NAMES, ignore_index=0)
                    per_image_rows.append({
                        "image": stem,
                        "miou": res["miou"],
                        "mdice": res["mdice"],
                        "pixel_acc": res["pixel_accuracy"],
                        **{f"iou_{k}": v["iou"] for k, v in res["per_class"].items()},
                    })
                else:
                    print(f"[warn] shape mismatch {stem}: pred={label.shape} gt={gt.shape}")

    if args.mask_dir is not None and per_image_rows:
        res = overall.compute(class_names=CLASS_NAMES, ignore_index=0)
        print("=== overall ===")
        print(format_metrics(res))
        with open(out_dir / "overall_metrics.json", "w") as f:
            json.dump(res, f, indent=2)
        keys = list(per_image_rows[0].keys())
        with open(out_dir / "per_image_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_image_rows:
                w.writerow(r)
        print(f"輸出: {out_dir}")

    (out_dir / "inference.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.ckpt).resolve()),
                "expert": expert,
                "foreground_id": foreground_id,
                "threshold": policy.threshold,
                "threshold_source": policy.source,
                "inference_rule": policy.inference_rule,
                "image_count": len(items),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    if args.preview_threshold:
        from threshold_preview import launch_viewer

        target = None
        if args.mask_dir:
            target = load_label(Path(args.mask_dir) / f"{Path(items[0]).stem}.png")
        launch_viewer(
            image=img,
            probability=prob,
            policy=policy,
            output_dir=out_dir / "threshold_preview",
            stem=Path(items[0]).stem,
            target=target,
        )


if __name__ == "__main__":
    main()
