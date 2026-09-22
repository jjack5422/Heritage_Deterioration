"""Export the locked outer-test predictions from each selected expert checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2_adapter.h0_core import load_trainable_state_dict
from sam2_adapter.reporting import _mask_rgb, _overlay_rgb, binary_metric_row
from sam2_adapter.runtime import _autocast, _batch_tensor, _seed_everything
from sam3_adapter.training_data import denormalize_image, prepare_expert_data_plan
from sam3_adapter.train import _build_model, _loader, parse_args


ROOT = Path(__file__).resolve().parent.parent
EXPERTS = ("scratch_crack", "loss", "shrinkage_craquelure")


def _gallery(expert: str, rows: list[dict[str, object]]) -> str:
    cards = []
    for row in rows:
        score = "N/A (empty GT)" if row["f1"] == "" else f"{float(row['f1']):.3f}"
        cards.append(
            f'<article><a href="{html.escape(str(row["composite_path"]), quote=True)}">'
            f'<img loading="lazy" src="{html.escape(str(row["composite_path"]), quote=True)}" '
            f'alt="Test inference: {html.escape(str(row["image"]), quote=True)}"></a>'
            f'<p><code>{html.escape(str(row["source_group"]))}</code> · '
            f'{html.escape(str(row["image"]))} · F1 {score}</p></article>'
        )
    return ("<!doctype html><html lang=\"zh-Hant\"><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(expert)} test inference</title>"
            "<style>body{font:16px system-ui;margin:2rem;max-width:1500px;background:#111827;color:#f9fafb}"
            "a{color:#67e8f9}article{margin:1.5rem 0;padding:1rem;background:#1f2937;border-radius:10px}"
            "img{width:100%;height:auto}p{overflow-wrap:anywhere}code{color:#fbbf24}</style>"
            f"<h1>{html.escape(expert)} — outer test inference</h1>"
            "<p>每張圖由左至右：原圖、GT（紅）、預測（青）、疊圖。疊圖紫色表示重疊。"
            "點圖可看原尺寸。閾值 0.5；test 未用於 checkpoint 選擇。</p>"
            f"<p>共 {len(rows)} 張，依來源圖群組及 tile 排序。"
            "單張 F1 在空白 GT 時標為 N/A。</p>"
            + "\n".join(cards) + "</html>\n")


def export_one(experiment_id: str, expert: str) -> Path:
    manifest = ROOT / "outputs" / "deterioration_statistics" / "transfer_512" / f"{expert}.json"
    args = parse_args(["--expert", expert, "--manifest", str(manifest), "--model-input-size", "512",
                       "--epochs", "60", "--num-workers", "0", "--experiment-id", experiment_id])
    plan = prepare_expert_data_plan(args.manifest, expert)
    fold = ROOT / "sam3_adapter" / "runs" / experiment_id / "1fold" / expert / "fold0"
    best_path = fold / "artifacts" / "checkpoints" / "best.pt"
    expected = json.loads((fold / "metrics" / "outer_test_metrics.json").read_text(encoding="utf-8"))
    selected = json.loads((fold / "metrics" / "experiment_summary.json").read_text(encoding="utf-8"))
    if expected["status"] != "completed" or selected["status"] != "completed":
        raise RuntimeError(f"{expert}: training/reporting is incomplete")
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    if (payload["expert"] != expert or payload["split_sha256"] != plan.split_sha256
            or payload["dataset_sha256"] != plan.adopted_dataset_sha256
            or int(payload["epoch"]) != int(selected["selected_epoch"])):
        raise RuntimeError(f"{expert}: checkpoint and test manifest do not match")
    _seed_everything(args.seed)
    device = torch.device("cuda")
    model = _build_model(args, device)
    load_trainable_state_dict(model, payload["adaptation_state"])
    model.eval()
    destination = fold / "test_inference"
    images_dir = destination / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    loader = _loader(plan.test, raw_ids=plan.raw_ids, train=False, args=args)
    source_groups = {str(row["tile"]): str(row["source_group"]) for row in plan.test}
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            images = _batch_tensor(batch, "image", device)
            target = _batch_tensor(batch, "target", device)
            with _autocast(device, args.amp):
                logits = model(images)
            predictions = (torch.sigmoid(logits[:, 0]) >= 0.5).cpu().numpy()
            for index, name in enumerate(batch["name"]):
                rgb = denormalize_image(images[index]).permute(1, 2, 0).mul(255).round().byte().numpy()
                gt = target[index].cpu().numpy() == 1
                pred = predictions[index]
                metric = binary_metric_row(gt, pred)
                panels = (rgb, _mask_rgb(gt, color=(244, 63, 94)),
                          _mask_rgb(pred, color=(6, 182, 212)), _overlay_rgb(rgb, gt, pred))
                file_name = hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:20] + ".png"
                path = images_dir / file_name
                Image.fromarray(np.concatenate(panels, axis=1), mode="RGB").save(path)
                rows.append({"image": name, "source_group": source_groups[str(name).split("__", 1)[1]],
                             **metric, "composite_path": f"images/{file_name}"})
            print(f"{expert}: {len(rows)}/{len(plan.test)} test tiles", flush=True)
    del model
    torch.cuda.empty_cache()
    totals = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn")}
    reference = expected["tile_micro"]
    if any(totals[key] != int(reference[key]) for key in totals):
        raise RuntimeError(f"{expert}: inference totals {totals} disagree with reported test {reference}")
    rows.sort(key=lambda row: (str(row["source_group"]), str(row["image"])))
    with (destination / "per_image_test.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "index.html").write_text(_gallery(expert, rows), encoding="utf-8")
    print(f"VERIFIED {expert}: {destination / 'index.html'}", flush=True)
    return destination / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="2026-09-20_transfer-512_seed42")
    parser.add_argument("--expert", choices=EXPERTS, action="append")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for test inference")
    for expert in args.expert or EXPERTS:
        export_one(args.experiment_id, expert)


if __name__ == "__main__":
    main()
