"""Export Best/Worst TensorBoard composites and the loss curve."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from PIL import Image, ImageDraw
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = EventAccumulator(str(args.logdir), size_guidance={"images": 0, "scalars": 0})
    events.Reload()
    train, validation = events.Scalars("loss/train"), events.Scalars("loss/validation")
    canvas = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(canvas)
    bounds = (110, 80, 1430, 800)
    draw.rectangle(bounds, outline="#667085", width=3)
    values = [row.value for row in (*train, *validation)]
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    max_step = max(row.step for row in (*train, *validation))
    def points(series):
        return [(
            bounds[0] + (row.step / max_step) * (bounds[2] - bounds[0]),
            bounds[3] - ((row.value - low) / span) * (bounds[3] - bounds[1]),
        ) for row in series]
    draw.line(points(train), fill="#2563eb", width=5)
    draw.line(points(validation), fill="#dc2626", width=5)
    draw.text((110, 30), "Training and validation loss", fill="#111827")
    draw.text((1120, 30), "Train", fill="#2563eb")
    draw.text((1250, 30), "Validation", fill="#dc2626")
    draw.text((20, 80), f"{high:.4f}", fill="#344054")
    draw.text((20, 780), f"{low:.4f}", fill="#344054")
    draw.text((700, 835), "Epoch", fill="#344054")
    canvas.save(args.output_dir / "loss_curve.png")

    rows: list[dict[str, object]] = []
    for tag in sorted(events.Tags().get("images", ())):
        parts = tag.split("/")
        if len(parts) < 3 or parts[:2] not in (["qualitative", "best"], ["qualitative", "worst"]):
            continue
        group, item = parts[1], events.Images(tag)[-1]
        destination = args.output_dir / group / f"{parts[-1]}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(io.BytesIO(item.encoded_image_string)) as image:
            image.convert("RGB").save(destination)
        rows.append({"group": group, "rank": int(parts[-1].split("_", 1)[0]), "tag": tag, "step": item.step, "path": destination.relative_to(args.output_dir).as_posix()})
    if {str(row["group"]) for row in rows} != {"best", "worst"}:
        raise RuntimeError("incomplete Best/Worst TensorBoard images")
    with (args.output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("group", "rank", "tag", "step", "path"))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["group"]), int(row["rank"]))))


if __name__ == "__main__":
    main()
