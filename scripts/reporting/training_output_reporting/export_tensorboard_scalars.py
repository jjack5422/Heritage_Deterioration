"""Export mandatory TensorBoard scalars to canonical CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "loss/train": "train_loss",
    "loss/validation": "val_loss",
    "metrics/f1": "f1",
    "metrics/precision": "precision",
    "metrics/recall": "recall",
    "metrics/iou": "iou",
    "optimizer/lr": "learning_rate",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--scalars-out", type=Path, required=True)
    parser.add_argument("--epochs-out", type=Path, required=True)
    args = parser.parse_args()
    events = EventAccumulator(str(args.logdir), size_guidance={"scalars": 0})
    events.Reload()
    available = set(events.Tags().get("scalars", ()))
    missing = sorted(set(TAGS) - available)
    if missing:
        raise RuntimeError(f"missing scalar tags: {missing}")
    scalar_rows: list[dict[str, object]] = []
    epoch_rows: dict[int, dict[str, object]] = {}
    for tag, column in TAGS.items():
        for item in events.Scalars(tag):
            scalar_rows.append({"tag": tag, "step": item.step, "wall_time": item.wall_time, "value": item.value})
            epoch_rows.setdefault(item.step, {"epoch": item.step})[column] = item.value
    args.scalars_out.parent.mkdir(parents=True, exist_ok=True)
    with args.scalars_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("tag", "step", "wall_time", "value"))
        writer.writeheader()
        writer.writerows(sorted(scalar_rows, key=lambda row: (int(row["step"]), str(row["tag"]))))
    columns = ("epoch", *TAGS.values())
    with args.epochs_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(epoch_rows[epoch] for epoch in sorted(epoch_rows))


if __name__ == "__main__":
    main()
