"""Protect pairing, mask semantics, and durable manual-review decisions."""

import csv
import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from scripts.data.review_dataset import ReviewDataset, ReviewStore, create_app


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "source"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "SegmentationClass").mkdir()
    (root / "labelmap.txt").write_text("background:0,0,0::\ncrack:255,24,3::\n", encoding="utf-8")
    for name in ("甲.png", "乙.png"):
        Image.new("RGB", (3, 2), (20, 40, 60)).save(root / "JPEGImages" / name)
        mask = Image.new("RGB", (3, 2))
        mask.putpixel((1, 0), (255, 24, 3))
        mask.save(root / "SegmentationClass" / name)
    return ReviewDataset(root)


def test_mask_overlay_preserves_background_and_class_colors(dataset, tmp_path):
    app = create_app(dataset, ReviewStore(dataset, tmp_path / "review"))
    client = app.test_client()
    response = client.get("/image/0/overlay")
    pixels = np.array(Image.open(io.BytesIO(response.data)))
    assert pixels[0, 0, 3] == 0
    assert pixels[0, 1].tolist() == [255, 24, 3, 255]
    assert client.get("/image/99/mask").status_code == 404
    mask_path = dataset.items[0]["mask"]
    Image.new("RGB", (3, 2), (1, 2, 3)).save(mask_path)
    assert client.get("/image/0/mask").status_code == 422


def test_decisions_resume_and_export_only_kept_images(dataset, tmp_path):
    output = tmp_path / "review"
    store = ReviewStore(dataset, output)
    client = create_app(dataset, store).test_client()
    page = client.get("/").get_data(as_text=True)
    token = json.loads(re.search(r"const token=(\"[^\"]+\")", page).group(1))
    assert client.post("/api/review", json={}).status_code == 403
    headers = {"X-Review-Token": token}
    for index, status in enumerate(("keep", "reject")):
        response = client.post("/api/review", headers=headers, json={"index": index, "status": status, "note": "邊界清楚\n待團隊核對"})
        assert response.status_code == 200
    restored = ReviewStore(dataset, output)
    assert restored.records == store.records
    assert len(list(csv.DictReader((output / "review.csv").open(encoding="utf-8-sig")))) == 2
    selected = list(csv.DictReader(io.StringIO(client.get("/export/selected").data.decode("utf-8-sig"))))
    assert [row["name"] for row in selected] == [dataset.items[0]["name"]]
    assert selected[0]["note"] == "邊界清楚\n待團隊核對"
    assert selected[0]["image"] == dataset.items[0]["image"]
    assert client.post("/api/review", headers=headers, json={"index": -1, "status": "keep"}).status_code == 400
    assert client.post("/api/review", headers=headers, json={"index": 0, "status": []}).status_code == 400
    store.save(0, "unreviewed", "")
    assert not list(csv.DictReader(io.StringIO(store.csv_text(store.records, True))))


def test_saved_progress_cannot_be_applied_to_changed_labels(dataset, tmp_path):
    output = tmp_path / "review"
    ReviewStore(dataset, output).save(0, "keep", "")
    Image.new("RGB", (3, 2)).save(dataset.items[0]["mask"])
    with pytest.raises(ValueError, match="different/changed"):
        ReviewStore(ReviewDataset(dataset.root), output)


def test_missing_pair_and_mismatched_dimensions_are_reported(dataset):
    Image.new("RGB", (4, 2)).save(dataset.items[0]["mask"])
    with pytest.raises(ValueError, match="dimensions differ"):
        dataset.load_pair(0)
    from pathlib import Path

    Path(dataset.items[0]["mask"]).unlink()
    with pytest.raises(ValueError, match="Missing matching mask"):
        ReviewDataset(dataset.root)


def test_id_masks_follow_manifest_instead_of_assuming_class_numbers(tmp_path):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "manifest.json").write_text(json.dumps({"label_contract": {"class_ids": {"background": 0, "craquelure": 1, "loss": 2}, "ignore_value": 255}}))
    Image.new("RGB", (3, 1)).save(root / "images/a.png")
    Image.fromarray(np.array([[0, 1, 255]], dtype=np.uint8)).save(root / "masks/a.png")
    _, colored, foreground = ReviewDataset(root).load_pair(0)
    assert colored.tolist() == [[[0, 0, 0], [102, 255, 102], [160, 160, 160]]]
    assert foreground.tolist() == [[False, True, True]]


def test_failed_save_keeps_previous_decision(dataset, tmp_path, monkeypatch):
    store = ReviewStore(dataset, tmp_path / "review")
    store.save(0, "keep", "original note")

    def fail(*args):
        raise OSError("Disk full")

    monkeypatch.setattr("scripts.data.review_dataset.atomic_write", fail)
    with pytest.raises(OSError, match="Disk full"):
        store.save(0, "reject", "changed note")
    restored = ReviewStore(dataset, store.output)
    assert restored.records == store.records
    assert restored.records[dataset.items[0]["name"]]["status"] == "keep"
