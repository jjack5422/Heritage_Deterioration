"""Protect dataset114 review, quarantine, and restoration behavior."""

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.data.review_dataset114 import Dataset114, ReviewStore, create_app


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / "dataset114"
    (root / "MGLST/images").mkdir(parents=True)
    (root / "MGLST/masks").mkdir()
    (root / "metadata").mkdir()
    classes = {
        "background": {"id": 0, "name": "background"},
        "categories": [
            {"id": 1, "code": "D-01", "name": "Crack"},
            {"id": 2, "code": "D-02", "name": "Loss"},
            {"id": 4, "code": "D-04", "name": "Craquelure"},
        ],
    }
    (root / "metadata/classes.json").write_text(json.dumps(classes), encoding="utf-8")
    fields = ["task_id", "image", "mask", "classes_present"]
    rows = []
    for index, class_id in enumerate((1, 4)):
        name = f"tile_{index}"
        image = root / f"MGLST/images/{name}.jpg"
        mask = root / f"MGLST/masks/{name}.png"
        Image.new("RGB", (3, 2), (30, 60, 90)).save(image)
        values = np.zeros((2, 3), dtype=np.uint8)
        values[0, 1] = class_id
        Image.fromarray(values).save(mask)
        rows.append({
            "task_id": f"image_{index}",
            "image": image.relative_to(root).as_posix(),
            "mask": mask.relative_to(root).as_posix(),
            "classes_present": str(class_id),
        })
    with (root / "metadata/manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return root


def manifest_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def auth_client(dataset):
    store = ReviewStore(dataset)
    app = create_app(dataset, store)
    client = app.test_client()
    page = client.get("/").get_data(as_text=True)
    token = json.loads(re.search(r"const token=(\"[^\"]+\")", page).group(1))
    return store, client, {"X-Review-Token": token}


def test_reject_quarantines_pair_updates_manifest_and_resumes(dataset_root):
    dataset = Dataset114(dataset_root)
    store, client, headers = auth_client(dataset)
    item = dataset.items[0]
    original_image = item["image"]
    original_mask = item["mask"]

    response = client.post(
        "/api/review",
        headers=headers,
        json={"index": 0, "status": "reject", "note": "明顯漏標"},
    )

    assert response.status_code == 200
    assert not original_image.exists()
    assert not original_mask.exists()
    assert item["image"].is_file()
    assert item["mask"].is_file()
    assert item["rejected"] is True
    assert len(manifest_rows(dataset_root / "metadata/manifest.csv")) == 1
    assert [row["image"] for row in manifest_rows(dataset_root / "rejected_tiles/manifest.csv")] == [item["key"]]
    assert json.loads((dataset_root / "metadata/manual_review.json").read_text())["records"][item["key"]]["note"] == "明顯漏標"
    assert json.loads((dataset_root / "metadata/manual_review.jsonl").read_text().splitlines()[-1])["action"] == "reject"

    resumed = Dataset114(dataset_root)
    resumed_store = ReviewStore(resumed)
    assert resumed.by_key[item["key"]]["rejected"] is True
    assert resumed_store.records[item["key"]]["status"] == "reject"
    assert resumed.load_pair(0)[0].shape == (2, 3, 3)


def test_undo_restores_pair_manifest_and_previous_decision(dataset_root):
    dataset = Dataset114(dataset_root)
    store, client, headers = auth_client(dataset)
    original_order = [row["image"] for row in manifest_rows(dataset_root / "metadata/manifest.csv")]
    key = dataset.items[0]["key"]
    assert client.post("/api/review", headers=headers, json={"index": 0, "status": "unsure", "note": "先複查"}).status_code == 200
    assert client.post("/api/review", headers=headers, json={"index": 0, "status": "reject", "note": "暫時隔離"}).status_code == 200

    response = client.post("/api/undo", headers=headers)

    assert response.status_code == 200
    assert dataset.by_key[key]["rejected"] is False
    assert dataset.by_key[key]["image"].is_file()
    assert dataset.by_key[key]["mask"].is_file()
    assert store.records[key]["status"] == "unsure"
    assert store.records[key]["note"] == "先複查"
    assert [row["image"] for row in manifest_rows(dataset_root / "metadata/manifest.csv")] == original_order
    assert manifest_rows(dataset_root / "rejected_tiles/manifest.csv") == []
    assert client.post("/api/undo", headers=headers).status_code == 409


def test_display_name_correction_preserves_review_progress(dataset_root):
    dataset = Dataset114(dataset_root)
    store = ReviewStore(dataset)
    store.save(0, "keep", "人工確認")
    classes_path = dataset_root / "metadata/classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    classes["categories"][1]["name"] = "Corrected display name"
    classes_path.write_text(json.dumps(classes), encoding="utf-8")

    renamed_dataset = Dataset114(dataset_root)
    resumed = ReviewStore(renamed_dataset)

    assert renamed_dataset.fingerprint == dataset.fingerprint
    assert resumed.records[dataset.items[0]["key"]]["status"] == "keep"


def test_manifest_path_cannot_escape_dataset(dataset_root, tmp_path):
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (1, 1)).save(outside)
    rows = manifest_rows(dataset_root / "metadata/manifest.csv")
    rows[0]["image"] = "../outside.jpg"
    fields = list(rows[0])
    with (dataset_root / "metadata/manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="escapes dataset root"):
        Dataset114(dataset_root)
