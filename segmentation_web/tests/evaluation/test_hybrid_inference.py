"""Verify independent masks, atomic model switching, and hybrid web results."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from adapters.base import SegmentationAdapter
from api import create_app
from config import load_settings
from imaging.deterioration_overlay import compose_deterioration_masks
from inference import HybridInferenceError, InferenceManager
from registry import (
    HYBRID_CLASS_ID, HYBRID_COMPONENTS, HYBRID_MODEL_ID, HYBRID_WEIGHT,
    InvalidWeightError, get_weights,
)
from ui import ApiClientError, InferenceApiClient, create_ui


class ExpertAdapter(SegmentationAdapter):
    def __init__(self, model, events, *, fail=False, gate=None):
        self.model, self.events, self.fail, self.gate = model, events, fail, gate

    def load(self, path):
        self.events.append(("load", self.model, path.name))

    def predict(self, image, threshold=0.5):
        self.events.append(("predict", self.model, threshold))
        if self.gate:
            entered, release = self.gate
            entered.set()
            assert release.wait(5)
        if self.fail:
            raise RuntimeError("test inference failure")
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        index = [item["model"] for item in HYBRID_COMPONENTS].index(self.model)
        mask[0, index] = 255
        mask[1:, 2:10] = 255
        return {"mask": mask, "overlay": image.copy()}

    def unload(self):
        self.events.append(("unload", self.model))


def manager_for_test(tmp_path, *, fail=None, gate=None, resolver=None):
    events = []
    factories = {
        item["model"]: (
            lambda model=item["model"]: ExpertAdapter(
                model, events, fail=model == fail,
                gate=gate if model == HYBRID_COMPONENTS[0]["model"] else None,
            )
        ) for item in HYBRID_COMPONENTS
    }
    manager = InferenceManager(
        model_root=tmp_path, adapter_factories=factories,
        weight_resolver=resolver or (lambda model, weight, root: root / model / weight),
    )
    return manager, events


def test_composition_preserves_colors_background_and_double_triple_overlaps():
    source = Image.new("RGB", (24, 3), (12, 34, 56))
    arrays = {item["class"]: np.zeros((3, 24), dtype=np.uint8) for item in HYBRID_COMPONENTS}
    for index, item in enumerate(HYBRID_COMPONENTS):
        arrays[item["class"]][0, index] = 255
        arrays[item["class"]][2, :] = 255  # triple overlaps: both hatch and charcoal
    arrays["craquelure"][1, :] = arrays["crack"][1, :] = 255
    masks = {key: Image.fromarray(value) for key, value in arrays.items()}
    colors = {item["class"]: item["color"] for item in HYBRID_COMPONENTS}
    result = compose_deterioration_masks(source, masks, colors)
    reverse = compose_deterioration_masks(source, dict(reversed(list(masks.items()))), colors)
    np.testing.assert_array_equal(result["mask"], reverse["mask"])
    for index, item in enumerate(HYBRID_COMPONENTS):
        assert result["mask"].getpixel((index, 0)) == item["color"]
        np.testing.assert_array_equal(masks[item["class"]], arrays[item["class"]])
    assert result["mask"].getpixel((23, 0)) == (0, 0, 0)
    assert result["overlay"].getpixel((23, 0)) == source.getpixel((23, 0))
    assert result["overlap_pixels"] == 48
    assert result["mask"].getpixel((0, 1)) == (255, 255, 255)
    assert result["mask"].getpixel((6, 1)) == (70, 70, 70)
    assert result["overlay"].getpixel((0, 1)) == (255, 255, 255)
    with pytest.raises(ValueError, match="match the source"):
        compose_deterioration_masks(source, {"crack": Image.new("L", (1, 1))}, {"crack": colors["crack"]})


def test_hybrid_uses_three_fixed_best_weights_and_independent_thresholds(tmp_path):
    manager, events = manager_for_test(tmp_path)
    thresholds = {"craquelure": .25, "crack": .6, "loss": .8}
    result = manager.predict_hybrid(Image.new("RGB", (12, 3)), thresholds=thresholds)
    assert [(item[1], item[2]) for item in events if item[0] == "predict"] == [
        (item["model"], thresholds[item["class"]]) for item in HYBRID_COMPONENTS
    ]
    assert [item[2] for item in events if item[0] == "load"] == ["best.pt"] * 3
    assert len([item for item in events if item[0] == "unload"]) == 2
    assert manager.cached_key == ("sam3_adapter_loss", "best.pt")
    assert result["mask"].mode == "RGB"
    assert result["thresholds"] == thresholds
    assert len(result["components"]) == 3
    assert result["metadata"]["overlap_pixels"] == 16
    assert [item["deterioration_class"] for item in result["components"]] == [
        item["class"] for item in HYBRID_COMPONENTS
    ]
    manager.unload()


@pytest.mark.parametrize("thresholds", [{"crack": float("nan")}, {"loss": 1.1}, {"unknown": .5}, {"loss": "bad"}])
def test_invalid_hybrid_thresholds_do_not_load_models(tmp_path, thresholds):
    manager, events = manager_for_test(tmp_path)
    with pytest.raises(ValueError, match="threshold"):
        manager.predict_hybrid(Image.new("RGB", (12, 3)), thresholds=thresholds)
    assert events == []


def test_hybrid_preflights_all_weights_before_inference(tmp_path):
    def resolver(model, weight, root):
        if model == "sam3_adapter_loss":
            raise InvalidWeightError("missing checkpoint")
        return root / model / weight
    manager, events = manager_for_test(tmp_path, resolver=resolver)
    with pytest.raises(HybridInferenceError, match="sam3_adapter_loss"):
        manager.predict_hybrid(Image.new("RGB", (12, 3)))
    assert events == []


def test_middle_component_failure_cleans_cache_and_does_not_run_last_model(tmp_path):
    manager, events = manager_for_test(tmp_path, fail="sam2_adapter_crack")
    with pytest.raises(HybridInferenceError, match="sam2_adapter_crack"):
        manager.predict_hybrid(Image.new("RGB", (12, 3)))
    assert manager.cached_key is None
    assert [item[1] for item in events if item[0] == "predict"] == [
        "sam3_adapter_craquelure", "sam2_adapter_crack",
    ]
    assert events[-1] == ("unload", "sam2_adapter_crack")


def test_other_requests_cannot_interleave_hybrid_components(tmp_path):
    entered, release, contender_started = threading.Event(), threading.Event(), threading.Event()
    manager, events = manager_for_test(tmp_path, gate=(entered, release))
    image = Image.new("RGB", (12, 3))
    def contender():
        contender_started.set()
        return manager.predict("sam3_adapter_loss", "best.pt", image, .9)
    with ThreadPoolExecutor(max_workers=2) as pool:
        hybrid = pool.submit(manager.predict_hybrid, image)
        assert entered.wait(5)
        other = pool.submit(contender)
        assert contender_started.wait(5)
        release.set()
        hybrid.result(timeout=10)
        other.result(timeout=10)
    assert [(item[1], item[2]) for item in events if item[0] == "predict"] == [
        (item["model"], .5) for item in HYBRID_COMPONENTS
    ] + [("sam3_adapter_loss", .9)]
    manager.unload()


def test_hybrid_weight_preset_requires_every_best_checkpoint(tmp_path):
    roots = {}
    for item in HYBRID_COMPONENTS:
        root = tmp_path / item["class"]
        root.mkdir()
        (root / "best.pt").touch()
        roots[item["model"]] = root
    assert get_weights(HYBRID_MODEL_ID, weight_roots=roots) == [HYBRID_WEIGHT]
    (roots["sam2_adapter_crack"] / "best.pt").unlink()
    assert get_weights(HYBRID_MODEL_ID, weight_roots=roots) == []


class LocalApiSession:
    def __init__(self, client):
        self.client, self.last_form = client, None

    def post(self, _url, *, files, data, **_kwargs):
        self.last_form = data
        filename, content, _mime = files["image"]
        response = self.client.post("/api/infer", data={
            **data, "image": (io.BytesIO(content), filename),
        }, content_type="multipart/form-data")
        # requests.Response interface used by the UI client.
        return SimpleNamespace(status_code=response.status_code, json=response.get_json)


def test_hybrid_flask_and_ui_round_trip_and_atomic_failure(tmp_path, monkeypatch):
    settings = replace(load_settings(), private_demo_mode=False)
    manager, _events = manager_for_test(tmp_path)
    app = create_app(settings=settings, inference_manager=manager)
    app.config.update(TESTING=True)
    session = LocalApiSession(app.test_client())
    client = InferenceApiClient("http://localhost", session=session)
    path = tmp_path / "wall.png"
    Image.new("RGB", (12, 3), "navy").save(path)
    thresholds = {"craquelure": .3, "crack": .6, "loss": .7}
    result = client.infer(path, HYBRID_MODEL_ID, HYBRID_WEIGHT, .5, HYBRID_CLASS_ID, thresholds)
    assert result["mask"].mode == "RGB"
    assert [item["mask"].mode for item in result["components"]] == ["L"] * 3
    assert all(item["weight"] == "best.pt" for item in result["components"])
    assert "重疊像素：16" in result["metadata"]
    assert session.last_form["threshold_loss"] == "0.7"
    for item in result["components"]:
        assert item["threshold"] == thresholds[item["class"]]

    demo = create_ui(settings=settings, api_client=client)
    functions = {fn.fn.__name__: fn.fn for fn in demo.fns.values()}
    output = functions["run_inference"](str(path), HYBRID_MODEL_ID, HYBRID_WEIGHT, HYBRID_CLASS_ID, .5, .3, .6, .7)
    assert len(output) == 13
    displayed = [output[0], output[1]["value"], output[2]["value"], *output[4:10]]
    for item in displayed:
        if isinstance(item, str):
            with Image.open(item) as rendered:
                assert rendered.size == (12, 3)
                assert rendered.format == "PNG"
        else:
            assert item.size == (12, 3)
    cached_views = output[10]
    assert set(cached_views) == {"all", "craquelure", "crack", "loss"}
    assert output[11]["value"] == "all"
    assert output[11]["interactive"] is True
    assert output[1]["label"] == "彩色遮罩"
    def forbid_inference(*_args, **_kwargs):
        raise AssertionError("Viewing a completed overlay must not rerun inference")
    monkeypatch.setattr(client, "infer", forbid_inference)
    for view in cached_views:
        selected, legend, selected_mask = functions["select_overlay_view"](view, cached_views)
        with Image.open(selected_mask["value"]) as mask:
            assert mask.size == (12, 3)
            assert mask.format == "PNG"
            if view == "all":
                assert mask.mode == "RGB"
                np.testing.assert_array_equal(mask, result["mask"])
                assert selected_mask["label"] == "彩色遮罩"
            else:
                expected = next(item for item in result["components"] if item["class"] == view)
                assert mask.mode == "L"
                assert set(np.unique(mask)) <= {0, 255}
                np.testing.assert_array_equal(mask, expected["mask"])
                assert selected_mask["label"] == f"{expected['label']}二值遮罩"
        with Image.open(selected["value"]) as rendered:
            assert rendered.size == (12, 3)
            if view != "all":
                index = [item["class"] for item in HYBRID_COMPONENTS].index(view)
                # Only the selected class's isolated foreground is colored.
                for other_index in range(3):
                    if other_index != index:
                        assert rendered.getpixel((other_index, 0)) == (0, 0, 128)
                assert rendered.getpixel((index, 0)) != (0, 0, 128)
                assert "重疊區域" not in legend["value"]
    assert functions["select_overlay_view"]("all", {})[0]["value"] is None
    assert functions["select_overlay_view"]("all", {})[2]["value"] is None
    mode = functions["update_model_mode"](HYBRID_MODEL_ID)
    assert mode[0]["visible"] is False
    assert all(item["visible"] for item in mode[1:4])
    assert mode[4]["label"] == "彩色遮罩"
    single_mode = functions["update_model_mode"]("sam3_adapter_loss")
    assert single_mode[0]["visible"] is True
    assert not any(item["visible"] for item in single_mode[1:4])
    assert single_mode[4]["label"] == "二值遮罩"
    assert single_mode[8:14] == (None,) * 6
    assert single_mode[14] == {}
    assert single_mode[15]["interactive"] is False
    assert single_mode[15]["value"] == "all"
    components = demo.get_config_file()["components"]
    images = [item["props"] for item in components if item["type"] == "image"]
    assert all(item["format"] == "png" for item in images if "遮罩" in item["label"] or "疊合圖" in item["label"])
    manager.unload()

    failed_manager, _events = manager_for_test(tmp_path, fail="sam2_adapter_crack")
    failed_app = create_app(settings=settings, inference_manager=failed_manager)
    failed_session = LocalApiSession(failed_app.test_client())
    failed_client = InferenceApiClient("http://localhost", session=failed_session)
    with pytest.raises(ApiClientError, match="sam2_adapter_crack"):
        failed_client.infer(path, HYBRID_MODEL_ID, HYBRID_WEIGHT, .5, HYBRID_CLASS_ID)
    bad_response = session.client.post("/api/infer", data={
        "image": (io.BytesIO(path.read_bytes()), "wall.png"), "model": HYBRID_MODEL_ID,
        "weight": HYBRID_WEIGHT, "threshold_loss": "nan",
    })
    assert bad_response.status_code == 400
    assert bad_response.get_json()["error"] == "Invalid threshold"
