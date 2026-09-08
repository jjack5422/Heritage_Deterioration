from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from adapters.da_sam3 import DASAM3Adapter
from config import Settings
from dual_adapter_sam3 import checkpoints as checkpoints_module
from dual_adapter_sam3 import model as model_module


class _TwoChannelModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed_min = float("nan")
        self.observed_max = float("nan")

    def forward(self, batch: torch.Tensor):
        self.observed_min = float(batch.min())
        self.observed_max = float(batch.max())
        height, width = batch.shape[-2:]
        crack = torch.full(
            (batch.shape[0], height, width),
            10.0,
            device=batch.device,
        )
        loss = torch.full_like(crack, -10.0)
        return SimpleNamespace(logits=torch.stack((crack, loss), dim=1))


def _settings() -> Settings:
    return Settings(
        model_root=Path("/data/models"),
        flask_host="127.0.0.1",
        flask_port=5000,
        gradio_host="127.0.0.1",
        gradio_port=7860,
        flask_api_url="http://127.0.0.1:5000",
        max_upload_mb=10,
        request_timeout_seconds=120.0,
        sam3_base_checkpoint=None,
        inference_tile_size=512,
        inference_stride=384,
        inference_batch_size=1,
        sam3_input_size=512,
    )


class _LoadModel(nn.Module):
    model_variant = "visual_da_sam3"

    def __init__(self) -> None:
        super().__init__()
        self.to_device: str | None = None

    def to(self, device):
        self.to_device = str(device)
        return self


def test_da_sam3_selects_requested_channel_and_uses_zero_one_input() -> None:
    adapter = DASAM3Adapter(settings=_settings(), device="cpu")
    model = _TwoChannelModel()
    adapter.model = model
    image = Image.new("RGB", (8, 6), "white")

    crack = adapter.predict(image, 0.5, "crack_craquelure")
    loss = adapter.predict(image, 0.5, "loss")

    assert np.all(crack["mask"] == 255)
    assert np.all(loss["mask"] == 0)
    assert model.observed_min == pytest.approx(0.0)
    assert model.observed_max == pytest.approx(1.0)
    assert crack["metadata"]["deterioration_class"] == "crack_craquelure"
    assert loss["metadata"]["deterioration_class"] == "loss"


def test_da_sam3_rejects_missing_or_unknown_class() -> None:
    adapter = DASAM3Adapter(settings=_settings(), device="cpu")
    adapter.model = _TwoChannelModel()
    batch = torch.zeros(1, 3, 512, 512)

    with pytest.raises(ValueError, match="Invalid deterioration class"):
        adapter._predict_batch_for_class(batch, None)
    with pytest.raises(ValueError, match="Invalid deterioration class"):
        adapter._predict_batch_for_class(batch, "other")


def test_da_sam3_load_moves_injected_modules_before_loading_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        _settings(),
        sam3_base_checkpoint=tmp_path / "sam3.pt",
    )
    model = _LoadModel()
    observed: dict[str, object] = {}

    def build_model(registry, **kwargs):
        observed["variant"] = kwargs["model_variant"]
        observed["base_checkpoint"] = kwargs["checkpoint"]
        observed["full_pixel_decoder"] = kwargs["full_pixel_decoder"]
        return model

    def load_checkpoint(path, loaded_model, **kwargs):
        observed["model_device"] = loaded_model.to_device
        observed["checkpoint"] = path
        observed.update(kwargs)
        return {"stage": "stage2", "epoch": 64}

    monkeypatch.setattr(model_module, "build_dual_adapter_model", build_model)
    monkeypatch.setattr(
        checkpoints_module,
        "load_adaptation_checkpoint",
        load_checkpoint,
    )
    checkpoint = tmp_path / "stage2_best.pt"
    adapter = DASAM3Adapter(settings=settings, device="cpu")

    adapter.load(checkpoint)

    assert observed["variant"] == "visual_da_sam3"
    assert observed["base_checkpoint"] == tmp_path / "sam3.pt"
    assert observed["full_pixel_decoder"] is True
    assert observed["model_device"] == "cpu"
    assert observed["checkpoint"] == checkpoint
    assert adapter.model is model
    assert adapter.load_metadata["checkpoint_stage"] == "stage2"
    assert adapter.load_metadata["checkpoint_epoch"] == 64
    assert adapter.load_metadata["full_pixel_decoder"] is True
