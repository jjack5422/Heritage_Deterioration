from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn

import scripts.evaluation.smoke_visual_da_sam3 as smoke
from dual_adapter_sam3.model import DualAdapterSam3Output


class _FakeSmokeModel(nn.Module):
    model_variant = "visual_da_sam3"

    def __init__(self) -> None:
        super().__init__()
        self.visual_adapter = nn.Module()
        self.visual_adapter.scale = nn.Parameter(torch.zeros(()))
        self.frozen_weight = nn.Parameter(torch.ones(()), requires_grad=False)
        self._stage = "stage1"

    def configure_stage(self, stage: str) -> tuple[str, ...]:
        self._stage = stage
        self.visual_adapter.scale.requires_grad_(stage == "stage1")
        self.frozen_weight.requires_grad_(False)
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def set_router_temperature(self, _temperature: float) -> None:
        return None

    def forward(self, images: torch.Tensor) -> DualAdapterSam3Output:
        base = images[:, :1] * (self.visual_adapter.scale + 0.1) * self.frozen_weight
        logits = base.repeat(1, 2, 1, 1)
        presence = logits.mean(dim=(2, 3))
        return DualAdapterSam3Output(logits, presence, (), 1)


def test_smoke_parser_locks_approved_defaults() -> None:
    args = smoke.build_parser().parse_args([])
    assert args.batch_size == 4
    assert args.steps == 2
    assert args.max_vram_gib == 24.0
    assert args.seed == 42
    assert not hasattr(args, "output")
    assert not hasattr(args, "run_dir")


def test_smoke_orchestration_runs_two_steps_without_stdout_or_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    built: list[str] = []
    model = _FakeSmokeModel()

    def fake_builder(_registry, *, model_variant: str, **_kwargs):
        built.append(model_variant)
        return model

    monkeypatch.setattr(smoke, "build_dual_adapter_model", fake_builder)
    args = smoke.build_parser().parse_args([])
    before = set(tmp_path.rglob("*"))
    summary = smoke.run_smoke(args, device=torch.device("cpu"))
    after = set(tmp_path.rglob("*"))

    assert built == ["visual_da_sam3"]
    assert summary["model_variant"] == "visual_da_sam3"
    assert summary["output_shape"] == [4, 2, 512, 512]
    assert summary["vision_forward_calls"] == [1, 1]
    assert len(summary["losses"]) == 2
    assert summary["visual_gradient_tensors"] == summary["visual_parameter_tensors"] == 1
    assert summary["frozen_gradient_tensors"] == 0
    assert capsys.readouterr().out == ""
    assert after == before


def test_smoke_script_has_no_reporting_or_output_path_dependency() -> None:
    source = inspect.getsource(smoke)
    assert "RunLayout" not in source
    assert "SummaryWriter" not in source
    assert "build_training_report" not in source
    assert "torch.save" not in source
