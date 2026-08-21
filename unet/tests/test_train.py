from __future__ import annotations

import json
import math
import sys
import csv
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from crackseg_common.augment import train_transforms
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = SRC.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC))

from crackseg_common.reporting.outputs import RunLayout  # noqa: E402
from crackseg_common.thresholding import ThresholdPolicy  # noqa: E402
from crackseg_common.training_runtime import (  # noqa: E402
    ExpertDataset,
    JointDataset,
    MergedForegroundDataset,
    SegLoss,
    TrainingRun,
    build_optimizer,
    build_scheduler,
    cost_weight_for_epoch,
    evaluate,
    prepare_dataset,
    restore_checkpoint,
    save_checkpoint,
    train_epoch,
    write_json,
)
from train import main, parser  # noqa: E402
from unet_model import resolve_model_args  # noqa: E402


def test_unet_train_entrypoint_stays_small() -> None:
    line_count = len((SRC / "train.py").read_text(encoding="utf-8").splitlines())

    assert line_count <= 40


def test_training_defaults_match_resnet50_finetuning_on_local_gpu() -> None:
    args = parser().parse_args(["--dataset-root", "dataset", "--expert", "crack"])
    resolve_model_args(args)

    assert args.backbone == "resnet50"
    assert args.architecture == "unet"
    assert args.encoder == "resnet50"
    assert args.decoder_channels is None
    assert args.batch_size == 32
    assert args.gradient_accumulation_steps == 1
    assert args.lr == pytest.approx(3e-4)
    assert args.encoder_lr_mult == pytest.approx(0.1)
    assert args.lr_factor == pytest.approx(0.5)
    assert args.lr_patience == 6
    assert args.lr_threshold == pytest.approx(0.005)
    assert args.lr_cooldown == 2
    assert args.epochs == 80
    assert args.early_stop_patience is None


def test_convnext_large_profile_uses_measured_safe_defaults() -> None:
    args = parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--expert",
            "crack_craquelure",
            "--backbone",
            "convnext_large",
        ]
    )

    resolve_model_args(args)

    assert args.backbone == "convnext_large"
    assert args.encoder == "tu-convnext_large"
    assert args.batch_size == 16
    assert args.gradient_accumulation_steps == 1
    assert args.lr == pytest.approx(2e-4)
    assert args.encoder_lr_mult == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("backbone", "encoder"),
    [
        ("convnext_tiny", "tu-convnext_tiny.fb_in22k_ft_in1k"),
        ("convnext_base", "tu-convnext_base.fb_in22k_ft_in1k"),
    ],
)
def test_convnext_tiny_and_base_profiles_match_large_training_protocol(
    backbone: str,
    encoder: str,
) -> None:
    args = parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--expert",
            "crack_craquelure",
            "--backbone",
            backbone,
        ]
    )

    resolve_model_args(args)

    assert args.backbone == backbone
    assert args.encoder == encoder
    assert args.batch_size == 16
    assert args.gradient_accumulation_steps == 1
    assert args.lr == pytest.approx(2e-4)
    assert args.encoder_lr_mult == pytest.approx(0.05)


def test_convnext_large_profile_preserves_explicit_training_overrides() -> None:
    args = parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--expert",
            "crack_craquelure",
            "--backbone",
            "convnext_large",
            "--batch-size",
            "8",
            "--lr",
            "0.0001",
            "--encoder-lr-mult",
            "0.02",
            "--gradient-accumulation-steps",
            "4",
        ]
    )

    resolve_model_args(args)

    assert args.encoder == "tu-convnext_large"
    assert args.batch_size == 8
    assert args.lr == pytest.approx(1e-4)
    assert args.encoder_lr_mult == pytest.approx(0.02)
    assert args.gradient_accumulation_steps == 4


def test_validate_only_creates_the_standard_run_layout_and_reproducibility_records(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    output = tmp_path / "run"

    assert main(
        [
            "--dataset-root",
            str(tmp_path),
            "--expert",
            "loss",
            "--output-dir",
            str(output),
            "--validate-only",
        ]
    ) == 0

    assert (output / "config" / "args.json").is_file()
    assert (output / "config" / "split_plan.json").is_file()
    assert (output / "metrics").is_dir()
    assert (output / "tensorboard").is_dir()


def test_optimizer_uses_lower_learning_rate_for_pretrained_encoder() -> None:
    model = torch.nn.Module()
    model.encoder = torch.nn.Conv2d(3, 4, 1)
    model.decoder = torch.nn.Conv2d(4, 2, 1)

    optimizer = build_optimizer(model, lr=3e-4, encoder_lr_mult=0.1, weight_decay=1e-4)
    learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}

    assert learning_rates == {"decoder": pytest.approx(3e-4), "encoder": pytest.approx(3e-5)}


def test_plateau_scheduler_reduces_both_learning_rates_and_preserves_ratio() -> None:
    model = torch.nn.Module()
    model.encoder = torch.nn.Conv2d(3, 4, 1)
    model.decoder = torch.nn.Conv2d(4, 2, 1)
    optimizer = build_optimizer(model, lr=3e-4, encoder_lr_mult=0.1, weight_decay=1e-4)
    scheduler = build_scheduler(
        optimizer,
        factor=0.5,
        patience=1,
        threshold=0.005,
        cooldown=0,
        min_lr=1e-6,
        min_encoder_lr=1e-7,
    )

    for loss in (1.0, 1.1, 1.1):
        scheduler.step(loss)

    learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert learning_rates == {"decoder": pytest.approx(1.5e-4), "encoder": pytest.approx(1.5e-5)}


def test_training_augmentation_is_mild_and_contains_no_noise_or_blur() -> None:
    transforms = train_transforms(512)
    names = [type(transform).__name__ for transform in transforms.transforms]

    assert names == [
        "LongestMaxSize",
        "PadIfNeeded",
        "HorizontalFlip",
        "VerticalFlip",
        "RandomRotate90",
        "CLAHE",
        "RandomBrightnessContrast",
        "Normalize",
        "ToTensorV2",
    ]
    clahe = transforms.transforms[5]
    assert clahe.clip_limit == (1.0, 2.0)
    assert clahe.tile_grid_size == (8, 8)
    assert clahe.p == pytest.approx(0.25)
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[50:100, 60:120] = 1
    mask[0, 0] = 255
    result = transforms(image=image, mask=mask)

    assert result["image"].shape == (3, 512, 512)
    assert result["mask"].shape == (512, 512)
    assert set(result["mask"].unique().tolist()) <= {0, 1, 255}


def write_fixture_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_dataset(root: Path) -> list[str]:
    names = [f"panel{letter}_R1_C01__y00000_x00000.png" for letter in "ABCD"]
    (root / "images").mkdir()
    (root / "masks").mkdir()
    masks = [
        np.array([[0, 1], [0, 1]], dtype=np.uint8),
        np.array([[0, 1], [2, 2]], dtype=np.uint8),
        np.array([[0, 1], [0, 1]], dtype=np.uint8),
        np.array([[0, 1], [2, 2]], dtype=np.uint8),
    ]
    for name, mask in zip(names, masks, strict=True):
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), "RGB").save(root / "images" / name)
        Image.fromarray(mask, "L").save(root / "masks" / name)

    write_fixture_json(
        root / "manifest.json",
        {
            "pair_count": 4,
            "training_eligible": True,
            "manifest_sha256": "dataset-hash",
            "image_manifest_sha256": "images-hash",
            "mask_manifest_sha256": "masks-hash",
            "mask_values": [0, 1, 2],
            "label_contract": {
                "class_ids": {"loss": 2, "background": 0, "crack": 1},
                "ignore_value": 255,
            },
        },
    )
    write_fixture_json(root / "tile_index.json", {"items": [{"tile": name} for name in names]})
    for fold, holdout in enumerate(names):
        train = [name for name in names if name != holdout]
        write_fixture_json(
            root / "splits" / f"fold{fold}.json",
            {
                "outer_fold": fold,
                "manifest_sha256": f"fold-{fold}-hash",
                "holdout_tiles": [holdout],
                "folds": [{"train": train, "val": []}],
                "data_contract": {
                    "eligible_tile_count": 4,
                    "class_names": ["background", "crack", "loss"],
                    "ignore_value": 255,
                    "task_image_manifest_sha256": "images-hash",
                    "task_mask_manifest_sha256": "masks-hash",
                },
            },
        )
    return names


def make_joint_dataset(root: Path) -> list[str]:
    names = [f"panel{letter}_R1_C01__y00000_x00000.png" for letter in "ABCDE"]
    (root / "images").mkdir()
    (root / "masks").mkdir()
    mask = np.array([[0, 1, 4], [2, 3, 255]], dtype=np.uint8)
    for name in names:
        Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8), "RGB").save(
            root / "images" / name
        )
        Image.fromarray(mask, "L").save(root / "masks" / name)
    class_names = ["background", "crack", "loss", "shrinkage", "craquelure"]
    write_fixture_json(
        root / "manifest.json",
        {
            "pair_count": 5,
            "training_eligible": True,
            "manifest_sha256": "joint-hash",
            "image_manifest_sha256": "images-hash",
            "mask_manifest_sha256": "masks-hash",
            "label_contract": {
                "class_ids": {name: index for index, name in enumerate(class_names)},
                "ignore_value": 255,
            },
        },
    )
    write_fixture_json(root / "tile_index.json", {"items": [{"tile": name} for name in names]})
    for fold, holdout in enumerate(names):
        write_fixture_json(
            root / "splits" / f"fold{fold}.json",
            {
                "outer_fold": fold,
                "holdout_tiles": [holdout],
                "folds": [{"train": [name for name in names if name != holdout]}],
                "data_contract": {
                    "class_names": class_names,
                    "ignore_value": 255,
                    "task_image_manifest_sha256": "images-hash",
                    "task_mask_manifest_sha256": "masks-hash",
                },
            },
        )
    return names


def test_prepare_dataset_builds_leak_free_binary_plan_for_one_expert(tmp_path: Path) -> None:
    names = make_dataset(tmp_path)

    plan = prepare_dataset(tmp_path, expert="loss", outer_fold=0)

    assert plan.source_class_names == ("background", "crack", "loss")
    assert plan.class_names == ("background", "loss")
    assert plan.expert_name == "loss"
    assert plan.expert_id == 2
    assert plan.inner_fold == 1
    assert plan.train == (names[2], names[3])
    assert plan.val == (names[1],)
    assert plan.test == (names[0],)
    assert plan.train_counts.tolist() == [6, 2]
    assert plan.val_counts.tolist() == [2, 2]
    assert not (set(plan.train) & set(plan.val))
    assert not (set(plan.train) & set(plan.test))


def test_prepare_dataset_builds_joint_three_class_plan_and_ignores_other_damage(
    tmp_path: Path,
) -> None:
    names = make_joint_dataset(tmp_path)

    plan = prepare_dataset(tmp_path, expert="crack_craquelure", outer_fold=0)

    assert plan.class_names == ("background", "crack", "craquelure")
    assert plan.source_class_ids == (1, 4)
    assert plan.inner_fold == 1
    assert plan.train == tuple(names[2:])
    assert plan.val == (names[1],)
    assert plan.test == (names[0],)
    assert plan.train_counts.tolist() == [3, 3, 3]
    assert plan.val_counts.tolist() == [1, 1, 1]


def test_prepare_dataset_rotates_validation_once_across_outer_folds(tmp_path: Path) -> None:
    names = make_joint_dataset(tmp_path)

    plans = [
        prepare_dataset(tmp_path, expert="crack_craquelure", outer_fold=outer_fold)
        for outer_fold in range(5)
    ]

    assert [plan.inner_fold for plan in plans] == [1, 2, 3, 4, 0]
    assert [plan.test for plan in plans] == [(names[index],) for index in range(5)]
    assert [plan.val for plan in plans] == [
        (names[(index + 1) % 5],) for index in range(5)
    ]
    assert len({frozenset(plan.val) for plan in plans}) == 5


def test_prepare_dataset_rejects_invalid_automatic_validation_instead_of_falling_back(
    tmp_path: Path,
) -> None:
    names = make_joint_dataset(tmp_path)
    mask_without_craquelure = np.array([[0, 1, 0], [2, 3, 255]], dtype=np.uint8)
    Image.fromarray(mask_without_craquelure, "L").save(tmp_path / "masks" / names[1])

    with pytest.raises(ValueError, match=r"validation fold 1.*--inner-fold"):
        prepare_dataset(tmp_path, expert="crack_craquelure", outer_fold=0)


def test_prepare_dataset_rejects_unknown_expert(tmp_path: Path) -> None:
    make_dataset(tmp_path)

    with pytest.raises(ValueError, match="未知 expert"):
        prepare_dataset(tmp_path, expert="flaking", outer_fold=0)


def test_prepare_dataset_rejects_split_without_positive_training_examples(tmp_path: Path) -> None:
    make_dataset(tmp_path)

    with pytest.raises(ValueError, match="loss.*正樣本"):
        prepare_dataset(tmp_path, expert="loss", outer_fold=1, inner_fold=3)


def test_prepare_dataset_rejects_missing_split_names(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    fold0 = json.loads((tmp_path / "splits" / "fold0.json").read_text())
    fold0["folds"][0]["train"][0] = "missing.png"
    write_fixture_json(tmp_path / "splits" / "fold0.json", fold0)

    with pytest.raises(ValueError, match="tile_index"):
        prepare_dataset(tmp_path, expert="loss", outer_fold=0)


class SourceDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        del index
        return {
            "image": torch.zeros(3, 2, 3),
            "mask": torch.tensor([[0, 1, 2], [255, 2, 1]]),
            "name": "panelA_R1_C01__y00000_x00000.png",
        }


def test_expert_dataset_maps_only_selected_damage_to_foreground() -> None:
    item = ExpertDataset(SourceDataset(), expert_id=2, ignore_value=255)[0]

    assert item["mask"].tolist() == [[0, 0, 1], [255, 1, 0]]
    assert item["name"] == "panelA_R1_C01__y00000_x00000.png"


def test_joint_dataset_maps_crack_and_craquelure_and_ignores_other_damage() -> None:
    item = JointDataset(SourceDataset(), crack_id=1, craquelure_id=2, ignore_value=255)[0]

    assert item["mask"].tolist() == [[0, 1, 2], [255, 2, 1]]


def test_merged_foreground_dataset_excludes_every_non_crack_damage_label() -> None:
    class MergedSourceDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            del index
            return {
                "image": torch.zeros(3, 2, 4),
                "mask": torch.tensor([[0, 1, 2, 3], [4, 5, 255, 1]]),
                "name": "panelA_R1_C01__y00000_x00000.png",
            }

    item = MergedForegroundDataset(
        MergedSourceDataset(), foreground_id=1, ignore_value=255
    )[0]

    assert item["mask"].tolist() == [[0, 1, 255, 255], [255, 255, 255, 1]]


def test_binary_loss_ignores_255_and_matches_weighted_cross_entropy() -> None:
    weights = torch.tensor([0.25, 2.0])
    criterion = SegLoss(2, ignore_value=255, class_weights=weights, dice_weight=0.0)
    logits = torch.tensor(
        [[[[2.0, 0.0], [0.0, 5.0]], [[0.0, 2.0], [2.0, 0.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1], [1, 255]]])

    loss, parts = criterion(logits, target)
    expected = torch.nn.functional.cross_entropy(logits, target, weight=weights, ignore_index=255)

    assert parts["ce"] == pytest.approx(float(expected.detach()), abs=1e-7)
    loss.backward()
    assert torch.all(logits.grad[:, :, 1, 1] == 0)


def test_directional_cost_penalizes_craquelure_probability_assigned_to_crack() -> None:
    target = torch.tensor([[[2]]])
    wrong_logits = torch.tensor([[[[0.0]], [[8.0]], [[0.0]]]])
    correct_logits = torch.tensor([[[[0.0]], [[0.0]], [[8.0]]]])
    criterion = SegLoss(
        3,
        ce_weight=0.4,
        dice_weight=0.4,
        cost_weight=0.2,
        craquelure_to_crack_cost=10.0,
    )

    _, wrong = criterion(wrong_logits, target)
    _, correct = criterion(correct_logits, target)

    assert wrong["cost"] > 0.99
    assert correct["cost"] < 0.01


def test_directional_cost_warmup_reaches_configured_weight_after_epoch_15() -> None:
    assert cost_weight_for_epoch(1, maximum=0.2) == 0.0
    assert cost_weight_for_epoch(5, maximum=0.2) == 0.0
    assert cost_weight_for_epoch(10, maximum=0.2) == pytest.approx(0.1)
    assert cost_weight_for_epoch(15, maximum=0.2) == pytest.approx(0.2)
    assert cost_weight_for_epoch(80, maximum=0.2) == pytest.approx(0.2)


class TinyExpertDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        image = torch.zeros(3, 8, 8)
        image[index] = 1
        mask = torch.zeros(8, 8, dtype=torch.long)
        mask[:, 2:4] = 1
        mask[0, 0] = 255
        return {
            "image": image,
            "mask": mask,
            "name": f"panel{index}_R1_C01__y00000_x00000.png",
        }


class TinyJointDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        mask = torch.zeros(6, 6, dtype=torch.long)
        mask[:, 1:3] = 1
        mask[:, 4:6] = 2
        image = torch.zeros(3, 6, 6)
        image[0] = mask == 1
        image[1] = mask == 2
        return {
            "image": image,
            "mask": mask,
            "name": f"jointPanel{index}_R1_C01__y00000_x00000.png",
        }


class ExactJointModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        background = ((images[:, 0] + images[:, 1]) == 0).float() * 8
        return torch.stack((background, images[:, 0] * 8, images[:, 1] * 8), dim=1)


def test_joint_panel_evaluation_reports_three_classes_and_directional_confusion() -> None:
    loader = torch.utils.data.DataLoader(TinyJointDataset(), batch_size=2)
    criterion = SegLoss(3, ignore_value=255, craquelure_to_crack_cost=10)

    metrics = evaluate(
        ExactJointModel(),
        loader,
        criterion,
        torch.device("cpu"),
        "crack_craquelure",
        ignore_value=255,
    )

    assert set(metrics["tile_micro"]["per_class"]) == {
        "background",
        "crack",
        "craquelure",
    }
    assert metrics["cross_confusion"]["craquelure_to_crack"]["count"] == 0
    assert metrics["cross_confusion"]["craquelure_to_crack"]["rate"] == 0.0
    assert metrics["tile_micro"]["miou"] == pytest.approx(1.0)
    assert metrics["inference_rule"] == "softmax_argmax"
    assert metrics["threshold"] is None


def test_cpu_train_and_expert_panel_evaluation_smoke() -> None:
    loader = torch.utils.data.DataLoader(TinyExpertDataset(), batch_size=2)
    model = torch.nn.Conv2d(3, 2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = SegLoss(2, ignore_value=255)

    train_metrics = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"), scaler=None)
    metrics = evaluate(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        "loss",
        ignore_value=255,
    )

    assert math.isfinite(train_metrics["loss"])
    assert math.isfinite(metrics["loss"])
    assert metrics["expert_panel_macro"]["panel_count"] == 2
    assert set(metrics["tile_micro"]["per_class"]) == {"background", "loss"}


def test_gradient_accumulation_matches_larger_batches_including_final_partial_group() -> None:
    samples = [
        {
            "image": torch.tensor([[[value]]], dtype=torch.float32),
            "mask": torch.tensor(value * 0.5, dtype=torch.float32),
        }
        for value in (1.0, 2.0, 3.0)
    ]

    class ScalarMSE(torch.nn.Module):
        def forward(
            self, logits: torch.Tensor, masks: torch.Tensor
        ) -> tuple[torch.Tensor, dict[str, float]]:
            loss = torch.nn.functional.mse_loss(logits[:, 0, 0, 0], masks)
            return loss, {"mse": float(loss.detach())}

    accumulated = torch.nn.Conv2d(1, 1, 1, bias=False)
    accumulated.weight.data.fill_(0.25)
    direct = torch.nn.Conv2d(1, 1, 1, bias=False)
    direct.load_state_dict(accumulated.state_dict())
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    direct_optimizer = torch.optim.SGD(direct.parameters(), lr=0.1)

    accumulated_metrics = train_epoch(
        accumulated,
        torch.utils.data.DataLoader(samples, batch_size=1, shuffle=False),
        accumulated_optimizer,
        ScalarMSE(),
        torch.device("cpu"),
        scaler=None,
        gradient_accumulation_steps=2,
    )
    direct_metrics = train_epoch(
        direct,
        torch.utils.data.DataLoader(samples, batch_size=2, shuffle=False),
        direct_optimizer,
        ScalarMSE(),
        torch.device("cpu"),
        scaler=None,
    )

    assert accumulated.weight.detach() == pytest.approx(direct.weight.detach())
    assert accumulated_metrics == pytest.approx(direct_metrics)


def test_one_epoch_run_writes_tensorboard_metrics_checkpoint_qualitative_images_and_dashboard(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    plan = prepare_dataset(tmp_path, expert="loss", outer_fold=0)
    loader = torch.utils.data.DataLoader(TinyExpertDataset(), batch_size=2)
    model = torch.nn.Conv2d(3, 2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optimizer.param_groups[0]["name"] = "decoder"
    layout = RunLayout.create(tmp_path / "run")
    run = TrainingRun(
        args=Namespace(resume=None, epochs=1, early_stop_patience=2, encoder="resnet18"),
        plan=plan,
        layout=layout,
        device=torch.device("cpu"),
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        model=model,
        criterion=SegLoss(2, ignore_value=255),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min"),
        scaler=None,
    )

    run.execute()

    assert (layout.metrics / "epochs.csv").is_file()
    assert (layout.metrics / "tensorboard_scalars.csv").is_file()
    assert list(layout.tensorboard.glob("events.out.tfevents.*"))
    assert list((layout.tensorboard / "images" / "best").glob("*.png"))
    assert list((layout.tensorboard / "images" / "worst").glob("*.png"))
    assert (layout.tensorboard / "images" / "manifest.csv").is_file()
    assert (layout.tensorboard / "images" / "loss_curve.png").is_file()
    assert (layout.tensorboard / "images" / "pr_roc_curve.png").is_file()
    assert (layout.metrics / "outer_test_metrics.json").is_file()
    policy_path = layout.config / "threshold_policy.json"
    assert policy_path.is_file()
    assert (layout.metrics / "threshold_metrics.csv").is_file()
    policy = ThresholdPolicy.load(policy_path)
    outer_test = json.loads((layout.metrics / "outer_test_metrics.json").read_text())
    assert outer_test["threshold"] == pytest.approx(policy.threshold)
    assert policy.source == "validation_calibrated"
    assert (layout.checkpoints / "best.pt").is_file()
    assert (layout.checkpoints / "last.pt").is_file()
    assert len(list(layout.qualitative.glob("*/overlay.png"))) == len(TinyExpertDataset())
    assert (layout.reports / "index.html").is_file()
    assert "epoch=1" in (layout.logs / "train.log").read_text(encoding="utf-8")
    accumulator = EventAccumulator(str(layout.tensorboard))
    accumulator.Reload()
    image_tags = set(accumulator.Tags()["images"])
    assert any(tag.startswith("qualitative/best/") for tag in image_tags)
    assert any(tag.startswith("qualitative/worst/") for tag in image_tags)
    assert "threshold/pr_roc_curve" in image_tags


def test_joint_run_writes_best_last_comparison_without_touching_outer_test(
    tmp_path: Path,
) -> None:
    make_joint_dataset(tmp_path)
    plan = prepare_dataset(tmp_path, expert="crack_craquelure", outer_fold=0)
    loader = torch.utils.data.DataLoader(TinyJointDataset(), batch_size=2)
    model = torch.nn.Conv2d(3, 3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optimizer.param_groups[0]["name"] = "decoder"
    layout = RunLayout.create(tmp_path / "joint-run")
    run = TrainingRun(
        args=Namespace(
            resume=None,
            epochs=2,
            early_stop_patience=None,
            encoder="resnet18",
            cost_weight=0.2,
            craquelure_to_crack_cost=10.0,
            skip_outer_test=True,
        ),
        plan=plan,
        layout=layout,
        device=torch.device("cpu"),
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        model=model,
        criterion=SegLoss(3, ignore_value=255, craquelure_to_crack_cost=10.0),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min"),
        scaler=None,
    )

    run.execute()

    rows = list(csv.DictReader((layout.metrics / "epochs.csv").open()))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert (layout.metrics / "checkpoint_comparison.json").is_file()
    assert not (layout.metrics / "outer_test_metrics.json").exists()
    assert len(list(layout.qualitative.glob("*/overlay.png"))) == 2
    assert (layout.tensorboard / "images" / "loss_curve.png").is_file()
    assert (layout.reports / "index.html").is_file()


class UnevenPanelDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict:
        target = 0 if index < 2 else 1
        panel = "panelA" if index < 2 else "panelB"
        return {
            "image": torch.zeros(3, 2, 2),
            "mask": torch.full((2, 2), target, dtype=torch.long),
            "name": f"{panel}_R1_C0{index + 1}__y00000_x00000.png",
        }


def test_panel_macro_loss_is_equal_weighted_and_batch_independent() -> None:
    model = torch.nn.Conv2d(3, 2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([1.0, -1.0]))
    criterion = SegLoss(2, ce_weight=1.0, dice_weight=0.0)

    metrics_by_batch_size = [
        evaluate(
            model,
            torch.utils.data.DataLoader(UnevenPanelDataset(), batch_size=batch_size),
            criterion,
            torch.device("cpu"),
            "crack",
            ignore_value=255,
        )
        for batch_size in (1, 2, 3)
    ]

    expected_panel_macro = (
        torch.nn.functional.softplus(torch.tensor(-2.0))
        + torch.nn.functional.softplus(torch.tensor(2.0))
    ) / 2
    assert [metrics["panel_macro_loss"] for metrics in metrics_by_batch_size] == pytest.approx(
        [float(expected_panel_macro)] * 3,
    )


class NegativeOnlyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        del index
        return {
            "image": torch.zeros(3, 4, 4),
            "mask": torch.zeros(4, 4, dtype=torch.long),
            "name": "panelN_R1_C01__y00000_x00000.png",
        }


def test_evaluation_records_null_instead_of_perfect_iou_for_empty_expert_test(
    tmp_path: Path,
) -> None:
    loader = torch.utils.data.DataLoader(NegativeOnlyDataset())
    model = torch.nn.Conv2d(3, 2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([1.0, 0.0]))

    metrics = evaluate(
        model,
        loader,
        SegLoss(2),
        torch.device("cpu"),
        "flaking",
        ignore_value=255,
    )
    output = tmp_path / "metrics.json"
    write_json(output, metrics)
    saved = json.loads(output.read_text())

    assert metrics["expert_panel_macro"]["iou"] is None
    assert metrics["expert_panel_macro"]["positive_panel_count"] == 0
    assert saved["tile_micro"]["miou"] is None


def test_checkpoint_round_trip_is_bound_to_one_expert(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    plan = prepare_dataset(tmp_path, expert="loss", outer_fold=0)
    model = torch.nn.Conv2d(3, 2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    expected = [parameter.detach().clone() for parameter in model.parameters()]
    path = tmp_path / "last.pt"
    args = Namespace(encoder="resnet18")

    save_checkpoint(
        path,
        epoch=3,
        model=model,
        optimizer=optimizer,
        scaler=None,
        scheduler=scheduler,
        best=0.2,
        bad_epochs=2,
        args=args,
        plan=plan,
        val={"miou": 0.2},
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    epoch, best, bad_epochs = restore_checkpoint(
        path,
        model,
        optimizer,
        scaler=None,
        scheduler=scheduler,
        plan=plan,
    )
    payload = torch.load(path, weights_only=False)

    assert epoch == 3
    assert best == pytest.approx(0.2)
    assert bad_epochs == 2
    assert payload["selection_metric"] == "val_panel_macro_loss"
    assert payload["selection_mode"] == "min"
    assert payload["args"]["class_names"] == "background,loss"
    assert payload["source_class_names"] == ["background", "crack", "loss"]
    assert payload["expert"] == {"name": "loss", "source_class_id": 2}
    for actual, wanted in zip(model.parameters(), expected, strict=True):
        assert torch.equal(actual, wanted)

    with pytest.raises(ValueError, match="expert"):
        wrong_plan = prepare_dataset(tmp_path, expert="crack", outer_fold=0)
        restore_checkpoint(
            path,
            model,
            optimizer,
            scaler=None,
            scheduler=scheduler,
            plan=wrong_plan,
        )
