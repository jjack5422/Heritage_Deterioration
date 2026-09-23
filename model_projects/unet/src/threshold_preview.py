"""Local desktop threshold preview backed by a cached probability map."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from crackseg_common.thresholding import (
    INFERENCE_RULE,
    ThresholdPolicy,
    apply_threshold,
    load_foreground_probability,
)


MASK_COLOUR = np.array((255, 55, 80), dtype=np.uint8)
OVERLAY_COLOUR = np.array((56, 189, 248), dtype=np.float32)
_PICKER_APPLICATION = None
EXPERT_SOURCE_IDS = {"crack": 1, "craquelure": 4}


@dataclass(frozen=True)
class GalleryEntry:
    """One expert/fold checkpoint and its exported validation images."""

    experiment: str
    expert: str
    fold: int
    checkpoint: Path
    images: tuple[Path, ...]
    policy: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.experiment} / {self.expert} / fold{self.fold}"


@dataclass
class GalleryState:
    """Current validation image plus probabilities cached for one checkpoint."""

    images: tuple[Path, ...]
    index: int = 0
    probability_cache: dict[Path, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.images:
            raise ValueError("gallery 至少需要一張 validation 圖片")
        self.index %= len(self.images)

    @property
    def current(self) -> Path:
        return self.images[self.index]

    def move(self, delta: int | None) -> Path:
        if delta is not None:
            self.index = (self.index + delta) % len(self.images)
        return self.current

    def probability(self, loader: Callable[[Path], np.ndarray]) -> np.ndarray:
        if self.current not in self.probability_cache:
            self.probability_cache[self.current] = load_foreground_probability(
                loader(self.current)
            )
        return self.probability_cache[self.current]


def navigation_delta(key: str) -> int | None:
    """Map gallery shortcuts: D is previous and F is next."""

    return {"D": -1, "F": 1}.get(key.upper())


def discover_gallery_entries(project_root: str | Path) -> list[GalleryEntry]:
    """Find trained binary experts and validation images under project ``runs``."""

    runs_dir = Path(project_root) / "runs"
    entries: list[GalleryEntry] = []
    pattern = "*/5fold/*/fold*/artifacts/checkpoints/best.pt"
    for checkpoint in runs_dir.glob(pattern):
        relative = checkpoint.relative_to(runs_dir)
        experiment, _, expert, fold_name = relative.parts[:4]
        if expert not in EXPERT_SOURCE_IDS or not fold_name.startswith("fold"):
            continue
        try:
            fold = int(fold_name.removeprefix("fold"))
        except ValueError:
            continue
        fold_dir = checkpoint.parents[2]
        images = tuple(
            sorted(
                (fold_dir / "artifacts" / "qualitative").glob("*/input.png"),
                key=lambda path: path.parent.name,
            )
        )
        if not images:
            continue
        policy = fold_dir / "config" / "threshold_policy.json"
        entries.append(
            GalleryEntry(
                experiment=experiment,
                expert=expert,
                fold=fold,
                checkpoint=checkpoint,
                images=images,
                policy=policy if policy.is_file() else None,
            )
        )
    expert_order = {"crack": 0, "craquelure": 1}
    return sorted(
        entries,
        key=lambda entry: (
            expert_order[entry.expert],
            entry.experiment,
            entry.fold,
        ),
    )


def probability_heatmap(probability: np.ndarray) -> np.ndarray:
    """Render a compact blue-green-red probability map without matplotlib."""

    value = load_foreground_probability(probability)
    red = np.clip(2 * value - 0.5, 0, 1)
    green = np.clip(1.5 - np.abs(2 * value - 1), 0, 1)
    blue = np.clip(1.5 - 2 * value, 0, 1)
    return (np.stack((red, green, blue), axis=-1) * 255).round().astype(np.uint8)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    rendered = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rendered[mask == 1] = MASK_COLOUR
    return rendered


def overlay_rgb(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask == 1] = (
        result[mask == 1] * (1 - alpha) + OVERLAY_COLOUR * alpha
    )
    return result.round().astype(np.uint8)


def build_preview_panels(
    image: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    probability = load_foreground_probability(probability)
    image = _validate_rgb(image)
    if image.shape[:2] != probability.shape:
        raise ValueError(f"image/probability 尺寸不同: {image.shape[:2]} / {probability.shape}")
    mask = apply_threshold(probability, threshold)
    return {
        "original": image,
        "probability": probability_heatmap(probability),
        "mask": mask_rgb(mask),
        "overlay": overlay_rgb(image, mask),
    }


def export_threshold_outputs(
    *,
    output_dir: str | Path,
    stem: str,
    image: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    policy: ThresholdPolicy,
) -> dict[str, Path]:
    """Export full-resolution outputs and the exact policy used to create them."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probability = load_foreground_probability(probability)
    image = _validate_rgb(image)
    if image.shape[:2] != probability.shape:
        raise ValueError(f"image/probability 尺寸不同: {image.shape[:2]} / {probability.shape}")
    selected_policy = policy.with_manual_threshold(threshold)
    mask = apply_threshold(probability, threshold)
    paths = {
        "mask": output_dir / f"{stem}_mask.png",
        "overlay": output_dir / f"{stem}_overlay.png",
        "metadata": output_dir / f"{stem}_threshold.json",
        "policy": output_dir / "threshold_policy.json",
    }
    Image.fromarray(mask, "L").save(paths["mask"])
    Image.fromarray(overlay_rgb(image, mask), "RGB").save(paths["overlay"])
    metadata = {
        "image": stem,
        "expert": selected_policy.expert,
        "threshold": float(threshold),
        "source": selected_policy.source,
        "inference_rule": INFERENCE_RULE,
        "mask": paths["mask"].name,
        "overlay": paths["overlay"].name,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_policy.save(paths["policy"])
    return paths


def score_preview(
    probability: np.ndarray,
    target: np.ndarray | None,
    threshold: float,
    foreground_id: int,
    ignore_value: int = 255,
) -> dict[str, float | int | None]:
    prediction = apply_threshold(probability, threshold).astype(bool)
    if target is None:
        return {
            "coverage": float(prediction.mean()),
            "predicted_pixels": int(prediction.sum()),
        }
    target = np.asarray(target)
    if target.shape != prediction.shape:
        raise ValueError(f"target/probability 尺寸不同: {target.shape} / {prediction.shape}")
    valid = target != ignore_value
    positive = target == foreground_id
    tp = int(np.count_nonzero(valid & positive & prediction))
    fp = int(np.count_nonzero(valid & ~positive & prediction))
    fn = int(np.count_nonzero(valid & positive & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image 必須是 HxWx3 RGB，收到 {array.shape}")
    return array.astype(np.uint8, copy=False)


def _preview_size(height: int, width: int, maximum_side: int = 720) -> tuple[int, int]:
    scale = min(1.0, maximum_side / max(height, width))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _resize_cached_preview(
    image: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = probability.shape
    size = _preview_size(height, width)
    if size == (width, height):
        return image, probability
    rgb = np.asarray(Image.fromarray(image, "RGB").resize(size, Image.Resampling.LANCZOS))
    probability_image = Image.fromarray(probability.astype(np.float32), mode="F")
    resized_probability = np.asarray(
        probability_image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32
    )
    return rgb, resized_probability


def _format_metrics(metrics: dict[str, Any]) -> str:
    if "f1" in metrics:
        return (
            f"Precision {metrics['precision']:.3f}   Recall {metrics['recall']:.3f}   "
            f"F1 {metrics['f1']:.3f}   IoU {metrics['iou']:.3f}   "
            f"FP {metrics['fp']:,}"
        )
    return (
        f"Detected area {metrics['coverage'] * 100:.2f}%   "
        f"Predicted pixels {metrics['predicted_pixels']:,}"
    )


class GalleryInference:
    """Keep one expert model loaded while gallery probabilities are generated."""

    def __init__(self, device: str | None = None) -> None:
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._checkpoint: Path | None = None
        self._model: Any = None
        self._policies: dict[Path, ThresholdPolicy] = {}

    def policy_for(self, entry: GalleryEntry) -> ThresholdPolicy:
        if entry.checkpoint in self._policies:
            return self._policies[entry.checkpoint]
        if entry.policy is not None:
            policy = ThresholdPolicy.load(entry.policy)
        else:
            policy = ThresholdPolicy.default(
                entry.expert,
                EXPERT_SOURCE_IDS[entry.expert],
            )
        self._policies[entry.checkpoint] = policy
        return policy

    def probability(self, entry: GalleryEntry, image_path: Path) -> np.ndarray:
        from predict_full import load_image_rgb, load_model_from_ckpt, predict_full

        if self._checkpoint != entry.checkpoint:
            self._model, payload = load_model_from_ckpt(entry.checkpoint, self.device)
            self._checkpoint = entry.checkpoint
            checkpoint_expert = payload.get("expert", {})
            expert = str(checkpoint_expert.get("name", entry.expert))
            foreground_id = int(
                checkpoint_expert.get(
                    "source_class_id",
                    EXPERT_SOURCE_IDS[entry.expert],
                )
            )
            if expert != entry.expert:
                raise ValueError(
                    f"checkpoint expert 不符: 路徑={entry.expert}, checkpoint={expert}"
                )
            policy = self.policy_for(entry)
            if policy.expert != expert or policy.foreground_id != foreground_id:
                raise ValueError(
                    "threshold policy 與 checkpoint 不符: "
                    f"{policy.expert}/{policy.foreground_id} != {expert}/{foreground_id}"
                )
        image = load_image_rgb(image_path)
        probability = predict_full(
            self._model,
            image,
            self.device,
            tile=512,
            stride=384,
            batch_size=4,
            use_amp=True,
        )
        if probability.ndim != 3 or probability.shape[0] != 2:
            raise ValueError(
                "threshold gallery 只支援 background + 單一劣化的二元 expert"
            )
        return probability[1]


def launch_viewer(
    *,
    image: np.ndarray,
    probability: np.ndarray,
    policy: ThresholdPolicy,
    output_dir: Path,
    stem: str,
    target: np.ndarray | None = None,
) -> int:
    """Launch Qt only here, so calibration/CLI inference do not require GUI imports."""

    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QDoubleSpinBox,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError(
            "尚未安裝 PySide6；請執行 uv pip install --python ../crackseg_env/bin/python "
            "-r requirements-gui.txt"
        ) from exc

    image = _validate_rgb(image)
    probability = load_foreground_probability(probability)
    if image.shape[:2] != probability.shape:
        raise ValueError(f"image/probability 尺寸不同: {image.shape[:2]} / {probability.shape}")
    preview_image, preview_probability = _resize_cached_preview(image, probability)

    class ThresholdWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"{policy.expert} threshold preview — {stem}")
            self.image_labels: dict[str, QLabel] = {}
            root = QWidget()
            layout = QVBoxLayout(root)
            grid = QGridLayout()
            titles = {
                "original": "1. Original",
                "probability": "2. Foreground probability",
                "mask": "3. Binary mask",
                "overlay": "4. Overlay",
            }
            for index, (name, title) in enumerate(titles.items()):
                cell = QVBoxLayout()
                heading = QLabel(title)
                heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label = QLabel()
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(320, 220)
                label.setStyleSheet("background: #171717; border: 1px solid #444;")
                self.image_labels[name] = label
                cell.addWidget(heading)
                cell.addWidget(label)
                grid.addLayout(cell, index // 2, index % 2)
            layout.addLayout(grid, stretch=1)

            controls = QHBoxLayout()
            controls.addWidget(QLabel("Threshold"))
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(5, 95)
            self.slider.setSingleStep(1)
            self.slider.setTickInterval(5)
            self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.spin = QDoubleSpinBox()
            self.spin.setRange(0.05, 0.95)
            self.spin.setDecimals(2)
            self.spin.setSingleStep(0.01)
            controls.addWidget(self.slider, stretch=1)
            controls.addWidget(self.spin)
            for name, label in (("sensitive", "Sensitive"), ("balanced", "Balanced"), ("clean", "Clean")):
                button = QPushButton(label)
                button.clicked.connect(
                    lambda checked=False, preset=name: self.set_threshold(
                        policy.presets[preset].threshold
                    )
                )
                controls.addWidget(button)
            reset = QPushButton("Reset 0.50")
            reset.clicked.connect(lambda: self.set_threshold(0.5))
            controls.addWidget(reset)
            save = QPushButton("Export")
            save.clicked.connect(self.export)
            controls.addWidget(save)
            layout.addLayout(controls)
            self.metrics = QLabel()
            self.metrics.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.metrics)
            if policy.source == "validation_calibrated":
                summary = "   ".join(
                    f"{name.title()} {item.threshold:.2f} (P {item.precision:.2f} / R {item.recall:.2f})"
                    for name, item in policy.presets.items()
                )
                validation = QLabel(f"Validation presets: {summary}")
                validation.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(validation)
            self.setCentralWidget(root)

            self.timer = QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.setInterval(75)
            self.timer.timeout.connect(self.render)
            self.slider.valueChanged.connect(self._slider_changed)
            self.spin.valueChanged.connect(self._spin_changed)
            self.set_threshold(policy.threshold)
            self.resize(1400, 900)

        def set_threshold(self, threshold: float) -> None:
            value = min(0.95, max(0.05, round(float(threshold), 2)))
            self.slider.blockSignals(True)
            self.spin.blockSignals(True)
            self.slider.setValue(round(value * 100))
            self.spin.setValue(value)
            self.slider.blockSignals(False)
            self.spin.blockSignals(False)
            self.render()

        def _slider_changed(self, value: int) -> None:
            self.spin.blockSignals(True)
            self.spin.setValue(value / 100)
            self.spin.blockSignals(False)
            self.timer.start()

        def _spin_changed(self, value: float) -> None:
            self.slider.blockSignals(True)
            self.slider.setValue(round(value * 100))
            self.slider.blockSignals(False)
            self.timer.start()

        def render(self) -> None:
            threshold = self.spin.value()
            panels = build_preview_panels(preview_image, preview_probability, threshold)
            for name, panel in panels.items():
                contiguous = np.ascontiguousarray(panel)
                qimage = QImage(
                    contiguous.data,
                    contiguous.shape[1],
                    contiguous.shape[0],
                    contiguous.strides[0],
                    QImage.Format.Format_RGB888,
                ).copy()
                self.image_labels[name].setPixmap(QPixmap.fromImage(qimage))
            values = score_preview(
                probability, target, threshold, policy.foreground_id
            )
            self.metrics.setText(f"threshold = {threshold:.2f}   {_format_metrics(values)}")

        def export(self) -> None:
            paths = export_threshold_outputs(
                output_dir=output_dir,
                stem=stem,
                image=image,
                probability=probability,
                threshold=self.spin.value(),
                policy=policy,
            )
            QMessageBox.information(
                self,
                "Export complete",
                f"Saved mask, overlay and policy to:\n{paths['mask'].parent}",
            )

    application = QApplication.instance() or QApplication(sys.argv)
    window = ThresholdWindow()
    window.show()
    return application.exec()


def launch_gallery(
    project_root: str | Path,
    *,
    entries: list[GalleryEntry] | None = None,
    inference: GalleryInference | None = None,
) -> int:
    """Open the no-argument expert/fold validation threshold browser."""

    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QDoubleSpinBox,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError(
            "尚未安裝 PySide6；請執行 uv pip install --python ../crackseg_env/bin/python "
            "-r requirements-gui.txt"
        ) from exc

    project_root = Path(project_root).resolve()
    gallery_entries = entries or discover_gallery_entries(project_root)
    if not gallery_entries:
        raise FileNotFoundError(
            f"找不到可瀏覽的 best.pt 與 validation 圖片: {project_root / 'runs'}"
        )
    inference_session = inference or GalleryInference()

    class GalleryWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.entries = gallery_entries
            self.states: dict[Path, GalleryState] = {}
            self.image_labels: dict[str, QLabel] = {}
            self.full_image: np.ndarray | None = None
            self.full_probability: np.ndarray | None = None
            self.preview_image: np.ndarray | None = None
            self.preview_probability: np.ndarray | None = None
            self.policy: ThresholdPolicy | None = None

            self.setWindowTitle("Crack / Craquelure threshold browser")
            root = QWidget()
            layout = QVBoxLayout(root)
            layout.addLayout(self._build_navigation())
            layout.addLayout(self._build_panels(), stretch=1)
            layout.addLayout(self._build_threshold_controls())
            self.metrics = QLabel()
            self.metrics.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.metrics)
            self.status = QLabel("D：上一張    F：下一張")
            self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.status)
            self.setCentralWidget(root)

            self.timer = QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.setInterval(50)
            self.timer.timeout.connect(self.render)
            self.slider.valueChanged.connect(self._slider_changed)
            self.spin.valueChanged.connect(self._spin_changed)

            self.previous_shortcut = QShortcut(QKeySequence("D"), self)
            self.previous_shortcut.activated.connect(lambda: self.move_image(-1))
            self.next_shortcut = QShortcut(QKeySequence("F"), self)
            self.next_shortcut.activated.connect(lambda: self.move_image(1))

            self.model.currentIndexChanged.connect(self._entry_changed)
            self.resize(1400, 940)
            self.load_current(reset_threshold=True)

        @property
        def entry(self) -> GalleryEntry:
            return self.entries[self.model.currentIndex()]

        @property
        def state(self) -> GalleryState:
            if self.entry.checkpoint not in self.states:
                self.states[self.entry.checkpoint] = GalleryState(self.entry.images)
            return self.states[self.entry.checkpoint]

        def _build_navigation(self) -> QHBoxLayout:
            controls = QHBoxLayout()
            controls.addWidget(QLabel("Model"))
            self.model = QComboBox()
            for entry in self.entries:
                self.model.addItem(entry.label)
            controls.addWidget(self.model, stretch=1)
            previous = QPushButton("D  上一張")
            previous.clicked.connect(lambda: self.move_image(-1))
            controls.addWidget(previous)
            following = QPushButton("F  下一張")
            following.clicked.connect(lambda: self.move_image(1))
            controls.addWidget(following)
            self.image_position = QLabel()
            controls.addWidget(self.image_position)
            return controls

        def _build_panels(self) -> QGridLayout:
            grid = QGridLayout()
            titles = {
                "original": "1. Original",
                "probability": "2. Foreground probability",
                "mask": "3. Binary mask",
                "overlay": "4. Overlay",
            }
            for index, (name, title) in enumerate(titles.items()):
                cell = QVBoxLayout()
                heading = QLabel(title)
                heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label = QLabel()
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(320, 220)
                label.setStyleSheet("background: #171717; border: 1px solid #444;")
                self.image_labels[name] = label
                cell.addWidget(heading)
                cell.addWidget(label)
                grid.addLayout(cell, index // 2, index % 2)
            return grid

        def _build_threshold_controls(self) -> QHBoxLayout:
            controls = QHBoxLayout()
            controls.addWidget(QLabel("Threshold"))
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(5, 95)
            self.slider.setSingleStep(1)
            self.slider.setTickInterval(5)
            self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            controls.addWidget(self.slider, stretch=1)
            self.spin = QDoubleSpinBox()
            self.spin.setRange(0.05, 0.95)
            self.spin.setDecimals(2)
            self.spin.setSingleStep(0.01)
            controls.addWidget(self.spin)
            for name, label in (
                ("sensitive", "Sensitive"),
                ("balanced", "Balanced"),
                ("clean", "Clean"),
            ):
                button = QPushButton(label)
                button.clicked.connect(
                    lambda checked=False, preset=name: self.apply_preset(preset)
                )
                controls.addWidget(button)
            reset = QPushButton("Reset 0.50")
            reset.clicked.connect(lambda: self.set_threshold(0.5))
            controls.addWidget(reset)
            export = QPushButton("Export")
            export.clicked.connect(self.export)
            controls.addWidget(export)
            return controls

        def _entry_changed(self, index: int) -> None:
            if index >= 0:
                self.load_current(reset_threshold=True)

        def move_image(self, delta: int) -> None:
            self.state.move(delta)
            self.load_current(reset_threshold=False)

        def load_current(self, *, reset_threshold: bool) -> None:
            image_path = self.state.current
            was_cached = image_path in self.state.probability_cache
            self.status.setText(
                f"{'讀取快取' if was_cached else '模型推論中'}：{image_path.parent.name}"
            )
            QApplication.processEvents()
            try:
                with Image.open(image_path) as source:
                    image = np.asarray(source.convert("RGB"))
                probability = self.state.probability(
                    lambda path: inference_session.probability(self.entry, path)
                )
            except Exception as error:
                self.status.setText(f"載入失敗：{error}")
                QMessageBox.critical(self, "Threshold preview error", str(error))
                return
            self.full_image = image
            self.full_probability = probability
            self.preview_image, self.preview_probability = _resize_cached_preview(
                image,
                probability,
            )
            self.policy = inference_session.policy_for(self.entry)
            self.image_position.setText(
                f"{self.state.index + 1} / {len(self.state.images)}   "
                f"{image_path.parent.name}"
            )
            if reset_threshold:
                self.set_threshold(self.policy.threshold)
            else:
                self.render()
            source = "cache" if was_cached else f"{inference_session.device} inference"
            self.status.setText(
                f"{source} 完成    D：上一張    F：下一張    "
                "拉桿不會重新執行模型"
            )

        def set_threshold(self, threshold: float) -> None:
            value = min(0.95, max(0.05, round(float(threshold), 2)))
            self.slider.blockSignals(True)
            self.spin.blockSignals(True)
            self.slider.setValue(round(value * 100))
            self.spin.setValue(value)
            self.slider.blockSignals(False)
            self.spin.blockSignals(False)
            self.render()

        def apply_preset(self, preset: str) -> None:
            if self.policy is not None:
                self.set_threshold(self.policy.presets[preset].threshold)

        def _slider_changed(self, value: int) -> None:
            self.spin.blockSignals(True)
            self.spin.setValue(value / 100)
            self.spin.blockSignals(False)
            self.timer.start()

        def _spin_changed(self, value: float) -> None:
            self.slider.blockSignals(True)
            self.slider.setValue(round(value * 100))
            self.slider.blockSignals(False)
            self.timer.start()

        def render(self) -> None:
            if self.preview_image is None or self.preview_probability is None:
                return
            threshold = self.spin.value()
            panels = build_preview_panels(
                self.preview_image,
                self.preview_probability,
                threshold,
            )
            for name, panel in panels.items():
                contiguous = np.ascontiguousarray(panel)
                qimage = QImage(
                    contiguous.data,
                    contiguous.shape[1],
                    contiguous.shape[0],
                    contiguous.strides[0],
                    QImage.Format.Format_RGB888,
                ).copy()
                self.image_labels[name].setPixmap(QPixmap.fromImage(qimage))
            values = score_preview(
                self.full_probability,
                None,
                threshold,
                self.policy.foreground_id,
            )
            self.metrics.setText(
                f"threshold = {threshold:.2f}   {_format_metrics(values)}"
            )

        def export(self) -> None:
            if self.full_image is None or self.full_probability is None:
                return
            output_dir = (
                project_root
                / "outputs"
                / "threshold_preview"
                / self.entry.experiment
                / self.entry.expert
                / f"fold{self.entry.fold}"
            )
            paths = export_threshold_outputs(
                output_dir=output_dir,
                stem=self.state.current.parent.name,
                image=self.full_image,
                probability=self.full_probability,
                threshold=self.spin.value(),
                policy=self.policy,
            )
            QMessageBox.information(
                self,
                "Export complete",
                f"Saved mask, overlay and policy to:\n{paths['mask'].parent}",
            )

    application = QApplication.instance() or QApplication(sys.argv)
    window = GalleryWindow()
    window.show()
    return application.exec()


def choose_preview_sources(
    checkpoint: str | None,
    image: str | None,
) -> tuple[str, str]:
    """Ask for any missing checkpoint/image through native desktop dialogs."""

    try:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication, QFileDialog
    except ImportError as exc:
        raise RuntimeError(
            "尚未安裝 PySide6；請依 requirements-gui.txt 安裝桌面 viewer"
        ) from exc

    global _PICKER_APPLICATION
    _PICKER_APPLICATION = QApplication.instance() or QApplication(sys.argv)
    settings = QSettings("cihcilab", "crack-threshold-preview")
    project_root = Path(__file__).resolve().parents[1]
    if checkpoint is None:
        start = str(settings.value("checkpoint_dir", str(project_root / "runs")))
        checkpoint, _ = QFileDialog.getOpenFileName(
            None,
            "選擇 Crack 或 Craquelure checkpoint",
            start,
            "PyTorch checkpoint (*.pt)",
        )
        if not checkpoint:
            raise SystemExit("已取消 checkpoint 選擇")
        settings.setValue("checkpoint_dir", str(Path(checkpoint).parent))
    if image is None:
        qualitative = Path(checkpoint).parent.parent / "qualitative"
        default_image_dir = qualitative if qualitative.is_dir() else project_root
        start = str(settings.value("image_dir", str(default_image_dir)))
        image, _ = QFileDialog.getOpenFileName(
            None,
            "選擇要調整 threshold 的圖片",
            start,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not image:
            raise SystemExit("已取消圖片選擇")
        settings.setValue("image_dir", str(Path(image).parent))
    return checkpoint, image


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "不帶參數開啟 validation 圖片瀏覽器；也可指定 image/probability "
            "預覽既有 probability。"
        )
    )
    result.add_argument("--image", help="原始 RGB 圖片")
    result.add_argument("--probability", help="predict_full --save_prob 產生的 .npy")
    result.add_argument("--policy", help="validation 校正產生的 threshold_policy.json")
    result.add_argument("--expert", default="crack", help="沒有 --policy 時使用")
    result.add_argument("--foreground-id", type=int, default=1, help="原始 mask 中的 expert class ID")
    result.add_argument("--mask", help="可選；顯示此張影像隨 threshold 變動的指標")
    result.add_argument("--output-dir", default="outputs/threshold_preview")
    result.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help=argparse.SUPPRESS,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.image is None and args.probability is None:
        return launch_gallery(Path(args.project_root))
    if args.image is None or args.probability is None:
        argument_parser.error("--image 與 --probability 必須一起指定")
    image_path = Path(args.image)
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    probability = np.load(args.probability, allow_pickle=False)
    policy = (
        ThresholdPolicy.load(args.policy)
        if args.policy
        else ThresholdPolicy.default(args.expert, args.foreground_id)
    )
    target = None
    if args.mask:
        with Image.open(args.mask) as source:
            target = np.asarray(source)
        if target.ndim == 3:
            target = target[..., 0]
    return launch_viewer(
        image=image,
        probability=probability,
        policy=policy,
        output_dir=Path(args.output_dir),
        stem=image_path.stem,
        target=target,
    )


if __name__ == "__main__":
    raise SystemExit(main())
