from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from adapters.tiled_inference import sliding_positions, tiled_foreground_probability


def test_sliding_positions_cover_short_exact_and_overlapping_lengths() -> None:
    assert sliding_positions(300, tile_size=512, stride=384) == (0,)
    assert sliding_positions(512, tile_size=512, stride=384) == (0,)
    assert sliding_positions(1024, tile_size=512, stride=384) == (0, 384, 512)


def test_tiled_probability_restores_arbitrary_original_size() -> None:
    image = Image.new("RGB", (701, 533), "white")
    observed_batches: list[tuple[int, ...]] = []

    def predict_batch(batch: torch.Tensor) -> torch.Tensor:
        observed_batches.append(tuple(batch.shape))
        return torch.full(
            (batch.shape[0], batch.shape[2], batch.shape[3]),
            0.75,
            device=batch.device,
        )

    probability, tile_count = tiled_foreground_probability(
        image,
        predict_batch,
        device=torch.device("cpu"),
        tile_size=512,
        stride=384,
        batch_size=2,
    )

    assert probability.shape == (533, 701)
    assert probability.dtype == np.float32
    assert np.allclose(probability, 0.75)
    assert tile_count == 4
    assert observed_batches == [(2, 3, 512, 512), (2, 3, 512, 512)]


def test_tiled_probability_rejects_non_finite_model_output() -> None:
    def predict_batch(batch: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (batch.shape[0], batch.shape[2], batch.shape[3]),
            float("nan"),
        )

    try:
        tiled_foreground_probability(
            Image.new("RGB", (32, 24), "black"),
            predict_batch,
            device=torch.device("cpu"),
            tile_size=32,
            stride=16,
            batch_size=1,
        )
    except RuntimeError as exc:
        assert "non-finite" in str(exc)
    else:  # pragma: no cover - explicit diagnostic
        raise AssertionError("non-finite output should be rejected")
