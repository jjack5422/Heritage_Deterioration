"""Segmentation model adapters with a shared inference contract."""

from adapters.base import SegmentationAdapter
from adapters.convnext_unet import ConvNextUnetAdapter
from adapters.dummy import DummyAdapter
from adapters.resunet import ResUNetAdapter
from adapters.sam2_adapter import SAM2Adapter
from adapters.sam3_adapter import SAM3Adapter

__all__ = [
    "ConvNextUnetAdapter",
    "DummyAdapter",
    "ResUNetAdapter",
    "SAM2Adapter",
    "SAM3Adapter",
    "SegmentationAdapter",
]
