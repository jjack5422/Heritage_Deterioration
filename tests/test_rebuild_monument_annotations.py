from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rebuild_monument_annotations import (  # noqa: E402
    CLASS_IDS,
    decode_cvat_rle,
    render_cvat_image,
)


class CvatMaskConversionTests(unittest.TestCase):
    def test_decode_cvat_rle_places_foreground_in_offset_bounding_box(self) -> None:
        decoded = decode_cvat_rle(
            rle="0, 1, 2, 1",
            image_width=4,
            image_height=3,
            left=1,
            top=1,
            width=2,
            height=2,
        )

        expected = np.zeros((3, 4), dtype=bool)
        expected[1, 1] = True
        expected[2, 2] = True
        np.testing.assert_array_equal(decoded, expected)

    def test_render_cvat_image_uses_foreground_classes_not_background_shape(self) -> None:
        image = ET.fromstring(
            """
            <image name="tile.png" width="4" height="3">
              <mask label="background" rle="12" left="0" top="0" width="4" height="3" z_order="0" />
              <mask label="crack" rle="0, 1, 10, 1" left="0" top="0" width="4" height="3" z_order="0" />
              <mask label="loss" rle="6, 1, 5" left="0" top="0" width="4" height="3" z_order="0" />
            </image>
            """
        )

        rendered = render_cvat_image(image)

        expected = np.zeros((3, 4), dtype=np.uint8)
        expected[0, 0] = CLASS_IDS["crack"]
        expected[1, 2] = CLASS_IDS["loss"]
        expected[2, 3] = CLASS_IDS["crack"]
        np.testing.assert_array_equal(rendered.class_mask, expected)
        self.assertEqual(rendered.cross_class_overlap_pixels, 0)

    def test_render_cvat_image_resolves_cross_class_overlap_by_explicit_priority(self) -> None:
        image = ET.fromstring(
            """
            <image name="tile.png" width="2" height="2">
              <mask label="crack" rle="0, 1, 3" left="0" top="0" width="2" height="2" z_order="0" />
              <mask label="loss" rle="0, 1, 3" left="0" top="0" width="2" height="2" z_order="0" />
            </image>
            """
        )

        rendered = render_cvat_image(image)

        self.assertEqual(rendered.cross_class_overlap_pixels, 1)
        self.assertEqual(rendered.cross_class_overlap_by_pair, {"loss__crack": 1})
        self.assertEqual(rendered.class_mask[0, 0], CLASS_IDS["crack"])


if __name__ == "__main__":
    unittest.main()
