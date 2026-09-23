"""U-Net project entry point for shared threshold calibration."""

from crackseg_common.thresholding import *  # noqa: F403
from crackseg_common.thresholding import main


if __name__ == "__main__":
    raise SystemExit(main())
