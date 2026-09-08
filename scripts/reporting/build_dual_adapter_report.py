#!/usr/bin/env python3
"""Rebuild Dual-Adapter SAM3 class-specific validation reports from saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from dual_adapter_sam3.class_report import build_class_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, action="append")
    source.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    if args.experiment_root is not None:
        run_dirs = sorted((args.experiment_root / "5fold" / "da_sam3").glob("fold*"))
    else:
        run_dirs = args.run_dir
    if not run_dirs:
        parser.error("no fold run directories were found")
    for run_dir in run_dirs:
        report = build_class_report(run_dir)
        print(report)


if __name__ == "__main__":
    main()
