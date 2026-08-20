# SAM2-Adapter

This directory owns the SAM2-Adapter implementation, tests, and experiment
artifacts. It is independent from the `sam2_sac` H0 baseline project.

The model trains two one-vs-rest experts (Crack and Craquelure), injects
stage-aware visual adapters into the native SAM2.1 Hiera-L trunk, and trains the
active native mask-decoder path. Dual-expert thresholds are selected only from
nested validation data before the frozen outer-test evaluation.

Run the controlled five-fold experiment from the workspace root:

```bash
crackseg_env/bin/python -m sam2_adapter.train_adapter \
  --folds 0 1 2 3 4 --experts crack craquelure \
  --experiment-id 2026-08-20_sam2-adapter-hiera-large_dual-expert_512_80ep_seed42
```

Outputs stay under `sam2_adapter/runs/<experiment_id>/5fold/<expert>/fold<k>/`.
The historical SAC H0 code and runs remain under `sam2_sac/`.
