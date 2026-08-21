# SAM2-Adapter

This directory owns the SAM2-Adapter implementation, tests, and experiment
artifacts. It is independent from the `sam2_sac` H0 baseline project.

The model trains one binary merged-crack foreground model, injects stage-aware
visual adapters into the native SAM2.1 Hiera-L trunk, and trains the active
native mask-decoder path. Crack and craquelure are not separate targets or
models. The merged dataset's pre-registered binary threshold is 0.5; there is
no validation threshold search or expert fusion.

Run the controlled five-fold experiment from the workspace root:

```bash
crackseg_env/bin/python -m sam2_adapter.train_adapter \
  --folds 0 1 2 3 4 \
  --experiment-id 2026-08-21_sam2-adapter-hiera-large_merged-crack_512_80ep_seed42
```

Outputs stay under
`sam2_adapter/runs/<experiment_id>/5fold/foreground/fold<k>/`.
Raw label 1 is the merged crack/craquelure foreground, raw label 0 is
background, and other deterioration labels are excluded from this binary loss
and scoring task.
The historical SAC H0 code and runs remain under `sam2_sac/`.
