# Monument DA-SAM3

Two-channel, fixed-concept segmentation of `crack_craquelure` and `loss` in
painted cultural heritage. The approved model uses the full SAM3 concept path
at 512 x 512, three DA-MoE fusion layers, four rank-8 DPE experts, DER top-2
routing, and two-stage specialization.

The user supplies only an image. Canonical English concepts are versioned in
`configs/concepts.yaml`; aliases are metadata and are not prompt augmentation.

Run artifacts are written under `runs/<experiment_id>/5fold/da_sam3/fold<k>`
and are intentionally excluded from Git.
