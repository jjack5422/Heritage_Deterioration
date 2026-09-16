# Dataset115 Expert Views Design

## Goal

Convert the reviewed `dataset115_filtered` annotations into three traceable binary training views without modifying the original per-D-code masks.

The canonical experts are:

- `shrinkage_craquelure`: union of `D-03` Shrinkage and `D-04` Craquelure.
- `scratch_crack`: union of `D-11` Scratch and `D-01` Crack.
- `loss`: `D-02` Loss only.

This change prepares training data only. It does not create train/validation splits or start training.

## Source Contract

The source dataset is `dataset115_filtered` and contains 715 reviewed image tiles. Each tile may have zero or more independent binary masks under its corresponding mask directory. Source masks use background `0` and foreground `255`; different D-code masks may overlap.

`dataset115_filtered/metadata/manifest.csv` is the source inventory. The build must reject missing images, missing declared masks, invalid mask dimensions, non-binary masks, duplicate source-mask records, or checksum mismatches.

## Corrected Class Semantics

Update `dataset115_filtered/metadata/classes.json` so that:

| D-code | Canonical name |
|---|---|
| `D-01` | `Crack` |
| `D-02` | `Loss` |
| `D-04` | `Craquelure` |

All other D-code names remain unchanged. The expert mapping is also recorded in generated metadata; the source mask files remain untouched.

## Expert Mapping

```text
shrinkage_craquelure = D-03 OR D-04
scratch_crack         = D-11 OR D-01
loss                  = D-02
```

`D-26`, `D-16`, and `D-36` are not included. The user-defined Dataset115 semantics designate `D-01` as Crack and `D-02` as Loss, and those additional classes are absent from the filtered dataset's `present_ids`.

Other present D-codes remain in the source dataset but are background for these three binary expert views.

## Output Layout

```text
dataset115_filtered/expert_views/
├── shrinkage_craquelure/
│   ├── masks/<temple>/<source_group>/<tile>.png
│   └── manifest.csv
├── scratch_crack/
│   ├── masks/<temple>/<source_group>/<tile>.png
│   └── manifest.csv
└── loss/
    ├── masks/<temple>/<source_group>/<tile>.png
    └── manifest.csv

dataset115_filtered/metadata/expert_views.json
```

Images are not copied. Every generated manifest references the existing source image by a path relative to `dataset115_filtered`.

Each expert view contains all 715 reviewed tiles:

- A tile with one or more mapped source masks receives their pixel-wise OR.
- A tile without a mapped source mask receives a zero-valued mask and is an explicit negative sample.
- Output masks are single-channel `uint8` PNGs containing only `0` and `255`.

## Manifest Contract

Each expert manifest contains one row per source image with these fields:

```text
expert
temple
source_group
tile
image
mask
source_codes
foreground_pixels
is_positive
image_sha256
mask_sha256
```

`source_codes` lists the mapped D-codes actually present for that tile in stable numeric order. `is_positive` is derived from `foreground_pixels > 0`.

`metadata/expert_views.json` records:

- schema version and generation timestamp;
- source dataset and source manifest hash;
- corrected class semantics;
- expert-to-D-code mapping and `pixelwise_or` merge rule;
- foreground/background encoding;
- per-expert tile, positive-tile, negative-tile, and foreground-pixel counts;
- generated manifest hashes.

## Builder

Add `scripts/data/build_dataset115_expert_views.py`.

The builder reads only the source manifest and declared files. It writes all derived output to a staging directory under `dataset115_filtered`, validates the complete staging result, then atomically replaces `expert_views`. Existing derived output may be replaced; original image, mask, review, and source metadata files must never be deleted or rewritten, except for the approved name corrections in `metadata/classes.json`.

If validation fails, the existing `expert_views` remains intact and the staging directory is removed.

## Verification

The build is accepted only when:

1. Each expert has exactly 715 mask files and 715 manifest rows.
2. Every manifest image and output mask exists.
3. Every source image checksum matches `metadata/manifest.csv`.
4. Every generated mask matches its image dimensions and contains only `0` and `255`.
5. Each generated mask equals the pixel-wise OR of its declared source masks.
6. Every generated mask checksum matches its manifest value.
7. Source D-code mask paths and checksums remain unchanged.
8. `classes.json` contains the approved `D-01`, `D-02`, and `D-04` names.

A focused behavioral test covers union semantics, empty negative masks, overlapping source masks, and deterministic manifest ordering. After implementation, run the focused test and execute the builder against the full dataset as the smoke test.
