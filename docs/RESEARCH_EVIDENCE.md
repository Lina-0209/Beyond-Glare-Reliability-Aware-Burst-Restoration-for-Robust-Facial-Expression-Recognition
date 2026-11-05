# Research evidence and display provenance

This record binds the research introduction in both READMEs to the project's
recorded experiments. It contains aggregate results and source hashes, not face
images, person-level records, datasets, checkpoints, or prediction caches.
README claims C001–C010 are cross-referenced in HTML comments; each claim group
covers the preceding section. This is a repository overview, not a submitted paper
or a claim of compliance with any particular venue's author instructions.

## Sources

The following original experiment records are retained in the research workspace,
not bundled in this code release. Paths below are workspace-relative identifiers,
not downloadable public assets. Hashes identify the exact records used here.

| ID | Original record | Supporting location |
|---|---|---|
| E001 | `outputs/rafdb_cross_backbone_v1/FROZEN_PROTOCOL.json` | Frozen post-hoc protocol, inputs, source cohort, model identities, preprocessing and analysis settings. |
| E002 | `outputs/rafdb_cross_backbone_v1/summary.json` | `requested_test_samples`, `complete_paired_samples`, `coverage`, `metrics`, `comparisons`. |
| E003 | `outputs/final_validation_resplit/frame_ablation.json` | `results.celeb` and `results.ffhq`: fixed-checkpoint 1F/2F/4F metrics. |
| E004 | `outputs/final_validation_resplit/resplit_summary.json` | `split` and `berr.parameters`. |
| E005 | Included model implementation | [model](../models/berr/model.py), [encoder](../models/berr/backbone.py), [degradation/fusion](../models/berr/degradation.py), [decoder](../models/berr/restoration.py), [losses](../models/berr/losses.py), [configuration](../configs/berr/combined.yaml). |
| E006 | `outputs/final_validation_resplit/dataset_split.csv` | Per-domain and combined source/burst counts for every split. |
| E007 | Original deployment adapter and project integration record (not bundled) | `emotion_service.py`: single-JPEG `POST /recognize`; `vision_module/group_emotion_adapter.py`: face crop, rejection, seven-to-five mapping and public response; `docs/PROJECT_OVERVIEW_CN.md` §11: backend responsibilities and the four-frame extension boundary. |
| E008 | Earlier project pipeline diagram (retained historical asset) | `docs/figures/fig01_pipeline.png`; superseded in the READMEs by E009. |
| E009 | BERR architecture schematic, revised 2026-11-05 | `docs/figures/fig01_pipeline.svg`; topology checked against E005. Embedded photographic tiles come from existing diagnostic sample `5987_gpt_v00`; final face thumbnails use Ground Truth to illustrate ideal restoration. |

```text
E001  0c14db58e6a3f4aca8cfbe8656c6413b52a060c234d44fbe56e6ce04958495da
E002  211bd5dcfc78a3aa2d46777020a1fdda40a4f1a4a74a1c7246b70774cf762ade
E003  217f6bdb5faf64a8899620d4c9a7e7aa23c3bc2b94d6d7b188b4747b3e0d69a2
E004  697be6c2724039f4d9be5b66eae3e2889318e84e42461f15a22ac52abcc0fd61
E006  88072f05a3fca2ddfa0ca58203ea5472ef50b76090a533f0822a3fd31dc56db6
E007 emotion_service.py  e8afedf999c92c96aff0772e747f4d5abeb1e1cbb6fe4b4896099e7bd919f6ad
E007 group_emotion_adapter.py  2a984fda8e3303410dd8b06882169aa388c47ddd6512ba21a25176e8772b9d6f
E007 PROJECT_OVERVIEW_CN.md  3360bba2c63cb1b346f01d8180216a373de91965ad3453d55e43c154cc8ee569
E008  3b75891fc2f738c4a36a4ba28cbb42eaad6c79f568126c1bcfce5f8486c421aa
E009  41d77efb6a8aaf0e2d22cf12ef6e464838e64eeca39a535bff99904ea0a0ec23
```

| Claim | README section / purpose | Evidence |
|---|---|---|
| C001 | Full-project abstract: lightweight BERR, LensGlare-4, frozen FER, confidence/contract adaptation, group learning-platform integration boundary, and Strong-glare gains | E001, E002, E004–E007 |
| C002 | Contributions: architecture, dataset scale and paired validation | E001–E006 |
| C003 | Methodology: five stages, dimensions, reliability and training objective | E005 |
| C004 | Experiment protocol: cohort, coverage, scale, frozen models and scoring | E001, E002 |
| C005 | Cross-recognizer trends, Strong table, paired intervals and tests | E002 |
| C006 | Strong utility and the aligned-mean system-level comparison | E001, E002, E005 |
| C007 | Fixed-checkpoint temporal-evidence experiment | E003, E004, E006 |
| C008 | Lightweight eyeglass-glare removal: 1,003,847 BERR parameters, shared CNN, local ROI; downstream FER/detector excluded from the count | E004, E005 |
| C009 | Group learning-platform integration: documented single-JPEG contract, backend fusion/records, contract mapping and required burst extension | E005, E007 |
| C010 | BERR architecture: alignment/cropping, multi-scale reliability fusion, visible/prior decoding and local composition | E005, E009 |

The BERR parameter count was also checked by constructing the default model and
summing `p.numel()` over `model.parameters()`: **1,003,847**. This is a structural
count, not an end-to-end memory, latency, or real-time performance benchmark.
The group integration section describes the recorded contract and an explicit
four-frame extension route; it does not claim new deployment validation, automatic
BERR execution for single JPEGs, or validated five-level learning-state accuracy.
No group service implementation, credentials, hostnames, or infrastructure files
are added to the public release.

## Evaluation definitions

- This is a **post-hoc frozen-model extension**. No restoration/FER retraining,
  test-threshold calibration, or synthesis-parameter selection is performed for it.
- RAF-DB official test: 3,068 images. Historical detected cohort: 2,824. All selected
  paths, backbones, and severities complete this same cohort. The 244 `no_face`
  cases remain in the coverage denominator; the reported Accuracy/Macro-F1 are
  conditional on the common cohort, not full-test end-to-end accuracy.
- Native input: 100 × 100 RGB. Glare is synthesized deterministically in the lens
  region. Clean-derived synthetic geometry is shared with the recorded protocol;
  these results do not measure landmark detection on real glare video.
- Direct: native glare anchor. NAFNet: one anchor enlarged to 256 × 256 using OpenCV
  bilinear interpolation, then restored. Mean + NAFNet: four similarly enlarged
  frames, BERR's landmark similarity alignment and bilinear/border sampling,
  full-face equal-weight RGB average, one uint8 quantization, then the same NAFNet.
  BERR: four 256 × 256 frames, internal 128 × 128 ROI and local composition.
- The NAFNet advantage statements concern the **matched 256 × 256 restoration
  inputs**, not every possible NAFNet input scale or image-reconstruction metric.
  Reliability weighting, ROI processing, decoding, and composition all distinguish
  BERR from the mean baseline; the comparison does not isolate one module's effect.
- Both backbones consume identical cached PNG pixels per condition. enet_b2_7 and
  POSTER V2 use the recorded AffectNet-7 weights and preprocessing, not RAF-DB-trained
  replacements. The POSTER label crosswalk is the project's recorded mapping; see
  [third-party boundaries](../THIRD_PARTY_NOTICES.md).
- Seven-class top-1 Accuracy and class-unweighted Macro-F1 use every paired
  prediction with rejection disabled. These are not five-level learning-state
  metrics or confidence-selected accuracy. Clean references are point estimates,
  not theoretical upper bounds.
- Differences are computed before rounding. All reported differences use
  **percentage points (pp)**, not relative percentage improvements.
- Intervals: 5,000 source-paired percentile-bootstrap resamples, 95%, **unadjusted
  per comparison**. Accuracy tests: two-sided exact McNemar, Holm-adjusted separately
  within each comparison family of two backbones × three severities (six tests).
  Statistical support for Strong does not assert significance at every severity.
- Frame-evidence evaluation uses one frozen BERR on the 226 source-disjoint test
  bursts (152 CelebAMaskHQ-derived / 74 FFHQ-derived). It repeats 1 or 2 observations
  to preserve four input slots. The dataset split is clean-source-disjoint; no
  independently verified person-identity-disjoint claim is made.
- LensGlare-4: 1,792 / 225 / 226 training/validation/test bursts from 487 / 61 / 62
  clean sources. Total: 2,243 four-frame bursts, 8,972 frames, 610 sources.
  Naming the dataset does not imply that its images are publicly distributed.

## Full-strength main-path point estimates

All rows use the same n = 2,824. Units are percentages. These values are transcribed
from E002; the plots retain all main paths, including negative gains from averaging.

| Strength | Path | enet Accuracy | enet Macro-F1 | POSTER V2 Accuracy | POSTER V2 Macro-F1 |
|---|---|---:|---:|---:|---:|
| Mild | Direct | 76.35 | 63.02 | 75.04 | 62.00 |
| Mild | NAFNet | 76.59 | 63.66 | 75.50 | 62.70 |
| Mild | Mean + NAFNet | 75.50 | 62.13 | 74.89 | 61.52 |
| Mild | BERR | 77.02 | 64.21 | 75.92 | 63.26 |
| Medium | Direct | 75.60 | 62.41 | 73.48 | 60.46 |
| Medium | NAFNet | 76.20 | 62.93 | 74.61 | 61.78 |
| Medium | Mean + NAFNet | 74.33 | 61.00 | 73.87 | 60.84 |
| Medium | BERR | 76.88 | 63.87 | 76.10 | 63.46 |
| Strong | Direct | 74.47 | 60.95 | 73.12 | 60.01 |
| Strong | NAFNet | 75.67 | 62.41 | 74.29 | 61.37 |
| Strong | Mean + NAFNet | 73.34 | 59.54 | 73.48 | 60.42 |
| Strong | BERR | 76.84 | 63.61 | 76.06 | 63.17 |

## Display provenance

The two result figures are **unmodified copies of existing project plots**, not new analyses.
They contain aggregate charts, not source photographs. Original source: E002 and
the corresponding `results.csv`; original rendering/provenance is retained in
`outputs/rafdb_cross_backbone_v1/`. The blue revision is retained in its
`figure_revisions/20261029/` subdirectory. No cropping, panel removal, recoloring,
resampling, or PDF export is performed for this README update.

| Display | Bundled file | Version / encoding | SHA256 |
|---|---|---|---|
| Pipeline / BERR architecture | `figures/fig01_pipeline.svg` | Three-panel editable SVG; native vector architecture with embedded photographic/diagnostic images; final faces illustrate ideal restoration using Ground Truth | `41d77efb6a8aaf0e2d22cf12ef6e464838e64eeca39a535bff99904ea0a0ec23` |
| Figure 1 / project Fig. 6d | `figures/fig06d_rafdb_four_paths_two_fer.png` | Original four-path, dual-FER severity curves; point estimates with clean references | `beda1a4dcc6937e557ba486b4d3d85678fa646daf7117da895343c54c4d00255` |
| Figure 2 / project Fig. 6e | `figures/fig06e_rafdb_strong_utility_heatmap.png` | User-approved 2026-10-29 blue revision; signed pp gains; independent panel scales | `29383f7243b19ebc9160e8be297bf4be2e3506f40f4abcec5c4014e8d9b739c3` |

The README pipeline was replaced on 2026-11-04 and its example revised on
2026-11-05 at the project owner's request, using a BERR architecture schematic
checked against commit
`ad76b408a8b1130f70d9253d14a765e67f5e380d`. Panels (a)-(c) show the architecture,
pixel-wise reliability fusion, and mask-guided local composition. Downstream FER
and platform integration are outside this figure.

Photographic tiles are embedded from the existing validation diagnostic
`5987_gpt_v00` (epoch 57, step 20862), which uses synthetic glare on a photographic
face. Only input frame 0 is shown; rear sheets indicate the frame dimension.
The displayed anchor crop is taken directly from full-frame pixels. Its
191 × 59 image-space rectangle (x=30, y=98) is recorded in project metadata
`000005_hq5987_b0.json`; matching the exported anchor ROI confirms the crop.
Normalized prior/restored ROIs are reprojected into this rectangle for display. Network preprocessing still resizes crops to
128 × 128. Feature sheets and weight maps are schematic; V and S thumbnails are
RGB symbols because their images were not exported. The two final face thumbnails
use Ground Truth to illustrate the intended ideal restoration, not an observed
prediction. This distinction appears in both README captions. No new inference,
training, or performance measurement was run. The old PNG is retained as a
historical asset and is no longer the README pipeline.

Creator attribution: existing project experiment/plotting workflow; no new human
creator attribution is inferred. Permission scope: reuse in this repository's
documentation was requested by the project owner. Source-dataset/model terms remain
separate from the project code's MIT license. Confirm public redistribution and
institutional/group requirements before GitHub publication.

Automated checks compare README numbers with the recorded JSON, check local links,
and bind figure files by SHA256. Original scientific human-review status remains
**pending**; no human verifier or submission approval is inferred from these checks.
The portable code can run a new authorized experiment, but exact replay of the
recorded cohort requires its original manifests, data, and checkpoints, which are
not distributed here. No new experiment is run for this documentation update.

## Documentation workflow reference

AI-assisted drafting and evidence bookkeeping used the scientific-writing skill;
figure checks used paper-visualization. These are documentation aids, not sources
of model-performance evidence or method novelty. Human owners retain final review.

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.*
[arXiv:2609.00065](https://arxiv.org/abs/2609.00065).
