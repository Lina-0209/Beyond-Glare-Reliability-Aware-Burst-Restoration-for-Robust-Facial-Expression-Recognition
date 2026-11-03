# Code-only export scope

This is a focused export of the current project implementation, not a copy of the
entire research workspace or a claim of new scientific results.

## Retained

Core BERR/NAFNet networks, preprocessing, frozen FER integration, confidence policy,
ROI/video caching, dataset loaders/source splitting,
training/evaluation entry points, frame-evidence baseline and current aligned-mean
baseline. The optional glare gate remains experimental and requires local training.

## Excluded rather than deleted from the original project

All training/pretrained weights, face-landmarker assets, binary gate artifacts,
datasets, cached faces, predictions, and historical output directories;
research reports/slides, meeting notes, old experiment launchers and report scripts;
GPT/Qwen paid image-generation clients and old pilot/download/review scripts;
unused LibreFace/Qwen FER adapters; the single-image HTTP service and the
seven-to-five contract adapter used to integrate with other modules; cloud
hostnames/IPs, group-specific Docker settings, secrets, SSH material, hidden
resource forks and old Git history.

## Release-only maintenance

- Portable paths replace private output/checkpoint locations.
- Confidence threshold numbers are retained, while private metadata and old
  calibration result tables are removed.
- A portable manifest/cache evaluator replaces the private multi-stage historical
  dependencies. Its cohort selection is explicit; original per-image result caches
  are not bundled.
- Paired metric helpers are extracted without changing their calculation.
- The POSTER path compatibility patch is scoped to model loading, not left as a
  process-global `torch.load` replacement.
- A tiny synthetic smoke config, offline tests, GitHub Actions and release checks
  are added. No actual datasets or pretrained model inference are needed for those
  tests. Planned reliability/prior ablations are not falsely presented as completed.

No models were retrained and no new research experiments were run to assemble this
export. Original core architecture/weights remain separate from packaging changes.

## Research README update (2026-10-30)

- Both READMEs now introduce the method through an abstract, contributions, a
  five-stage methodology, and claim-oriented experiment summaries.
- Two existing aggregate plots (project Figs. 6d and the approved blue 6e) and
  aggregate tables document the frozen research results. No face photographs,
  licensed dataset images, weights, or prediction caches are added.
- `docs/RESEARCH_EVIDENCE.md` records cohort definitions, comparison scales,
  statistical conventions, claim/source links, source hashes and figure provenance.
- The release checker accepts exactly these two figure hashes; arbitrary image
  files remain rejected. Five offline tests exercise this documentation exception.
- This documentation update does not restore the removed group-specific HTTP
  integration or change model, training, inference, or evaluation behavior.

## Assembly validation

- Offline unit tests cover the retained research code. The earlier Python 3.10
  assembly run reused the research environment's dependencies and predates removal
  of the single-image HTTP contract tests.
- A one-step tiny synthetic training run completes forward/backward, validation,
  and checkpoint saving. Generated outputs are moved outside the code package.
- All 19 model source files match the source snapshot byte-for-byte; the numeric
  confidence thresholds are unchanged.
- Static checks find no missing local imports or embedded credential/private-path
  patterns. These checks do not certify absence of every possible secret.
- `pip check` reports the documented EmotiEffLib OpenCV metadata mismatch; runtime
  tests still pass.
- GitHub Actions configuration is provided but has not been run on GitHub.
