# Local data contract

LensGlare-4 is the name of the project's synthetic four-frame eyeglass-glare data,
not a new name or license for its original face-image sources. No images, identity
lists, private manifests or external image-generation API material are bundled.
Historical image-generation/upload tooling is intentionally outside this focused
code release; the code accepts independently authorized, locally prepared inputs.

Each prepared split has the following structure (paths are examples, not files
included in the repository):

```text
data/lensglare4/celeb/train/
  target/sample.png
  bursts/sample/frame_000.png ... frame_003.png
  burst_masks/sample/frame_000.png ... frame_003.png
  burst_landmarks/sample.json
  roi_bbox/sample.json
  geometry_heatmap_roi/sample.png
  metadata/sample.json
```

The loader also accepts optional `mask/`, `glare_layer/`, and `landmarks/` fields.
For fidelity to the training protocol, provide real per-frame landmarks and masks;
the loader's missing-field compatibility paths are not proof of valid annotations.
The `metadata` field `source_id` must identify the same clean source across all its
variants. Source-disjoint splitting does not establish person-disjoint splitting
unless a verified identity mapping also exists.

The model consumes:

- `burst`: `[B,4,3,H,W]`, RGB float in `[0,1]`.
- `landmarks`: `[B,4,K,2]`, per-frame pixel coordinates.
- `roi_bbox`: `[B,4]`, `[x0,y0,x1,y1]` for the reference face.
- `geometry_heatmap_roi`: `[B,1,R,R]`.
- Training additionally uses the clean reference target and per-frame glare masks.

For labelled FER, a CSV has `image_path,split,sample_id,label` (or `label_id` /
`label_name`). Paths can be absolute at runtime or relative to the CSV directory;
sample IDs must be unique in the requested split. Labels must have an explicit
crosswalk. RAF-DB label IDs and generic seven-class labels are kept in separate
mapping YAMLs. Do not guess label order from a model checkpoint name.
