#!/usr/bin/env python3
"""Build a local RAF-DB Basic Emotion manifest without altering raw files.

The RAF-DB archive must be obtained through the official access procedure.  This
script only discovers the official partition labels and image files, writes a
manifest, creates the local experiment directories, and creates a human-review
license-audit template.  It never downloads or uploads RAF-DB images.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

RAF_SOURCE_URL = "http://www.whdeng.cn/RAF/model1.html"
RAF_LABELS = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}


def _find_one(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        keys = {path.name, path.stem, path.stem.removeprefix("aligned_")}
        for key in keys:
            index.setdefault(key.lower(), path)
    return index


def _resolve_image(token: str, index: dict[str, Path]) -> Path | None:
    token_path = Path(token)
    candidates = [
        token,
        token_path.name,
        token_path.stem,
        token_path.stem.removeprefix("aligned_"),
        f"aligned_{token_path.stem}",
        f"aligned_{token_path.stem.removeprefix('aligned_')}",
    ]
    for candidate in candidates:
        if candidate.lower() in index:
            return index[candidate.lower()]
    match = re.search(r"(\d+)", token_path.stem)
    if match:
        number = match.group(1).lstrip("0") or "0"
        for key, path in index.items():
            normalized = Path(key).stem.removeprefix("aligned_").lstrip("0") or "0"
            if normalized == number:
                return path
    return None


def _write_license_template(project_root: Path, *, source_url: str, force: bool = False) -> Path:
    output = project_root / "outputs/rafdb_validation/dataset_license_audit.md"
    if output.exists() and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""# RAF-DB Dataset License Audit

**Status: PENDING HUMAN VERIFICATION — do not treat this template as a completed license audit.**

| Field | Record |
|---|---|
| Official source URL | {source_url} |
| Subset | RAF-DB Basic Emotion / single-label subset |
| Download date | TODO: record the actual date |
| Access method | TODO: record the official registration/password procedure used |
| Permitted usage | TODO: verify the official non-commercial research terms |
| Local-only restriction | TODO: confirm and retain dataset locally |
| No redistribution statement | TODO: confirm no RAF-DB images or derived dataset files will be redistributed |
| Human verifier | TODO: name/role of the accountable project member |
| Verification date | TODO |

Raw RAF-DB files must remain unmodified under `data/rafdb/raw/` and must not be
uploaded to external APIs or repositories.  Complete the TODO fields from the
actual official access record before reporting RAF-DB results.
""",
        encoding="utf-8",
    )
    return output


def build_manifest(labels_file: Path, image_root: Path, output_root: Path, *, project_root: Path | None = None) -> dict:
    index = _image_index(image_root)
    records: list[dict[str, str]] = []
    missing: list[str] = []
    seen: set[str] = set()
    with labels_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            token, raw_label = fields[0], fields[1]
            split = "train" if token.lower().startswith("train") else "test" if token.lower().startswith("test") else "unknown"
            if split == "unknown":
                raise ValueError(f"Cannot infer train/test split at {labels_file}:{line_number}: {line.strip()}")
            raw_label = str(raw_label).strip()
            if raw_label not in RAF_LABELS:
                raise ValueError(f"Unexpected RAF-DB Basic Emotion label {raw_label!r} at line {line_number}")
            resolved = _resolve_image(token, index)
            sample_id = Path(token).stem
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id in labels file: {sample_id}")
            seen.add(sample_id)
            if resolved is None:
                missing.append(token)
                continue
            records.append({
                "sample_id": sample_id,
                "image_path": str(resolved.resolve()),
                "label_id": raw_label,
                "label_name": RAF_LABELS[raw_label],
                "split": split,
            })

    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("raw", "aligned", "metadata", "glare/clean", "glare/medium_glare", "glare/strong_glare", "burst",):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    manifest = output_root / "metadata.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path", "label_id", "label_name", "split"])
        writer.writeheader()
        writer.writerows(records)

    project_root = project_root or Path(__file__).resolve().parents[1]
    license_audit = _write_license_template(project_root, source_url=RAF_SOURCE_URL)
    generation = {
        "labels_file": str(labels_file.resolve()),
        "image_root": str(image_root.resolve()),
        "output_root": str(output_root.resolve()),
        "manifest": str(manifest.resolve()),
        "records": len(records),
        "missing_images": missing,
        "split_counts": dict(Counter(row["split"] for row in records)),
        "label_counts": dict(Counter(row["label_name"] for row in records)),
        "license_audit": str(license_audit),
    }
    (output_root / "metadata/manifest_generation.json").write_text(json.dumps(generation, indent=2), encoding="utf-8")
    return generation


def build_directory_manifest(dataset_root: Path, output_root: Path, *, project_root: Path | None = None) -> dict:
    """Build a manifest from DATASET/{train,test}/{1..7}/*.jpg exports."""
    hierarchy_root = dataset_root / "DATASET" if (dataset_root / "DATASET").is_dir() else dataset_root
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for split in ("train", "test"):
        split_root = hierarchy_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Expected RAF-DB split directory: {split_root}")
        for label_id, label_name in RAF_LABELS.items():
            label_root = split_root / label_id
            if not label_root.is_dir():
                raise FileNotFoundError(f"Expected RAF-DB label directory: {label_root}")
            for image_path in sorted(label_root.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                sample_id = image_path.stem
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample_id in directory export: {sample_id}")
                seen.add(sample_id)
                records.append({
                    "sample_id": sample_id,
                    "image_path": str(image_path.resolve()),
                    "label_id": label_id,
                    "label_name": label_name,
                    "split": split,
                })

    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("raw", "aligned", "metadata", "glare/clean", "glare/medium_glare", "glare/strong_glare", "burst"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    manifest = output_root / "metadata.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path", "label_id", "label_name", "split"])
        writer.writeheader()
        writer.writerows(records)

    project_root = project_root or Path(__file__).resolve().parents[1]
    license_audit = _write_license_template(project_root, source_url=RAF_SOURCE_URL)
    generation = {
        "source_format": "split/class directory hierarchy",
        "dataset_root": str(hierarchy_root.resolve()),
        "output_root": str(output_root.resolve()),
        "manifest": str(manifest.resolve()),
        "records": len(records),
        "missing_images": [],
        "split_counts": dict(Counter(row["split"] for row in records)),
        "label_counts": dict(Counter(row["label_name"] for row in records)),
        "license_audit": str(license_audit),
    }
    (output_root / "metadata/manifest_generation.json").write_text(json.dumps(generation, indent=2), encoding="utf-8")
    return generation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raf-root", type=Path, required=True, help="Officially obtained RAF-DB Basic Emotion root")
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/rafdb"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    labels_file = args.labels_file or _find_one(args.raf_root, ("list_patition_label.txt", "list_partition_label.txt"))
    image_root = args.image_root or args.raf_root
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if labels_file is not None and labels_file.is_file():
        result = build_manifest(labels_file, image_root, args.output_root, project_root=project_root)
    else:
        result = build_directory_manifest(image_root, args.output_root, project_root=project_root)
    print(json.dumps({key: value for key, value in result.items() if key != "missing_images"}, indent=2))
    if result["missing_images"]:
        raise SystemExit(f"Missing {len(result['missing_images'])} RAF-DB images; inspect metadata.csv and audit output")


if __name__ == "__main__":
    main()
