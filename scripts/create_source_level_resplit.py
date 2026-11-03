#!/usr/bin/env python3
"""Create deterministic source-level train/val/test split views using symlinks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def sample_records(input_roots: list[Path]) -> list[dict]:
    records = []
    seen = set()
    for split_root in input_roots:
        for target in sorted((split_root / "target").iterdir()):
            if not target.is_file() or target.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            sample_id = target.stem
            if sample_id in seen:
                raise ValueError(f"Duplicate sample id across input roots: {sample_id}")
            seen.add(sample_id)
            metadata_path = split_root / "metadata" / f"{sample_id}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            source_id = str(metadata.get("source_id") or sample_id.split("_gpt_v", 1)[0])
            records.append({"sample_id": sample_id, "source_id": source_id,
                            "root": split_root, "target": target})
    return records


def stable_order(source_ids: list[str], seed: int) -> list[str]:
    return sorted(source_ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())


def allocate_group(source_ids: list[str], seed: int) -> dict[str, str]:
    ordered = stable_order(source_ids, seed)
    n = len(ordered)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    if n >= 3:
        n_train = min(n - 2, max(1, n_train))
        n_val = min(n - n_train - 1, max(1, n_val))
    result = {}
    for index, source_id in enumerate(ordered):
        result[source_id] = "train" if index < n_train else "val" if index < n_train + n_val else "test"
    return result


def find_artifacts(root: Path, sample_id: str) -> list[tuple[str, Path]]:
    artifacts = []
    for subdir in (
        "target", "mask", "glare_layer", "signed_glare_residual",
        "landmarks", "burst_landmarks", "roi_bbox", "geometry_heatmap_roi", "metadata",
    ):
        directory = root / subdir
        if not directory.exists():
            continue
        matches = sorted(directory.glob(f"{sample_id}.*"))
        if len(matches) > 1:
            raise ValueError(f"Multiple artifacts for {subdir}/{sample_id}: {matches}")
        if matches:
            artifacts.append((subdir, matches[0]))
    for subdir in ("bursts", "burst_masks"):
        candidate = root / subdir / sample_id
        if candidate.exists():
            artifacts.append((subdir, candidate))
    return artifacts


def create_view(domain: str, input_roots: list[Path], output_root: Path, seed: int) -> dict:
    records = sample_records(input_roots)
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source_id"]].append(record)

    # Stratify by variant count so each split contains comparable 1F/4F source mixtures.
    by_variant_count = defaultdict(list)
    for source_id, items in by_source.items():
        by_variant_count[len(items)].append(source_id)
    assignment = {}
    for variant_count, source_ids in sorted(by_variant_count.items()):
        assignment.update(allocate_group(source_ids, seed + variant_count * 1009))

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    for record in records:
        split = assignment[record["source_id"]]
        for subdir, source in find_artifacts(record["root"], record["sample_id"]):
            destination_dir = output_root / split / subdir
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source.name
            os.symlink(source.resolve(), destination, target_is_directory=source.is_dir())

    manifest = {
        "domain": domain,
        "seed": seed,
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "stratification": "source variant count",
        "inputs": [str(path.resolve()) for path in input_roots],
        "splits": {},
        "source_assignment": assignment,
    }
    for split in ("train", "val", "test"):
        source_ids = {source_id for source_id, assigned in assignment.items() if assigned == split}
        split_records = [record for record in records if record["source_id"] in source_ids]
        manifest["splits"][split] = {
            "sources": len(source_ids),
            "bursts": len(split_records),
            "variant_counts": dict(sorted(Counter(len(by_source[s]) for s in source_ids).items())),
        }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    manifest = create_view(args.domain, args.input_roots, args.output_root, args.seed)
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()

