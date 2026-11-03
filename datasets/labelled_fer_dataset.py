"""Manifest adapter for legally supplied labelled FER images."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image
from torch.utils.data import Dataset


class LabelledFERDataset(Dataset):
    """Read the standardized FER manifest without guessing label semantics."""

    def __init__(self, manifest, *, mapping_config, dataset_name="generic", split="test"):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        config = yaml.safe_load(Path(mapping_config).read_text(encoding="utf-8"))
        if dataset_name not in config["datasets"]:
            raise ValueError(f"No explicit label mapping configured for dataset: {dataset_name}")
        mapping = {str(k).strip().lower(): str(v) for k, v in config["datasets"][dataset_name].items()}
        canonical = set(config["canonical_labels"])
        self.excluded = Counter()
        self.source_label_counts = Counter()
        self.records = []
        sample_ids = set()
        with self.manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "split", "sample_id"}
            has_label = reader.fieldnames and ("label" in reader.fieldnames or "label_name" in reader.fieldnames or "label_id" in reader.fieldnames)
            if not reader.fieldnames or not required.issubset(reader.fieldnames) or not has_label:
                raise ValueError("Manifest must contain image_path, split, sample_id, and label/label_name/label_id")
            for row_number, row in enumerate(reader, start=2):
                if row["split"].strip().lower() != split.lower():
                    continue
                raw = (row.get("label_id") or row.get("label") or row.get("label_name") or "").strip().lower()
                self.source_label_counts[raw] += 1
                mapped = mapping.get(raw)
                if mapped is None and row.get("label_name", "").strip().lower() in canonical:
                    mapped = row["label_name"].strip().lower()
                if mapped not in canonical:
                    self.excluded[mapped or f"unmapped:{raw}"] += 1
                    continue
                path = Path(row["image_path"])
                path = path if path.is_absolute() else self.root / path
                if not path.is_file():
                    self.excluded["missing_image"] += 1
                    continue
                sample_id = row["sample_id"].strip()
                if not sample_id:
                    raise ValueError(f"Empty sample_id at manifest row {row_number}")
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample_id in selected split: {sample_id}")
                sample_ids.add(sample_id)
                self.records.append({
                    "sample_id": sample_id,
                    "image_path": path,
                    "label": mapped,
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            rgb = image.convert("RGB").copy()
        return {**record, "image": rgb}
