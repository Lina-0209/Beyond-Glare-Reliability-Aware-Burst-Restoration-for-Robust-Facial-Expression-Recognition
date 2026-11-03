"""Dataset for the v3 glare burst pipeline with landmarks and geometry heatmaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Mapping

import torch
from torch.utils.data import ConcatDataset, Dataset

from .transforms import list_image_files, load_image_tensor


class GlareBurstV3Dataset(Dataset):
    """Loads v3 burst data including landmarks, roi_bbox, and geometry_heatmap_roi.

    Expected directory layout under *root*::

        target/           sample_id.png
        mask/             sample_id.png
        glare_layer/      sample_id.png
        bursts/           sample_id/frame_000.png ...
        landmarks/        sample_id.json   (optional: single-frame [K][2] for target)
        burst_landmarks/  sample_id.json   (preferred: list length T of [K][2] per burst frame)
        roi_bbox/         sample_id.json   ([x0,y0,x1,y1])
        geometry_heatmap_roi/  sample_id.png
        metadata/         sample_id.json
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        roi_size: int = 128,
        burst_length: int = 4,
        source_domain: str | None = None,
        strict_burst_length: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.roi_size = roi_size
        self.burst_length = burst_length
        self.source_domain = source_domain or self._infer_source_domain(self.root)
        self.strict_burst_length = bool(strict_burst_length)
        self.target_dir = self.root / "target"
        self.samples = list_image_files(self.target_dir)

    @staticmethod
    def _infer_source_domain(root: Path) -> str:
        name = str(root).lower()
        return "ffhq" if "ffhq" in name else "celeb"

    def __len__(self) -> int:
        return len(self.samples)

    def _load_burst(self, sample_id: str) -> torch.Tensor:
        burst_dir = self.root / "bursts" / sample_id
        frame_paths = list_image_files(burst_dir)
        if self.strict_burst_length and len(frame_paths) != self.burst_length:
            raise ValueError(
                f"Expected exactly {self.burst_length} burst frames for '{sample_id}', "
                f"found {len(frame_paths)} in {burst_dir}"
            )
        frame_paths = frame_paths[: self.burst_length]
        frames: List[torch.Tensor] = [
            load_image_tensor(p, self.image_size) for p in frame_paths
        ]
        if not frames:
            raise FileNotFoundError(f"No burst frames for '{sample_id}' in {burst_dir}")
        while len(frames) < self.burst_length:
            frames.append(frames[-1].clone())
        return torch.stack(frames, dim=0)

    def _load_burst_masks(self, sample_id: str) -> torch.Tensor | None:
        """Load optional per-frame glare masks produced by v3/API generators."""
        mask_dir = self.root / "burst_masks" / sample_id
        paths = list_image_files(mask_dir)
        if not paths:
            return None
        if self.strict_burst_length and len(paths) != self.burst_length:
            raise ValueError(
                f"Expected exactly {self.burst_length} glare masks for '{sample_id}', "
                f"found {len(paths)} in {mask_dir}"
            )
        paths = paths[: self.burst_length]
        masks = [load_image_tensor(path, self.image_size, grayscale=True) for path in paths]
        while len(masks) < self.burst_length:
            masks.append(masks[-1].clone())
        return torch.stack(masks, dim=0)

    def _load_json(self, subdir: str, sample_id: str):
        path = self.root / subdir / f"{sample_id}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _parse_burst_landmarks(self, data, burst_t: int) -> torch.Tensor:
        """Parse burst landmarks from multiple schema variants.

        Supported:
        - list[T][K][2] (native v3)
        - dict with "frames": [{"frame_idx": i, "points_xy": [[x,y], ...]}, ...]
        - list[K][2] (single-frame fallback, expanded to T)
        """
        per_frame = data
        if isinstance(data, dict):
            frames = data.get("frames")
            if isinstance(frames, list):
                def _frame_idx(item):
                    if isinstance(item, dict):
                        return int(item.get("frame_idx", 0))
                    return 0

                ordered = sorted(frames, key=_frame_idx)
                parsed = []
                for fr in ordered:
                    if isinstance(fr, dict):
                        pts = fr.get("points_xy")
                        if pts is None:
                            pts = fr.get("points")
                        if pts is not None:
                            parsed.append(pts)
                    elif isinstance(fr, list):
                        parsed.append(fr)
                per_frame = parsed

        lm = torch.tensor(per_frame, dtype=torch.float32)
        if lm.ndim == 2:
            lm = lm.unsqueeze(0).expand(burst_t, -1, -1).contiguous()
        while lm.size(0) < burst_t:
            lm = torch.cat([lm, lm[-1:].clone()], dim=0)
        if lm.size(0) > burst_t:
            lm = lm[:burst_t]
        return lm

    def _parse_roi_bbox(self, data) -> torch.Tensor | None:
        """Parse ROI bbox from list or dict variants."""
        if data is None:
            return None
        if isinstance(data, dict):
            if "roi_bbox" in data:
                data = data["roi_bbox"]
            elif "roi_bbox_xyxy" in data:
                data = data["roi_bbox_xyxy"]
            else:
                return None
        bbox = torch.tensor(data, dtype=torch.float32)
        if bbox.numel() != 4:
            return None

        # Some datasets store ROI in original (e.g., 1024x1024) pixel space.
        # If bbox extends beyond current image size, scale it down.
        max_coord = float(bbox.max().item())
        if max_coord > float(self.image_size):
            scale = float(self.image_size) / max_coord
            bbox = bbox * scale

        x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
        x0 = max(0, min(x0, self.image_size - 1))
        y0 = max(0, min(y0, self.image_size - 1))
        x1 = max(x0 + 1, min(x1, self.image_size))
        y1 = max(y0 + 1, min(y1, self.image_size))
        return torch.tensor([x0, y0, x1, y1], dtype=torch.long)

    def __getitem__(self, index: int) -> dict:
        target_path = self.samples[index]
        sample_id = target_path.stem

        target = load_image_tensor(target_path, self.image_size)
        burst = self._load_burst(sample_id)
        metadata = self._load_json("metadata", sample_id) or {}
        source_id = str(metadata.get("source_id") or sample_id.split("_gpt_v", 1)[0])

        sample: dict = {
            "sample_id": sample_id,
            "target": target,
            "clean_target": target,
            "burst": burst,
            "source_domain": self.source_domain,
            "source_id": source_id,
            "anchor_frame_index": 0,
        }

        burst_masks = self._load_burst_masks(sample_id)
        if burst_masks is not None:
            sample["burst_masks"] = burst_masks
            sample["glare_masks"] = burst_masks

        mask_path = self.root / "mask" / f"{sample_id}.png"
        if mask_path.exists():
            sample["glare_mask"] = load_image_tensor(mask_path, self.image_size, grayscale=True)

        glare_path = self.root / "glare_layer" / f"{sample_id}.png"
        if glare_path.exists():
            sample["glare_layer"] = load_image_tensor(glare_path, self.image_size)

        heatmap_path = self.root / "geometry_heatmap_roi" / f"{sample_id}.png"
        if heatmap_path.exists():
            sample["geometry_heatmap_roi"] = load_image_tensor(
                heatmap_path, self.roi_size, grayscale=True
            )

        burst_t = burst.shape[0]

        burst_lm_path = self.root / "burst_landmarks" / f"{sample_id}.json"
        if burst_lm_path.exists():
            with burst_lm_path.open("r", encoding="utf-8") as f:
                per_frame = json.load(f)
            sample["landmarks"] = self._parse_burst_landmarks(per_frame, burst_t)
        else:
            lm_data = self._load_json("landmarks", sample_id)
            if lm_data is not None:
                sample["landmarks"] = self._parse_burst_landmarks(lm_data, burst_t)

        bbox_data = self._load_json("roi_bbox", sample_id)
        bbox = self._parse_roi_bbox(bbox_data)
        if bbox is not None:
            sample["roi_bbox"] = bbox

        return sample


class CombinedGlareBurstDataset(ConcatDataset):
    """Expose multiple final burst roots through one domain-aware interface."""

    def __init__(
        self,
        roots: Mapping[str, str | Path],
        image_size: int = 256,
        roi_size: int = 128,
        burst_length: int = 4,
        strict_burst_length: bool = True,
    ) -> None:
        if not roots:
            raise ValueError("At least one dataset root is required")
        datasets = [
            GlareBurstV3Dataset(
                root,
                image_size=image_size,
                roi_size=roi_size,
                burst_length=burst_length,
                source_domain=domain,
                strict_burst_length=strict_burst_length,
            )
            for domain, root in roots.items()
        ]
        empty = [dataset.root for dataset in datasets if len(dataset) == 0]
        if empty:
            raise FileNotFoundError(f"No samples found in dataset roots: {empty}")
        self.roots = {str(domain): Path(root) for domain, root in roots.items()}
        super().__init__(datasets)
