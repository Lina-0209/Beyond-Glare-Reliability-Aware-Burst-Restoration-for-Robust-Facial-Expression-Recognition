#!/usr/bin/env python3
"""Evaluate one frozen BERR checkpoint with repeated 1F, 2F, and full 4F evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import lpips
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.glare_burst_v3_dataset import GlareBurstV3Dataset
from models.common.metrics import psnr, ssim
from models.common.roi_utils import crop_and_resize
from scripts.train_berr import build_model, load_config, load_model_weights, move


MODES = {
    "BERR-1F-repeat": [0, 0, 0, 0],
    "BERR-2F-repeat": [0, 1, 0, 1],
    "BERR-4F": [0, 1, 2, 3],
}


def repeat_evidence(burst: torch.Tensor, landmarks: torch.Tensor, indices: list[int]):
    """Select/repeat frame evidence while preserving the anchor at position zero."""
    index = torch.tensor(indices, device=burst.device)
    return burst.index_select(1, index), landmarks.index_select(1, index)


def evaluate(model, dataset, device, perceptual, batch_size: int) -> dict[str, dict[str, float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    records = {mode: [] for mode in MODES}
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move(raw_batch, device)
            target = batch["target"]
            gt_roi = crop_and_resize(target, batch["roi_bbox"], model.roi_size)
            for mode, indices in MODES.items():
                burst, landmarks = repeat_evidence(batch["burst"], batch["landmarks"], indices)
                output = model(burst, landmarks, batch["roi_bbox"], batch["geometry_heatmap_roi"])
                full_lpips = perceptual(output["final"] * 2 - 1, target * 2 - 1).flatten()
                roi_lpips = perceptual(output["restored_roi"] * 2 - 1, gt_roi * 2 - 1).flatten()
                for sample_index in range(target.size(0)):
                    pred = output["final"][sample_index : sample_index + 1]
                    truth = target[sample_index : sample_index + 1]
                    pred_roi = output["restored_roi"][sample_index : sample_index + 1]
                    truth_roi = gt_roi[sample_index : sample_index + 1]
                    records[mode].append({
                        "full_psnr": psnr(pred, truth),
                        "full_ssim": ssim(pred, truth),
                        "full_lpips": float(full_lpips[sample_index]),
                        "roi_psnr": psnr(pred_roi, truth_roi),
                        "roi_ssim": ssim(pred_roi, truth_roi),
                        "roi_lpips": float(roi_lpips[sample_index]),
                    })
    return {
        mode: {
            key: sum(row[key] for row in rows) / len(rows)
            for key in rows[0]
        }
        for mode, rows in records.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/berr/combined.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--celeb-root", type=Path, required=True)
    parser.add_argument("--ffhq-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/frame_ablation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_model_weights(model, checkpoint)
    model.eval().requires_grad_(False)
    perceptual = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)

    datasets = {
        "celeb": GlareBurstV3Dataset(args.celeb_root, strict_burst_length=True),
        "ffhq": GlareBurstV3Dataset(args.ffhq_root, strict_burst_length=True),
    }
    results = {
        domain: evaluate(model, dataset, device, perceptual, args.batch_size)
        for domain, dataset in datasets.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frame_ablation.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "results": results}, indent=2),
        encoding="utf-8",
    )

    columns = ["Input"]
    for domain in ("Celeb", "FFHQ"):
        for metric in ("Full PSNR", "Full SSIM", "Full LPIPS", "ROI-PSNR", "ROI-SSIM", "ROI-LPIPS"):
            columns.append(f"{domain} {metric}")
    metric_keys = ("full_psnr", "full_ssim", "full_lpips", "roi_psnr", "roi_ssim", "roi_lpips")
    with (args.output_dir / "frame_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for mode in MODES:
            row = {"Input": mode}
            for display, key in (("Celeb", "celeb"), ("FFHQ", "ffhq")):
                for metric_name, metric_key in zip(columns[1:7], metric_keys):
                    suffix = metric_name.split(" ", 1)[1]
                    row[f"{display} {suffix}"] = results[key][mode][metric_key]
            writer.writerow(row)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
