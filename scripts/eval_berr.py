#!/usr/bin/env python3
"""Evaluate an end-to-end BERR checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lpips
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_berr import build_dataset, build_model, evaluate, load_config, move, resolve
from models.common.metrics import psnr, ssim
from models.common.roi_utils import crop_and_resize
from models.common.visualization import tensor_to_image


def _save_labeled_grid(
    tensors: list[torch.Tensor], labels: list[str], path: Path, columns: int = 3
) -> None:
    if len(tensors) != len(labels):
        raise ValueError("Each visualization tensor requires one label")
    tile_size, label_height = 256, 24
    rows = (len(tensors) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (tensor, label) in enumerate(zip(tensors, labels)):
        image = tensor_to_image(tensor).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + label_height)
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 6, y + 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_berr"))
    parser.add_argument("--data-root", type=Path, help="Override YAML roots with DATA_ROOT/train and DATA_ROOT/val.")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--save-samples", type=int, default=8)
    parser.add_argument("--device", help="Override automatic device selection, e.g. cuda:0 or cpu.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg["dataset"].pop("train_roots", None)
        cfg["dataset"].pop("val_roots", None)
        cfg["dataset"].pop("test_roots", None)
        cfg["dataset"]["train_root"] = str(args.data_root / "train")
        cfg["dataset"]["val_root"] = str(args.data_root / "val")
        cfg["dataset"]["test_root"] = str(args.data_root / "test")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = build_dataset(cfg, args.split, False)
    loader = DataLoader(dataset, int(cfg["eval"].get("batch_size", 2)), shuffle=False, num_workers=int(cfg["eval"].get("num_workers", 2)))
    model = build_model(cfg).to(device)
    checkpoint = torch.load(resolve(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    perceptual = lpips.LPIPS(net="alex").to(device).eval()

    metrics = evaluate(model, loader, device, cfg, args.max_batches)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample: list[dict] = []
    saved = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move(batch, device)
            out = model(batch["burst"], batch["landmarks"], batch["roi_bbox"], batch["geometry_heatmap_roi"])
            gt_roi = crop_and_resize(batch["target"], batch["roi_bbox"], model.roi_size)
            input_roi = crop_and_resize(batch["burst"][:, 0], batch["roi_bbox"], model.roi_size)
            output_lpips = perceptual(out["final"] * 2.0 - 1.0, batch["target"] * 2.0 - 1.0).flatten()
            input_lpips = perceptual(batch["burst"][:, 0] * 2.0 - 1.0, batch["target"] * 2.0 - 1.0).flatten()
            output_roi_lpips = perceptual(out["restored_roi"] * 2.0 - 1.0, gt_roi * 2.0 - 1.0).flatten()
            input_roi_lpips = perceptual(input_roi * 2.0 - 1.0, gt_roi * 2.0 - 1.0).flatten()
            for index in range(batch["target"].size(0)):
                sample_id = str(batch["sample_id"][index])
                if hasattr(dataset, "root"):
                    sample_root = dataset.root
                else:
                    domain = str(batch["source_domain"][index])
                    sample_root = dataset.roots[domain]
                metadata_path = sample_root / "metadata" / f"{sample_id}.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
                tier = metadata.get("api", {}).get("intensity", {}).get("tier", "unknown")
                record = {
                    "sample_id": sample_id,
                    "intensity_tier": tier,
                    "input_full_psnr": psnr(batch["burst"][index:index+1, 0], batch["target"][index:index+1]),
                    "output_full_psnr": psnr(out["final"][index:index+1], batch["target"][index:index+1]),
                    "input_full_ssim": ssim(batch["burst"][index:index+1, 0], batch["target"][index:index+1]),
                    "output_full_ssim": ssim(out["final"][index:index+1], batch["target"][index:index+1]),
                    "input_lpips": float(input_lpips[index]),
                    "output_lpips": float(output_lpips[index]),
                    "input_roi_psnr": psnr(input_roi[index:index+1], gt_roi[index:index+1]),
                    "output_roi_psnr": psnr(out["restored_roi"][index:index+1], gt_roi[index:index+1]),
                    "input_roi_ssim": ssim(input_roi[index:index+1], gt_roi[index:index+1]),
                    "output_roi_ssim": ssim(out["restored_roi"][index:index+1], gt_roi[index:index+1]),
                    "input_roi_lpips": float(input_roi_lpips[index]),
                    "output_roi_lpips": float(output_roi_lpips[index]),
                }
                per_sample.append(record)
                if saved < args.save_samples:
                    _save_labeled_grid(
                        [
                            batch["burst"][index, 0], batch["target"][index], out["final"][index],
                            out["anchor_roi"][index], out["evidence_roi"][index], out["prior_roi"][index],
                            out["restored_roi"][index], gt_roi[index], out["degradation"][index, 0],
                            out["all_degraded_roi"][index],
                        ],
                        [
                            "Input frame 0", "Ground truth", "Final output",
                            "Aligned anchor ROI", "Fused evidence ROI", "Learned prior ROI",
                            "Restored ROI", "Ground-truth ROI", "Predicted glare map",
                            "All-frames-degraded map",
                        ],
                        args.output_dir / f"{sample_id}.png",
                    )
                    saved += 1

    metric_names = [
        "input_full_psnr", "output_full_psnr", "input_full_ssim", "output_full_ssim",
        "input_lpips", "output_lpips", "input_roi_psnr", "output_roi_psnr",
        "input_roi_ssim", "output_roi_ssim", "input_roi_lpips", "output_roi_lpips",
    ]
    for name in metric_names:
        metrics[f"per_image_mean_{name}"] = sum(item[name] for item in per_sample) / max(len(per_sample), 1)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "per_sample_metrics.json").write_text(
        json.dumps(per_sample, indent=2), encoding="utf-8"
    )
    logging.info("%s", " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
