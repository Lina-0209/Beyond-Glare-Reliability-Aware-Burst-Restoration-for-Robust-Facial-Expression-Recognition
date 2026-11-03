#!/usr/bin/env python3
"""Train the end-to-end burst eyeglass reflection removal model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.glare_burst_v3_dataset import CombinedGlareBurstDataset, GlareBurstV3Dataset
from models.berr import BurstEyeglassReflectionRemoval
from models.berr.losses import compute_losses
from models.common.metrics import psnr, ssim
from models.common.roi_utils import crop_and_resize


class SyntheticSmokeDataset(Dataset):
    """Deterministic data for testing code paths without prepared data."""

    def __init__(self, length: int, image_size: int, roi_size: int, frames: int, seed: int) -> None:
        self.length, self.image_size, self.roi_size = length, image_size, roi_size
        self.frames, self.seed = frames, seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        g = torch.Generator().manual_seed(self.seed + index)
        s, t = self.image_size, self.frames
        target = F.avg_pool2d(torch.rand(1, 3, s, s, generator=g), 5, 1, 2)[0]
        yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
        burst, masks, landmarks = [], [], []
        base_landmarks = torch.rand(478, 2, generator=g) * (s - 1)
        for frame in range(t):
            cx = s * (0.42 + 0.05 * frame / max(t - 1, 1))
            cy = s * 0.40
            mask = torch.exp(-(((xx - cx) / (s * 0.14)) ** 2 + ((yy - cy) / (s * 0.07)) ** 2))[None]
            color = torch.tensor([0.9, 0.8, 0.6])[:, None, None]
            burst.append((target * (1 - 0.55 * mask) + color * 0.55 * mask).clamp(0, 1))
            masks.append(mask)
            landmarks.append(base_landmarks + torch.tensor([0.5 * frame, 0.2 * frame]))
        x0, y0, x1, y1 = int(.18*s), int(.25*s), int(.82*s), int(.58*s)
        return {
            "sample_id": f"smoke_{index:04d}",
            "target": target,
            "burst": torch.stack(burst),
            "burst_masks": torch.stack(masks),
            "glare_mask": masks[0],
            "landmarks": torch.stack(landmarks),
            "roi_bbox": torch.tensor([x0, y0, x1, y1]),
            "geometry_heatmap_roi": torch.ones(1, self.roi_size, self.roi_size),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="Warm-start model weights only; optimizer, scheduler, epoch, and step start fresh.",
    )
    parser.add_argument("--device", help="Override automatic device selection, e.g. cuda:0 or cpu.")
    parser.add_argument("--data-root", type=Path, help="Override YAML roots with DATA_ROOT/train and DATA_ROOT/val.")
    parser.add_argument("--max-steps", type=int, help="Stop early; useful for smoke tests.")
    parser.add_argument("--overfit-samples", type=int, help="Train on the first N samples for debugging.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict, split: str, dry_run: bool) -> Dataset:
    data = cfg["dataset"]
    if dry_run:
        return SyntheticSmokeDataset(
            int(cfg.get("dry_run", {}).get("num_samples", 8)),
            int(data["image_size"]), int(data["roi_size"]),
            int(data["burst_length"]), int(cfg["seed"]) + (1000 if split == "val" else 0),
        )
    roots_key = f"{split}_roots"
    common = {
        "image_size": int(data["image_size"]),
        "roi_size": int(data["roi_size"]),
        "burst_length": int(data["burst_length"]),
        "strict_burst_length": bool(data.get("strict_burst_length", True)),
    }
    if roots_key in data:
        configured = data[roots_key]
        if not isinstance(configured, dict):
            raise TypeError(f"dataset.{roots_key} must map source_domain to dataset root")
        roots = {str(domain): resolve(path) for domain, path in configured.items()}
        dataset = CombinedGlareBurstDataset(roots, **common)
    else:
        root = resolve(data[f"{split}_root"])
        dataset = GlareBurstV3Dataset(root, **common)
    if not dataset:
        raise FileNotFoundError(f"No samples found for split '{split}'")
    return dataset


def build_model(cfg: dict) -> BurstEyeglassReflectionRemoval:
    return BurstEyeglassReflectionRemoval(**cfg["model"])


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    cfg: dict,
    best_metric: float,
) -> None:
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step,
        "best_metric": float(best_metric), "config": cfg,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(resolve(path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    return checkpoint


def load_model_weights(model, checkpoint: dict, *, strict: bool = True) -> None:
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        logging.warning(
            "checkpoint compatibility: missing=%s unexpected=%s",
            incompatible.missing_keys, incompatible.unexpected_keys,
        )


def evaluate(model, loader, device, cfg: dict, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move(batch, device)
            out = model(batch["burst"], batch["landmarks"], batch["roi_bbox"], batch["geometry_heatmap_roi"])
            _, losses = compute_losses(model, out, batch, cfg["loss_weights"])
            gt_roi = crop_and_resize(batch["target"], batch["roi_bbox"], model.roi_size)
            metrics = {
                **losses,
                "full_psnr": psnr(out["final"], batch["target"]),
                "full_ssim": ssim(out["final"], batch["target"]),
                "roi_psnr": psnr(out["restored_roi"], gt_roi),
            }
            batch_size = batch["target"].size(0)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            count += batch_size
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg["dataset"].pop("train_roots", None)
        cfg["dataset"].pop("val_roots", None)
        cfg["dataset"]["train_root"] = str(args.data_root / "train")
        cfg["dataset"]["val_root"] = str(args.data_root / "val")
    seed_everything(int(cfg["seed"]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve(cfg["output_root"]) / f"train_{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log")],
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_set = build_dataset(cfg, "train", args.dry_run)
    val_set = build_dataset(cfg, "val", args.dry_run)
    if args.overfit_samples:
        train_set = Subset(train_set, range(min(args.overfit_samples, len(train_set))))
        val_set = Subset(val_set, range(min(args.overfit_samples, len(val_set))))
    train_cfg = cfg["train"]
    train_loader = DataLoader(
        train_set, int(train_cfg["batch_size"]), shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)), pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set, int(cfg["eval"].get("batch_size", 2)), shuffle=False,
        num_workers=int(cfg["eval"].get("num_workers", 2)), pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    start_epoch, step = 1, 0
    best_psnr = float("-inf")
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        load_model_weights(model, checkpoint)
        if "optimizer" not in checkpoint or "scheduler" not in checkpoint:
            raise KeyError("Resume checkpoint must include optimizer and scheduler states")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, step = int(checkpoint["epoch"]) + 1, int(checkpoint["step"])
        best_psnr = float(checkpoint.get("best_metric", float("-inf")))
        if best_psnr == float("-inf"):
            old_metrics = resolve(args.resume).parent / "best_metrics.json"
            if old_metrics.is_file():
                best_psnr = float(
                    json.loads(old_metrics.read_text(encoding="utf-8"))["roi_psnr"]
                )
        logging.info("resumed checkpoint=%s epoch=%d step=%d", args.resume, start_epoch - 1, step)
    elif args.init_checkpoint:
        checkpoint = load_checkpoint(args.init_checkpoint, device)
        load_model_weights(model, checkpoint)
        logging.info("warm-started model weights from %s", args.init_checkpoint)

    stop = False
    logging.info("device=%s train=%d val=%d parameters=%.2fM", device, len(train_set), len(val_set), sum(p.numel() for p in model.parameters()) / 1e6)
    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        model.train()
        running: dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            batch = move(batch, device)
            out = model(batch["burst"], batch["landmarks"], batch["roi_bbox"], batch["geometry_heatmap_roi"])
            loss, stats = compute_losses(model, out, batch, cfg["loss_weights"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            step, batches = step + 1, batches + 1
            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + value
            if step % int(train_cfg.get("log_interval", 20)) == 0:
                logging.info("epoch=%d step=%d loss=%.5f", epoch, step, stats["total"])
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        scheduler.step()
        logging.info("train epoch=%d %s", epoch, " ".join(f"{k}={v/max(batches,1):.5f}" for k, v in running.items()))
        metrics = evaluate(model, val_loader, device, cfg, cfg["eval"].get("max_batches"))
        logging.info("val epoch=%d %s", epoch, " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        current_psnr = float(metrics["roi_psnr"])
        improved = current_psnr > best_psnr
        if improved:
            best_psnr = current_psnr
        save_checkpoint(
            checkpoint_dir / "latest.pt", model, optimizer, scheduler,
            epoch, step, cfg, best_psnr,
        )
        if improved:
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, scheduler,
                epoch, step, cfg, best_psnr,
            )
            (checkpoint_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if stop:
            break

    (run_dir / "run_summary.json").write_text(json.dumps({
        "steps": step, "best_roi_psnr": best_psnr,
        "train_samples": len(train_set), "val_samples": len(val_set),
    }, indent=2), encoding="utf-8")
    logging.info("finished: %s", run_dir)


if __name__ == "__main__":
    main()
