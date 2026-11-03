from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from datasets.glare_burst_v3_dataset import CombinedGlareBurstDataset, GlareBurstV3Dataset
from models.berr import BurstEyeglassReflectionRemoval
from models.berr.losses import compute_losses


def make_batch(batch_size: int = 2, frames: int = 4, image_size: int = 64, roi_size: int = 32) -> dict:
    target = torch.rand(batch_size, 3, image_size, image_size)
    masks = torch.rand(batch_size, frames, 1, image_size, image_size) * 0.6
    burst = target[:, None] * (1 - masks) + masks
    landmarks = torch.rand(batch_size, frames, 32, 2) * (image_size - 1)
    landmarks[:] = landmarks[:, :1]
    bbox = torch.tensor([[12, 16, 52, 44]]).expand(batch_size, -1).clone()
    return {
        "target": target,
        "burst": burst,
        "burst_masks": masks,
        "glare_mask": masks[:, 0],
        "landmarks": landmarks,
        "roi_bbox": bbox,
        "geometry_heatmap_roi": torch.ones(batch_size, 1, roi_size, roi_size),
    }


class BERRModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = BurstEyeglassReflectionRemoval(
            roi_size=32, width=4, encoder_blocks=(1, 1, 1), decoder_blocks=(1, 1),
        )

    def test_forward_shapes_and_reliability(self) -> None:
        batch = make_batch()
        out = self.model(batch["burst"], batch["landmarks"], batch["roi_bbox"], batch["geometry_heatmap_roi"])
        self.assertEqual(out["final"].shape, batch["target"].shape)
        self.assertEqual(out["degradation"].shape, (2, 4, 1, 32, 32))
        self.assertTrue(torch.allclose(out["reliability"].sum(1), torch.ones_like(out["reliability"][:, 0]), atol=1e-5))
        self.assertTrue(torch.isfinite(out["final"]).all())

    def test_joint_loss_backward(self) -> None:
        batch = make_batch(batch_size=1)
        out = self.model(batch["burst"], batch["landmarks"], batch["roi_bbox"], batch["geometry_heatmap_roi"])
        weights = {key: 1.0 for key in ("roi", "full", "ssim", "mask", "visible", "prior", "outside", "alignment", "flow_smooth")}
        loss, _ = compute_losses(self.model, out, batch, weights)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_outside_roi_is_unchanged(self) -> None:
        batch = make_batch(batch_size=1)
        out = self.model(batch["burst"], batch["landmarks"], batch["roi_bbox"], batch["geometry_heatmap_roi"])
        outside = torch.ones_like(out["final"][:, :1])
        outside[:, :, 16:44, 12:52] = 0
        error = ((out["final"] - batch["burst"][:, 0]).abs() * outside).max()
        self.assertLess(float(error.detach()), 1e-7)


class BurstMaskLoaderTest(unittest.TestCase):
    def test_optional_burst_masks_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("target", "bursts/sample", "burst_masks/sample"):
                (root / name).mkdir(parents=True)
            rgb = Image.new("RGB", (16, 16), (100, 120, 140))
            mask = Image.new("L", (16, 16), 128)
            rgb.save(root / "target/sample.png")
            for index in range(4):
                rgb.save(root / f"bursts/sample/frame_{index:03d}.png")
                mask.save(root / f"burst_masks/sample/frame_{index:03d}.png")
            sample = GlareBurstV3Dataset(root, image_size=16, roi_size=8)[0]
            self.assertEqual(sample["burst_masks"].shape, (4, 1, 16, 16))
            self.assertEqual(sample["glare_masks"].shape, (4, 1, 16, 16))
            self.assertIs(sample["target"], sample["clean_target"])

    def test_incomplete_burst_fails_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target").mkdir(parents=True)
            (root / "bursts/sample").mkdir(parents=True)
            rgb = Image.new("RGB", (16, 16), (100, 120, 140))
            rgb.save(root / "target/sample.png")
            for index in range(3):
                rgb.save(root / f"bursts/sample/frame_{index:03d}.png")
            dataset = GlareBurstV3Dataset(root, image_size=16, roi_size=8)
            with self.assertRaisesRegex(ValueError, "exactly 4 burst frames"):
                _ = dataset[0]

    def test_combined_dataset_exposes_domain_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = {}
            rgb = Image.new("RGB", (16, 16), (100, 120, 140))
            for domain in ("celeb", "ffhq"):
                root = base / domain
                roots[domain] = root
                (root / "target").mkdir(parents=True)
                (root / "bursts/sample_gpt_v00").mkdir(parents=True)
                (root / "metadata").mkdir(parents=True)
                rgb.save(root / "target/sample_gpt_v00.png")
                for index in range(4):
                    rgb.save(root / f"bursts/sample_gpt_v00/frame_{index:03d}.png")
                (root / "metadata/sample_gpt_v00.json").write_text(
                    '{"source_id": "source-1"}', encoding="utf-8"
                )
            dataset = CombinedGlareBurstDataset(roots, image_size=16, roi_size=8)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["source_domain"], "celeb")
            self.assertEqual(dataset[1]["source_domain"], "ffhq")
            self.assertEqual(dataset[1]["source_id"], "source-1")


if __name__ == "__main__":
    unittest.main()
