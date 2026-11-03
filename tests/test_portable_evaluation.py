"""Offline tests for the public runner, requiring no faces or downloaded weights."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_fer_paths import CONDITIONS, analyze, digest


class PortableEvaluationTest(unittest.TestCase):
    def make_run(self, root):
        cache = {"records": [{"sample_id": "sample", "label": "happy"}],
                 "requested_split_rows": 2, "max_samples": None,
                 "manifest_exclusions": {}, "failures": [{"sample_id": "no_face"}]}
        (root / "cache_manifest.json").write_text(json.dumps(cache))
        row = {"sample_id": "sample", "label": "happy"}
        for condition in CONDITIONS:
            row[condition] = {"emotion": "happy", "confidence": .9}
        for backend in ("enet", "poster"):
            payload = {"cache_manifest_sha256": digest(root / "cache_manifest.json"),
                       "records": [copy.deepcopy(row)], "failures": []}
            (root / f"predictions_{backend}.json").write_text(json.dumps(payload))
        return argparse.Namespace(output_dir=root, backends=["enet", "poster"], bootstrap_trials=10)

    def test_common_cohort_and_full_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analyze(self.make_run(root))
            report = json.loads((root / "summary.json").read_text())
            self.assertEqual(report["complete_paired_sources"], 1)
            self.assertEqual(report["coverage_full_split"], .5)
            self.assertEqual(report["metrics"]["enet"]["clean"]["accuracy"], 1)
            self.assertEqual(report["comparisons"]["poster"]["strong"]["berr_minus_glare"]["mcnemar"]["holm_family_size"], 6)

    def test_reject_duplicate_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run(root)
            path = root / "predictions_enet.json"
            payload = json.loads(path.read_text())
            payload["records"] *= 2
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                analyze(args)

    def test_reject_changed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run(root)
            path = root / "cache_manifest.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "different cache"):
                analyze(args)

    def test_prepare_and_infer_without_external_assets(self):
        import numpy as np
        import torch
        from PIL import Image
        from datasets.face_geometry import _celeba_aligned_landmarks
        from models.berr import BurstEyeglassReflectionRemoval
        from models.baselines import NAFNetSingle
        from scripts.evaluate_fer_paths import prepare, infer
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (64, 64), (90, 100, 110)).save(root / "face.png")
            manifest = root / "input.csv"
            manifest.write_text("image_path,split,sample_id,label\nface.png,test,valid,4\nface.png,test,no_face,4\n")
            model_cfg = {"roi_size": 32, "width": 4, "encoder_blocks": [1,1,1],
                         "decoder_blocks": [1,1], "reliability_temperature": .15,
                         "use_offset_head": False, "max_residual_flow": 4., "feather_pixels": 3}
            ckpt = root / "tiny.pt"
            torch.save({"model": BurstEyeglassReflectionRemoval(**model_cfg).state_dict(),
                        "config": {"model": model_cfg}}, ckpt)
            naf = root / "naf.pt"
            torch.save({"model": NAFNetSingle(width=4, blocks=(1,1,1,1,1,1,1)).state_dict(),
                        "fixed_configuration": {"width": 4, "blocks": [1,1,1,1,1,1,1]}}, naf)
            cfg = root / "vision.yaml"
            cfg.write_text(json.dumps({"berr": {"checkpoint": str(ckpt), "config": None,
                                                 "image_size": 64, "roi_size": 32, "face_landmarker": None}}))
            args = argparse.Namespace(output_dir=root/"out", vision_config=cfg,
                manifest=manifest, mapping_config=Path("configs/rafdb_label_mapping.yaml"),
                dataset_name="rafdb_basic", split="test", max_samples=None,
                nafnet_checkpoint=naf, device="cpu", backend="enet")
            points = _celeba_aligned_landmarks(64,64)
            with patch("datasets.face_geometry.extract_landmarks_with_fallback",
                       side_effect=[(points,"mock"),(None,"mock_no_face")]):
                prepare(args)
            cache = json.loads((args.output_dir/"cache_manifest.json").read_text())
            self.assertEqual(len(cache["records"]),1)
            self.assertEqual(len(cache["failures"]),1)
            self.assertEqual(len(cache["records"][0]["images"]),19)
            with patch("vision_module.emotion.EmotionRecognizer") as recognizer:
                recognizer.return_value.predict.return_value = {"emotion":"happy", "confidence": .9}
                infer(args)
                self.assertEqual(recognizer.return_value.predict.call_count,19)
            predicted=json.loads((args.output_dir/"predictions_enet.json").read_text())
            self.assertEqual(predicted["failures"],[])
            self.assertEqual(len(predicted["records"]),1)

    def test_poster_torch_load_patch_is_scoped(self):
        import torch
        from vision_module.fer_backends.poster_backend import PosterV2Backend
        instance=PosterV2Backend.__new__(PosterV2Backend)
        instance.repo_root=Path("unused")
        original=torch.load
        with instance._upstream_path_remap():
            self.assertIsNot(torch.load,original)
        self.assertIs(torch.load,original)
