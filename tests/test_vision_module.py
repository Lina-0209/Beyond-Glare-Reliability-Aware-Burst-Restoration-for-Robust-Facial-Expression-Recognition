from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.face_geometry import _celeba_aligned_landmarks
from models.berr import BurstEyeglassReflectionRemoval
from vision_module import BERRRestorer, EmotionRecognizer, GlareRobustVisionModule
from vision_module.preprocessing import InsufficientFramesError, NoFaceError
from demo_vision import save_demo_outputs


class FakeFerBackend:
    idx_to_emotion_class = {
        0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happiness",
        4: "Neutral", 5: "Sadness", 6: "Surprise",
    }

    def predict_emotions(self, image, logits=False):
        assert image.dtype == np.uint8
        return ["Neutral"], np.array([[0.02, 0.01, 0.02, 0.10, 0.75, 0.05, 0.05]])


class FakeRestorer:
    def restore(self, frames, landmarks=None, roi_bbox=None):
        if len(frames) != 4:
            raise InsufficientFramesError("Expected exactly 4 frames")
        image = np.asarray(frames[0], dtype=np.uint8)
        return {
            "restored_frame": image,
            "anchor_frame": image,
            "glare_score": 0.3,
            "glare_removed": True,
            "degradation_maps": np.zeros((4, 8, 8), dtype=np.float32),
            "roi_bbox": [2, 2, 14, 10],
            "latency_ms": {"preprocessing": 1.0, "berr": 2.0},
        }


class NoFaceRestorer:
    def restore(self, frames, landmarks=None, roi_bbox=None):
        raise NoFaceError("No face detected in frame 0")


class VisionModuleTest(unittest.TestCase):
    def setUp(self):
        self.emotion = EmotionRecognizer(device="cpu", backend=FakeFerBackend())
        self.frames = [np.full((16, 16, 3), 100 + i, dtype=np.uint8) for i in range(4)]

    def test_emotion_probabilities_are_valid(self):
        result = self.emotion.predict(self.frames[0])
        self.assertEqual(result["emotion"], "neutral")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)
        self.assertEqual(set(result["probabilities"]), {
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        })

    def test_four_frames_produce_json_state(self):
        module = GlareRobustVisionModule(
            config={"vision": {"emotion_confidence_threshold": 0.60}},
            restorer=FakeRestorer(),
            emotion_recognizer=self.emotion,
        )
        result = module.predict(self.frames)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["emotion"], "neutral")
        self.assertFalse(result["low_confidence"])
        json.dumps(result)

    def test_insufficient_frames_fail_cleanly(self):
        module = GlareRobustVisionModule(
            restorer=FakeRestorer(), emotion_recognizer=self.emotion
        )
        result = module.predict(self.frames[:3])
        self.assertEqual(result["status"], "insufficient_frames")
        self.assertTrue(result["low_confidence"])

    def test_no_face_fails_cleanly(self):
        module = GlareRobustVisionModule(
            restorer=NoFaceRestorer(), emotion_recognizer=self.emotion
        )
        result = module.predict(self.frames)
        self.assertEqual(result["status"], "no_face")

    def test_missing_berr_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            BERRRestorer("definitely_missing_checkpoint.pt", device="cpu")

    def test_real_berr_wrapper_shape_and_glare_score(self):
        model_cfg = {
            "roi_size": 32,
            "width": 4,
            "encoder_blocks": [1, 1, 1],
            "decoder_blocks": [1, 1],
            "reliability_temperature": 0.15,
            "use_offset_head": False,
            "max_residual_flow": 4.0,
            "feather_pixels": 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.pt"
            model = BurstEyeglassReflectionRemoval(**model_cfg)
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "model": model_cfg,
                    "dataset": {"image_size": 64, "roi_size": 32},
                },
            }, checkpoint)
            restorer = BERRRestorer(checkpoint, device="cpu")
            frames = [np.full((64, 64, 3), 90 + i * 5, dtype=np.uint8) for i in range(4)]
            points = _celeba_aligned_landmarks(64, 64)
            result = restorer.restore(
                frames,
                landmarks=np.stack([points] * 4),
                roi_bbox=[10, 12, 54, 40],
            )
            self.assertEqual(result["restored_frame"].shape, (64, 64, 3))
            self.assertEqual(result["degradation_maps"].shape, (4, 32, 32))
            self.assertGreaterEqual(result["glare_score"], 0.0)
            self.assertLessEqual(result["glare_score"], 1.0)

    def test_demo_output_smoke(self):
        module = GlareRobustVisionModule(
            restorer=FakeRestorer(), emotion_recognizer=self.emotion
        )
        result, artifacts = module.predict_with_artifacts(self.frames)
        with tempfile.TemporaryDirectory() as directory:
            save_demo_outputs(Path(directory), result, artifacts)
            self.assertTrue((Path(directory) / "result.json").is_file())
            self.assertTrue((Path(directory) / "comparison.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
