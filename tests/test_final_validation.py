from __future__ import annotations

import csv
import json
import pickle
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.face_geometry import (
    _celeba_aligned_landmarks,
    _extract_landmarks_heuristic,
    _landmarks_from_face_bbox,
)
from datasets.labelled_fer_dataset import LabelledFERDataset
from vision_module import EmotionRecognizer, FrozenNAFNetRestorer, GlareRobustVisionModule
from vision_module.glare_stress import synthesize_glare_burst
from vision_module.glare_gate import FastGlareGate, glare_features
from vision_module.confidence import ConfidencePolicy
from vision_module.preprocessing import prepare_burst
from models.baselines import NAFNetSingle
from scripts.evaluate_frame_ablation import repeat_evidence
from evaluation.metrics import metrics, paired_bootstrap, transition_metrics


class _Fer:
    idx_to_emotion_class = {0:"Anger",1:"Disgust",2:"Fear",3:"Happiness",4:"Neutral",5:"Sadness",6:"Surprise"}
    def predict_emotions(self, image, logits=False):
        return ["Neutral"], np.array([[.01,.01,.01,.01,.92,.02,.02]])


class _Restorer:
    face_landmarker_path = None
    def __init__(self): self.calls = 0
    def restore(self, frames, **kwargs):
        self.calls += 1; anchor=np.asarray(frames[0],dtype=np.uint8)
        return {"restored_frame":anchor,"anchor_frame":anchor,"glare_score":.4,
                "glare_removed":True,"latency_ms":{"berr":1.0}}


class _Gate:
    def __init__(self, detected): self.detected=detected
    def predict(self, frame, **kwargs):
        return {"probability":.9 if self.detected else .1,"detected":self.detected,
                "landmarks":_celeba_aligned_landmarks(32,32),"roi_bbox":[7,7,25,16]}


class _AbstainGate:
    def predict(self, frame, **kwargs):
        return {"probability": .5, "detected": False, "route": "abstain",
                "landmarks": _celeba_aligned_landmarks(32, 32), "roi_bbox": [7, 7, 25, 16]}


class FinalValidationTest(unittest.TestCase):
    def setUp(self):
        self.frames=[np.full((32,32,3),80+i,np.uint8) for i in range(4)]
        self.fer=EmotionRecognizer(device="cpu",backend=_Fer())

    def test_stress_is_deterministic_and_anchor_outside_roi_exact(self):
        clean=np.full((256,256,3),90,np.uint8); points=_celeba_aligned_landmarks(256,256)
        a=synthesize_glare_burst(clean,"same","strong",landmarks=points)
        b=synthesize_glare_burst(clean,"same","strong",landmarks=points)
        self.assertTrue(np.array_equal(a["frames"][0],b["frames"][0]))
        for frame,clean_frame,mask in zip(a["frames"],a["clean_frames"],a["masks"]):
            self.assertTrue(np.array_equal(frame[mask == 0],clean_frame[mask == 0]))
        self.assertEqual(a["frames"][0].shape,clean.shape)

    def test_stress_has_four_varied_frames_and_severity(self):
        clean=np.full((256,256,3),80,np.uint8); points=_celeba_aligned_landmarks(256,256)
        mild=synthesize_glare_burst(clean,"severity","mild",landmarks=points)
        strong=synthesize_glare_burst(clean,"severity","strong",landmarks=points)
        self.assertEqual(len(mild["frames"]),4)
        self.assertFalse(np.array_equal(mild["frames"][1],mild["frames"][2]))
        mild_delta=np.abs(mild["frames"][0].astype(float)-mild["clean_frames"][0]).mean()
        strong_delta=np.abs(strong["frames"][0].astype(float)-strong["clean_frames"][0]).mean()
        self.assertGreater(strong_delta,mild_delta)
        for mask,lens in zip(strong["masks"],strong["lens_masks"]): self.assertTrue(np.all(mask <= lens + 1e-6))

    def test_detector_box_geometry_and_no_face_are_explicit(self):
        points = _landmarks_from_face_bbox((10, 12, 60, 70))
        self.assertEqual(points.shape, (478, 2))
        self.assertTrue(np.all(points[:, 0] >= 10))
        self.assertTrue(np.all(points[:, 0] <= 70))
        self.assertIsNone(_extract_landmarks_heuristic(np.zeros((100, 100, 3), dtype=np.uint8)))
        with patch("datasets.face_geometry._detect_face_bbox_opencv", return_value=((10, 12, 60, 70), "mock")):
            aligned = _extract_landmarks_heuristic(np.zeros((100, 100, 3), dtype=np.uint8))
        expected = _celeba_aligned_landmarks(100, 100)
        self.assertTrue(np.array_equal(aligned, expected))

    def test_label_mapping_excludes_contempt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); Image.fromarray(self.frames[0]).save(root/"a.png"); Image.fromarray(self.frames[0]).save(root/"b.png")
            with (root/"m.csv").open("w",newline="") as f:
                w=csv.writer(f); w.writerow(["image_path","label","split","sample_id"]); w.writerow(["a.png","1","test","a"]); w.writerow(["b.png","7","test","b"])
            ds=LabelledFERDataset(root/"m.csv",mapping_config=Path(__file__).parents[1]/"configs/fer_label_mappings.yaml",dataset_name="affectnet")
            self.assertEqual(len(ds),1); self.assertEqual(ds[0]["label"],"happy"); self.assertEqual(ds.excluded["__exclude_contempt__"],1)

    def test_manifest_requires_unique_sample_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); Image.fromarray(self.frames[0]).save(root/"a.png")
            with (root/"m.csv").open("w",newline="") as f:
                w=csv.writer(f); w.writerow(["image_path","label","split","sample_id"]); w.writerow(["a.png","happy","test","same"]); w.writerow(["a.png","sad","test","same"])
            with self.assertRaisesRegex(ValueError,"Duplicate sample_id"):
                LabelledFERDataset(root/"m.csv",mapping_config=Path(__file__).parents[1]/"configs/fer_label_mappings.yaml",dataset_name="generic")

    def test_modes_and_json_audit_fields(self):
        direct=GlareRobustVisionModule(restorer=_Restorer(),emotion_recognizer=self.fer,mode="direct")
        result=direct.predict(self.frames); self.assertFalse(result["berr_executed"]); self.assertEqual(result["restoration_mode"],"direct")
        restorer=_Restorer(); auto=GlareRobustVisionModule(restorer=restorer,emotion_recognizer=self.fer,mode="auto",glare_gate=_Gate(False))
        result=auto.predict(self.frames); self.assertFalse(result["berr_executed"]); self.assertEqual(restorer.calls,0); self.assertFalse(result["fast_glare_detected"])
        restorer=_Restorer(); auto=GlareRobustVisionModule(restorer=restorer,emotion_recognizer=self.fer,mode="auto",glare_gate=_Gate(True))
        result=auto.predict(self.frames); self.assertTrue(result["berr_executed"]); self.assertEqual(restorer.calls,1)
        restorer=_Restorer(); auto=GlareRobustVisionModule(restorer=restorer,emotion_recognizer=self.fer,mode="auto",glare_gate=_AbstainGate())
        result=auto.predict(self.frames); self.assertEqual(result["status"],"abstained"); self.assertEqual(result["routing_decision"],"abstain"); self.assertEqual(restorer.calls,0)

    def test_short_burst_reuses_anchor_landmarks_once(self):
        points = _celeba_aligned_landmarks(32, 32)
        with patch("vision_module.preprocessing.extract_landmarks", return_value=points) as detector:
            prepared = prepare_burst(self.frames, image_size=32, reuse_anchor_landmarks=True)
        self.assertEqual(detector.call_count, 1)
        self.assertTrue(np.array_equal(prepared.landmarks[0, 0], prepared.landmarks[0, 3]))

    def test_confidence_policy_top2_margin_and_uncertain_state(self):
        with tempfile.TemporaryDirectory() as d:
            artifact = Path(d) / "policy.json"
            artifact.write_text(json.dumps({
                "default_confidence_threshold": .5,
                "class_thresholds": {"neutral": .95},
                "margin_threshold": .1,
                "restoration_confidence_drop_threshold": -.05,
            }))
            policy = ConfidencePolicy(artifact)
            assessed = policy.assess(self.fer.predict(self.frames[0]))
            self.assertTrue(assessed["low_confidence"])
            self.assertEqual(assessed["top2_emotions"][0]["emotion"], "neutral")
            module = GlareRobustVisionModule(
                restorer=_Restorer(), emotion_recognizer=self.fer, mode="direct",
                config={"vision": {"confidence_policy": str(artifact)}},
            )
            result = module.predict(self.frames)
            self.assertEqual(result["status"], "uncertain")
            self.assertEqual(result["emotion"], "uncertain")
            self.assertEqual(result["raw_emotion"], "neutral")
            self.assertGreater(result["emotion_margin"], .8)

    def test_repeat_inputs_and_single_frame_baseline(self):
        import torch
        burst=torch.arange(4).view(1,4,1,1,1).expand(1,4,3,8,8).float()
        landmarks=torch.arange(4).view(1,4,1,1).expand(1,4,2,2).float()
        one,_=repeat_evidence(burst,landmarks,[0,0,0,0]); two,_=repeat_evidence(burst,landmarks,[0,1,0,1])
        self.assertEqual(one[:,1].mean(),0); self.assertEqual(two[:,3].mean(),1)
        output=NAFNetSingle(width=4,blocks=(1,1,1,1,1,1,1))(torch.rand(1,3,32,32))
        self.assertEqual(tuple(output.shape),(1,3,32,32))

    def test_frozen_nafnet_downstream_path(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/"naf.pt"; model=NAFNetSingle(width=4,blocks=(1,1,1,1,1,1,1))
            torch.save({"model":model.state_dict(),"fixed_configuration":{"width":4,"blocks":[1,1,1,1,1,1,1]}},checkpoint)
            restored=FrozenNAFNetRestorer(checkpoint,device="cpu").restore(np.full((31,33,3),80,np.uint8))
            self.assertEqual(restored["restored_frame"].shape,(31,33,3)); self.assertGreaterEqual(restored["latency_ms"],0)

    def test_fast_gate_and_labelled_metrics_smoke(self):
        feature=glare_features(self.frames[0],[4,4,28,20])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"gate.pkl"
            with path.open("wb") as f: pickle.dump({"feature_mean":np.zeros_like(feature),"feature_scale":np.ones_like(feature),"weight":np.zeros_like(feature),"bias":1.0,"threshold":.5},f)
            gate=FastGlareGate(path); result=gate.predict(self.frames[0],landmarks=_celeba_aligned_landmarks(32,32),roi_bbox=[4,4,28,20])
            self.assertTrue(result["detected"])
            self.assertEqual(result["route"], "restore")
        prediction={"emotion":"neutral","confidence":.9}
        evaluated=metrics([{"label":"neutral","clean":prediction,"glare":prediction,"restored":prediction}],"restored")
        self.assertEqual(evaluated["accuracy"],1.0)

    def test_paired_metrics_bootstrap_and_serialization(self):
        correct={"emotion":"happy","confidence":.9}; wrong={"emotion":"sad","confidence":.6}
        rows=[{"label":"happy","clean":correct,"glare":wrong,"nafnet":correct,"berr":correct} for _ in range(4)]
        ci=paired_bootstrap(rows,"glare","berr",trials=100,seed=1)
        self.assertEqual(ci["accuracy_difference_95ci"],[1.0,1.0])
        transition=transition_metrics(rows,"glare","nafnet"); self.assertEqual(transition["recovery_ratio"],1.0)
        self.assertIn("accuracy_difference_95ci",json.loads(json.dumps(ci)))

    def test_mocked_berr_downstream_path(self):
        restorer=_Restorer(); vision=GlareRobustVisionModule(restorer=restorer,emotion_recognizer=self.fer,mode="always_restore")
        state,artifacts=vision.predict_with_artifacts(self.frames,landmarks=np.stack([_celeba_aligned_landmarks(32,32)]*4),roi_bbox=[7,7,25,16])
        self.assertEqual(state["status"],"ok"); self.assertTrue(state["berr_executed"]); self.assertEqual(artifacts["restored_frame"].shape,(32,32,3))


if __name__ == "__main__": unittest.main()
