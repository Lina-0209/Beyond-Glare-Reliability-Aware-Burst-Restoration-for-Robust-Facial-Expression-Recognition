"""End-to-end BERR then FER vision pipeline."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .berr import BERRRestorer
from .config import load_vision_config, resolve_project_path
from .confidence import ConfidencePolicy
from .emotion import EmotionRecognizer
from .preprocessing import InsufficientFramesError, NoFaceError, PreprocessingError, _as_rgb_array, _frame_list
from .schemas import VisionState
from .glare_gate import FastGlareGate


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class GlareRobustVisionModule:
    """`four frames -> JSON state` module."""

    def __init__(
        self,
        berr_checkpoint=None,
        fer_checkpoint=None,
        device: str | None = None,
        config=None,
        *,
        restorer=None,
        emotion_recognizer=None,
        mode: str | None = None,
        glare_gate=None,
    ) -> None:
        if fer_checkpoint is not None:
            raise ValueError("The selected EmotiEffLib FER downloads its named pretrained weight")
        self.config = load_vision_config(config)
        selected_device = device or self.config["device"]
        berr_cfg = self.config["berr"]
        checkpoint = berr_checkpoint or berr_cfg.get("checkpoint")
        if restorer is None:
            if checkpoint is None:
                raise ValueError("berr_checkpoint is required")
            restorer = BERRRestorer(
                checkpoint=checkpoint,
                device=selected_device,
                config=berr_cfg.get("config"),
                image_size=int(berr_cfg["image_size"]),
                num_frames=int(berr_cfg["num_frames"]),
                face_landmarker_path=resolve_project_path(berr_cfg.get("face_landmarker")),
                glare_presence_threshold=float(berr_cfg["glare_presence_threshold"]),
                reuse_anchor_landmarks=bool(berr_cfg.get("reuse_anchor_landmarks", True)),
                geometry_cache_enabled=bool(berr_cfg.get("geometry_cache_enabled", False)),
                geometry_cache_max_age=int(berr_cfg.get("geometry_cache_max_age", 4)),
                geometry_cache_motion_threshold=float(
                    berr_cfg.get("geometry_cache_motion_threshold", 0.08)
                ),
            )
        self.restorer = restorer
        self.emotion_recognizer = emotion_recognizer or EmotionRecognizer(
            model_name=self.config["fer"]["model"],
            engine=self.config["fer"].get("engine", "onnx"),
            device=selected_device,
        )
        self.confidence_threshold = float(
            self.config["vision"]["emotion_confidence_threshold"]
        )
        vision_cfg = self.config["vision"]
        policy_path = resolve_project_path(vision_cfg.get("confidence_policy"))
        self.confidence_policy = ConfidencePolicy(
            policy_path,
            default_threshold=self.confidence_threshold,
            margin_threshold=float(vision_cfg.get("emotion_margin_threshold", 0.0)),
            restoration_drop_threshold=float(
                vision_cfg.get("restoration_confidence_drop_threshold", -0.05)
            ),
        )
        self.compare_pre_restore_fer = bool(vision_cfg.get("compare_pre_restore_fer", True))
        self.pre_restore_fer_interval = max(
            1, int(vision_cfg.get("pre_restore_fer_interval", 1))
        )
        self.parallel_gate_fer = bool(vision_cfg.get("parallel_gate_fer", True))
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision")
        self._prediction_count = 0
        self.mode = mode or self.config["vision"].get("mode", self.config["vision"].get("restoration_mode", "always_restore"))
        if self.mode not in {"direct", "always_restore", "auto"}:
            raise ValueError(f"Unsupported restoration mode: {self.mode}")
        self.glare_gate = glare_gate
        if self.mode == "auto" and self.glare_gate is None:
            artifact = resolve_project_path(self.config.get("glare_gate", {}).get("model_path"))
            if artifact is None or not artifact.is_file():
                raise FileNotFoundError(f"Auto mode requires a trained glare gate: {artifact}")
            self.glare_gate = FastGlareGate(artifact, face_landmarker_path=self.restorer.face_landmarker_path)

    def _failure(self, started: float, status: str, error: Exception) -> dict:
        elapsed = (time.perf_counter() - started) * 1000.0
        return VisionState(
            timestamp=_timestamp(),
            emotion="unknown",
            emotion_probs={},
            emotion_confidence=0.0,
            glare_score=0.0,
            glare_removed=False,
            vision_confidence=0.0,
            low_confidence=True,
            processing_time_ms=elapsed,
            status=status,
            restoration_mode=self.mode,
            berr_executed=False,
            error=str(error),
        ).to_dict()

    def predict_with_artifacts(self, frames, landmarks=None, roi_bbox=None) -> tuple[dict, dict]:
        started = time.perf_counter()
        gate_result = None
        routing_decision = None
        pre_emotion_future = None
        self._prediction_count += 1
        compare_pre_restore = self.compare_pre_restore_fer and (
            self.mode == "auto"
            or (self._prediction_count - 1) % self.pre_restore_fer_interval == 0
        )
        try:
            items = _frame_list(frames, int(self.config["berr"]["num_frames"]))
            anchor = items[0]
            if self.mode == "direct":
                routing_decision = "direct"
                restoration = {"restored_frame": anchor, "anchor_frame": anchor,
                               "glare_score": 0.0, "glare_removed": False,
                               "latency_ms": {"preprocessing": (time.perf_counter()-started)*1000.0}}
                berr_executed = False
            elif self.mode == "auto":
                if self.parallel_gate_fer:
                    pre_emotion_future = self._executor.submit(
                        self.emotion_recognizer.predict, anchor
                    )
                anchor_points = None
                if landmarks is not None:
                    raw = landmarks.detach().cpu().numpy() if hasattr(landmarks, "detach") else landmarks
                    anchor_points = raw[0] if getattr(raw, "ndim", 0) == 3 else raw
                gate_started = time.perf_counter()
                gate_result = self.glare_gate.predict(anchor, landmarks=anchor_points, roi_bbox=roi_bbox)
                gate_ms = (time.perf_counter()-gate_started)*1000.0
                routing_decision = gate_result.get(
                    "route", "restore" if gate_result["detected"] else "direct"
                )
                if routing_decision == "restore":
                    restoration = self.restorer.restore(items, landmarks=landmarks,
                        roi_bbox=gate_result["roi_bbox"] if roi_bbox is None else roi_bbox,
                        anchor_landmarks=None if landmarks is not None else gate_result["landmarks"])
                    berr_executed = True
                elif routing_decision == "direct":
                    restoration = {"restored_frame": anchor, "anchor_frame": anchor,
                                   "glare_score": gate_result["probability"], "glare_removed": False,
                                   "roi_bbox": gate_result["roi_bbox"], "latency_ms": {}}
                    berr_executed = False
                elif routing_decision == "abstain":
                    if pre_emotion_future is not None:
                        pre_emotion_future.cancel()
                    total_ms = (time.perf_counter() - started) * 1000.0
                    state = VisionState(
                        timestamp=_timestamp(),
                        emotion="unknown",
                        emotion_probs={},
                        emotion_confidence=0.0,
                        glare_score=float(gate_result["probability"]),
                        glare_removed=False,
                        vision_confidence=0.0,
                        low_confidence=True,
                        processing_time_ms=total_ms,
                        status="abstained",
                        restoration_mode=self.mode,
                        routing_decision="abstain",
                        berr_executed=False,
                        fast_glare_probability=float(gate_result["probability"]),
                        fast_glare_detected=False,
                        speculative_fer_started=pre_emotion_future is not None,
                        latency_ms={"glare_gate": gate_ms, "total": total_ms},
                    )
                    return state.to_dict(), {
                        "anchor_frame": anchor,
                        "restored_frame": None,
                        "degradation_maps": None,
                        "roi_bbox": gate_result["roi_bbox"],
                    }
                else:
                    raise ValueError(f"Unsupported glare-gate route: {routing_decision}")
                restoration.setdefault("latency_ms", {})["glare_gate"] = gate_ms
            else:
                routing_decision = "restore"
                if compare_pre_restore:
                    pre_emotion_future = self._executor.submit(
                        self.emotion_recognizer.predict, anchor
                    )
                restoration = self.restorer.restore(items, landmarks=landmarks, roi_bbox=roi_bbox)
                berr_executed = True
        except InsufficientFramesError as error:
            return self._failure(started, "insufficient_frames", error), {}
        except NoFaceError as error:
            return self._failure(started, "no_face", error), {}
        except (PreprocessingError, ValueError) as error:
            return self._failure(started, "preprocessing_failed", error), {}
        except Exception as error:  # stable inference boundary
            return self._failure(started, "inference_failed", error), {}

        pre_restoration_emotion = None
        try:
            if berr_executed and compare_pre_restore:
                pre_restoration_emotion = (
                    pre_emotion_future.result()
                    if pre_emotion_future is not None
                    else self.emotion_recognizer.predict(anchor)
                )
            if not berr_executed and pre_emotion_future is not None:
                emotion = pre_emotion_future.result()
            else:
                emotion = self.emotion_recognizer.predict(restoration["restored_frame"])
        except Exception as error:
            return self._failure(started, "inference_failed", error), {
                "anchor_frame": restoration.get("anchor_frame"),
                "restored_frame": restoration.get("restored_frame"),
            }

        confidence = float(emotion["confidence"])
        assessment = self.confidence_policy.assess(emotion)
        confidence_delta, confidence_drop = self.confidence_policy.restoration_drop(
            pre_restoration_emotion, emotion
        )
        latency = dict(restoration.get("latency_ms", {}))
        latency["fer"] = float(emotion.get("latency_ms", 0.0))
        if pre_restoration_emotion is not None:
            latency["pre_restore_fer"] = float(
                pre_restoration_emotion.get("latency_ms", 0.0)
            )
        total_ms = (time.perf_counter() - started) * 1000.0
        latency["total"] = total_ms
        low_confidence = bool(assessment["low_confidence"])
        state = VisionState(
            timestamp=_timestamp(),
            emotion="uncertain" if low_confidence else str(emotion["emotion"]),
            emotion_probs={key: float(value) for key, value in emotion["probabilities"].items()},
            emotion_confidence=confidence,
            glare_score=float(restoration["glare_score"]),
            glare_removed=bool(restoration["glare_removed"]),
            vision_confidence=confidence,
            low_confidence=low_confidence,
            processing_time_ms=total_ms,
            status="uncertain" if low_confidence else "ok",
            restoration_mode=self.mode,
            routing_decision=routing_decision,
            berr_executed=berr_executed,
            fast_glare_probability=None if gate_result is None else float(gate_result["probability"]),
            fast_glare_detected=None if gate_result is None else bool(gate_result["detected"]),
            raw_emotion=str(assessment["raw_emotion"]),
            top2_emotions=assessment["top2_emotions"],
            emotion_margin=float(assessment["emotion_margin"]),
            confidence_threshold=float(assessment["confidence_threshold"]),
            margin_threshold=float(assessment["margin_threshold"]),
            uncertainty_reasons=list(assessment["uncertainty_reasons"]),
            pre_restoration_emotion=None if pre_restoration_emotion is None else str(
                pre_restoration_emotion["emotion"]
            ),
            pre_restoration_confidence=None if pre_restoration_emotion is None else float(
                pre_restoration_emotion["confidence"]
            ),
            restoration_confidence_delta=confidence_delta,
            restoration_confidence_drop=confidence_drop,
            pre_restoration_comparison_performed=pre_restoration_emotion is not None,
            speculative_fer_started=(self.mode == "auto" and pre_emotion_future is not None),
            geometry_cache_hit=restoration.get("geometry_cache_hit"),
            landmark_strategy=restoration.get("landmark_strategy"),
            latency_ms=latency,
        )
        artifacts = {
            "anchor_frame": restoration["anchor_frame"],
            "restored_frame": restoration["restored_frame"],
            "degradation_maps": restoration.get("degradation_maps"),
            "roi_bbox": restoration.get("roi_bbox"),
        }
        return state.to_dict(), artifacts

    def predict(self, frames, landmarks=None, roi_bbox=None) -> dict:
        """Return a JSON-serializable state and hide model artifacts."""
        result, _ = self.predict_with_artifacts(frames, landmarks=landmarks, roi_bbox=roi_bbox)
        return result
