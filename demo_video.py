#!/usr/bin/env python3
"""Run the vision module over non-overlapping or sliding four-frame bursts."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2

from vision_module import GlareRobustVisionModule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/vision.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--stride", type=int, default=4, choices=(1, 4))
    parser.add_argument(
        "--geometry-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse face landmarks/ROI across nearby bursts with motion/age invalidation",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/vision_video.mp4"))
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera if args.camera is not None else str(args.video))
    if not capture.isOpened():
        raise RuntimeError("Could not open video/camera source")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    module = GlareRobustVisionModule(
        berr_checkpoint=args.checkpoint, config=args.config, device=args.device
    )
    if hasattr(module.restorer, "geometry_cache_enabled"):
        module.restorer.geometry_cache_enabled = args.geometry_cache
        module.restorer.reset_geometry_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    states_path = args.output.with_suffix(".jsonl")
    frames = deque(maxlen=4)
    index = 0
    capture_batch_ms = 0.0
    with states_path.open("w", encoding="utf-8") as states:
        while True:
            capture_started = time.perf_counter()
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            capture_batch_ms += (time.perf_counter() - capture_started) * 1000.0
            index += 1
            if len(frames) < 4 or (index - 4) % args.stride:
                continue
            inference_started = time.perf_counter()
            result, artifacts = module.predict_with_artifacts(list(frames))
            result.setdefault("latency_ms", {})["capture_or_decode"] = capture_batch_ms
            result["latency_ms"]["capture_to_state"] = (
                capture_batch_ms + (time.perf_counter() - inference_started) * 1000.0
            )
            capture_batch_ms = 0.0
            states.write(json.dumps(result) + "\n")
            display_rgb = artifacts.get("restored_frame")
            if display_rgb is None:
                display_rgb = frames[0]
            display = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(display, f"Emotion: {result['emotion']}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Confidence: {result['emotion_confidence']:.2f}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Glare: {result['glare_score']:.2f}", (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if writer is None:
                height, width = display.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps / args.stride, (width, height),
                )
            writer.write(display)
            if args.camera is not None:
                cv2.imshow("Glare-Robust Vision", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    capture.release()
    if writer is not None:
        writer.release()
    if args.camera is not None:
        cv2.destroyAllWindows()
    print(f"Saved video: {args.output}")
    print(f"Saved states: {states_path}")


if __name__ == "__main__":
    main()
