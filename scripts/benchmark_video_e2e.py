#!/usr/bin/env python3
"""Benchmark video decode-to-state latency with and without geometry caching."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_module import GlareRobustVisionModule


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _run(
    video: Path,
    module: GlareRobustVisionModule,
    runs: int,
    cache_enabled: bool,
    warmup_runs: int,
) -> dict:
    module.restorer.geometry_cache_enabled = cache_enabled
    module.restorer.reset_geometry_cache()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    frames = deque(maxlen=4)
    decode_batch_ms = 0.0
    records = []
    completed_bursts = 0
    while len(records) < runs:
        started = time.perf_counter()
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        decode_batch_ms += (time.perf_counter() - started) * 1000.0
        frames.append(rgb)
        if len(frames) < 4:
            continue
        pipeline_started = time.perf_counter()
        state = module.predict(list(frames))
        pipeline_ms = (time.perf_counter() - pipeline_started) * 1000.0
        if completed_bursts >= warmup_runs:
            records.append({
                "decode_ms": decode_batch_ms,
                "pipeline_ms": pipeline_ms,
                "end_to_end_ms": decode_batch_ms + pipeline_ms,
                "status": state["status"],
                "geometry_cache_hit": bool(state.get("geometry_cache_hit")),
            })
        completed_bursts += 1
        decode_batch_ms = 0.0
        frames.clear()
    capture.release()
    if not records:
        raise RuntimeError("Video produced no complete four-frame burst")
    return {
        "runs": len(records),
        "warmup_runs": warmup_runs,
        "geometry_cache_enabled": cache_enabled,
        "decode": _summary([row["decode_ms"] for row in records]),
        "pipeline": _summary([row["pipeline_ms"] for row in records]),
        "capture_to_state": _summary([row["end_to_end_ms"] for row in records]),
        "geometry_cache_hit_rate": float(np.mean([row["geometry_cache_hit"] for row in records])),
        "status_counts": dict(Counter(row["status"] for row in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "prepared_data/real_videos/IMG_2314.MOV")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vision.yaml")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/runtime/video_e2e.json")
    args = parser.parse_args()
    module = GlareRobustVisionModule(config=args.config, mode="always_restore")
    report = {
        "video": str(args.video.resolve()),
        "scope": "local decode/color-conversion through JSON state; display and file writing excluded",
        "without_geometry_cache": _run(args.video, module, args.runs, False, args.warmup_runs),
        "with_geometry_cache": _run(args.video, module, args.runs, True, args.warmup_runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
