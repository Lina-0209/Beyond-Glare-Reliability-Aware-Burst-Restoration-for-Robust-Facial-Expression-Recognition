#!/usr/bin/env python3
"""Run the vision module on exactly four RGB face frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vision_module import GlareRobustVisionModule


def save_demo_outputs(output_dir: Path, result: dict, artifacts: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if not artifacts:
        return
    anchor = Image.fromarray(artifacts["anchor_frame"])
    anchor.save(output_dir / "input_anchor.jpg", quality=95)
    if artifacts.get("restored_frame") is None:
        return
    restored = Image.fromarray(artifacts["restored_frame"])
    restored.save(output_dir / "restored.jpg", quality=95)
    comparison = Image.new("RGB", (anchor.width + restored.width, max(anchor.height, restored.height)))
    comparison.paste(anchor, (0, 0))
    comparison.paste(restored, (anchor.width, 0))
    comparison.save(output_dir / "comparison.jpg", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", nargs=4, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/vision.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vision_demo"))
    args = parser.parse_args()

    frames = [Image.open(path).convert("RGB") for path in args.frames]
    module = GlareRobustVisionModule(
        berr_checkpoint=args.checkpoint, config=args.config, device=args.device
    )
    result, artifacts = module.predict_with_artifacts(frames)
    save_demo_outputs(args.output_dir, result, artifacts)
    print(f"Status: {result['status']}")
    print(f"Emotion: {result['emotion'].title()}")
    print(f"Emotion confidence: {result['emotion_confidence']:.3f}")
    print(f"Glare score: {result['glare_score']:.3f}")
    print(f"Vision confidence: {result['vision_confidence']:.3f}")
    print(f"Low confidence: {result['low_confidence']}")
    if result.get("error"):
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
