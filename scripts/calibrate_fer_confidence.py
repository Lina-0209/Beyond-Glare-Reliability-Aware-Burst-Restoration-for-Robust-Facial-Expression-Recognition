#!/usr/bin/env python3
"""Calibrate FER abstention on RAF-DB train and evaluate once on RAF-DB test.

The FER and BERR weights remain frozen. Thresholds are selected only from the
official RAF-DB train split; the official test split is used only for the final
selective-risk report. Existing BERR test records are audited separately because
their stored format contains top-1 confidence but not the top-2 margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_module.config import load_vision_config
from vision_module.emotion import EmotionRecognizer

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "image_path", "label_name", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}: {path}")
    return rows


def _read_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"]: _decode_record(row) for row in csv.DictReader(handle)}


def _decode_record(row: dict) -> dict:
    decoded = dict(row)
    for key in ("confidence", "margin"):
        decoded[key] = float(decoded[key])
    decoded["correct"] = decoded.get("correct", "False").lower() == "true"
    return decoded


def _prediction_record(row: dict, prediction: dict) -> dict:
    ranked = sorted(prediction["probabilities"].items(), key=lambda item: (-item[1], item[0]))
    margin = float(ranked[0][1] - ranked[1][1])
    predicted = str(prediction["emotion"])
    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "label": row["label_name"],
        "prediction": predicted,
        "confidence": float(prediction["confidence"]),
        "margin": margin,
        "correct": predicted == row["label_name"],
    }


def _infer(rows: list[dict], cache_path: Path, recognizer: EmotionRecognizer) -> list[dict]:
    cached = _read_cache(cache_path)
    fields = ["sample_id", "split", "label", "prediction", "confidence", "margin", "correct"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if cache_path.is_file() else "w"
    with cache_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for index, row in enumerate(rows, start=1):
            if row["sample_id"] in cached:
                continue
            with Image.open(row["image_path"]) as image:
                prediction = recognizer.predict(np.asarray(image.convert("RGB"), dtype=np.uint8))
            record = _prediction_record(row, prediction)
            writer.writerow(record)
            handle.flush()
            cached[row["sample_id"]] = record
            if index % 250 == 0 or index == len(rows):
                print({"processed": index, "total": len(rows), "cached": len(cached)}, flush=True)
    return [cached[row["sample_id"]] for row in rows]


def _select_threshold(records: list[dict], target: float, minimum: int) -> dict:
    if not records:
        return {"threshold": 1.0, "accepted": 0, "precision": None, "target_met": False}
    candidates = sorted({0.0, *[float(row["confidence"]) for row in records]})
    choices = []
    for threshold in candidates:
        accepted = [row for row in records if float(row["confidence"]) >= threshold]
        if len(accepted) < minimum:
            continue
        precision = float(np.mean([row["correct"] for row in accepted]))
        choices.append((threshold, len(accepted), precision))
    meeting = [choice for choice in choices if choice[2] >= target]
    if meeting:
        threshold, accepted, precision = max(meeting, key=lambda item: (item[1], -item[0]))
        target_met = True
    elif choices:
        threshold, accepted, precision = max(choices, key=lambda item: (item[2], item[1]))
        target_met = False
    else:
        threshold, accepted, precision, target_met = 1.0, 0, None, False
    return {
        "threshold": float(threshold),
        "accepted": int(accepted),
        "precision": None if precision is None else float(precision),
        "target_met": target_met,
        "available": len(records),
    }


def _select_margin(records: list[dict], thresholds: dict[str, float], target: float, minimum: int) -> dict:
    eligible = [
        row for row in records
        if float(row["confidence"]) >= thresholds.get(row["prediction"], 1.0)
    ]
    candidates = sorted({0.0, *[float(row["margin"]) for row in eligible]})
    choices = []
    for threshold in candidates:
        accepted = [row for row in eligible if float(row["margin"]) >= threshold]
        if len(accepted) < minimum:
            continue
        accuracy = float(np.mean([row["correct"] for row in accepted]))
        if accuracy >= target:
            choices.append((threshold, len(accepted), accuracy))
    if not choices:
        return {"threshold": 0.0, "accepted": len(eligible), "selective_accuracy": (
            float(np.mean([row["correct"] for row in eligible])) if eligible else None
        ), "target_met": False}
    threshold, accepted, accuracy = max(choices, key=lambda item: (item[1], -item[0]))
    return {
        "threshold": float(threshold),
        "accepted": int(accepted),
        "selective_accuracy": float(accuracy),
        "target_met": True,
    }


def _evaluate(records: list[dict], thresholds: dict[str, float], margin_threshold: float) -> dict:
    accepted = [
        row for row in records
        if float(row["confidence"]) >= thresholds.get(row["prediction"], 1.0)
        and float(row["margin"]) >= margin_threshold
    ]
    per_class = {}
    for label in CLASSES:
        class_rows = [row for row in records if row["label"] == label]
        class_accepted = [row for row in accepted if row["label"] == label]
        per_class[label] = {
            "support": len(class_rows),
            "baseline_recall": float(np.mean([row["correct"] for row in class_rows])) if class_rows else None,
            "coverage": len(class_accepted) / len(class_rows) if class_rows else None,
            "selective_accuracy": float(np.mean([row["correct"] for row in class_accepted])) if class_accepted else None,
            "accepted": len(class_accepted),
        }
    per_predicted_class = {}
    for label in CLASSES:
        predicted_rows = [row for row in records if row["prediction"] == label]
        predicted_accepted = [row for row in accepted if row["prediction"] == label]
        per_predicted_class[label] = {
            "predicted": len(predicted_rows),
            "baseline_precision": float(np.mean([row["correct"] for row in predicted_rows])) if predicted_rows else None,
            "coverage": len(predicted_accepted) / len(predicted_rows) if predicted_rows else None,
            "selective_precision": float(np.mean([row["correct"] for row in predicted_accepted])) if predicted_accepted else None,
            "accepted": len(predicted_accepted),
        }
    return {
        "samples": len(records),
        "baseline_accuracy": float(np.mean([row["correct"] for row in records])),
        "coverage": len(accepted) / len(records),
        "accepted": len(accepted),
        "uncertain": len(records) - len(accepted),
        "selective_accuracy": float(np.mean([row["correct"] for row in accepted])) if accepted else None,
        "per_true_class": per_class,
        "per_predicted_class": per_predicted_class,
        "predicted_class_counts": dict(Counter(row["prediction"] for row in records)),
    }






def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/rafdb/metadata.csv")
    parser.add_argument("--vision-config", type=Path, default=ROOT / "configs/vision.yaml")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/fer_confidence_policy_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/confidence_policy")
    parser.add_argument("--target-selective-accuracy", type=float, default=0.80)
    parser.add_argument("--minimum-per-predicted-class", type=int, default=25)
    parser.add_argument("--restoration-drop-threshold", type=float, default=-0.05)
    args = parser.parse_args()

    rows = _load_manifest(args.manifest)
    calibration_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not calibration_rows or not test_rows:
        raise SystemExit("Both RAF-DB train and test rows are required")
    config = load_vision_config(args.vision_config)
    recognizer = EmotionRecognizer(
        model_name=config["fer"]["model"],
        engine=config["fer"].get("engine", "onnx"),
        device=config.get("device", "cuda"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = _infer(rows, args.output_dir / "fer_predictions.csv", recognizer)
    calibration = [row for row in predictions if row["split"] == "train"]
    test = [row for row in predictions if row["split"] == "test"]

    per_predicted_class = {}
    thresholds = {}
    for label in CLASSES:
        selected = [row for row in calibration if row["prediction"] == label]
        result = _select_threshold(
            selected, args.target_selective_accuracy, args.minimum_per_predicted_class
        )
        per_predicted_class[label] = result
        thresholds[label] = float(result["threshold"])
    margin = _select_margin(
        calibration,
        thresholds,
        args.target_selective_accuracy,
        args.minimum_per_predicted_class,
    )
    default_threshold = float(np.median(list(thresholds.values())))
    artifact = {
        "version": 1,
        "default_confidence_threshold": default_threshold,
        "class_thresholds": thresholds,
        "margin_threshold": float(margin["threshold"]),
        "restoration_confidence_drop_threshold": args.restoration_drop_threshold,
        "metadata": {
            "calibration_dataset": "RAF-DB Basic Emotion official train split",
            "calibration_manifest": str(args.manifest.resolve()),
            "calibration_samples": len(calibration),
            "test_samples_not_used_for_threshold_selection": len(test),
            "frozen_fer": f"EmotiEffLib/{config['fer']['model']}",
            "target_selective_accuracy": args.target_selective_accuracy,
            "minimum_per_predicted_class": args.minimum_per_predicted_class,
            "per_predicted_class_calibration": per_predicted_class,
            "margin_calibration": margin,
        },
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    report = {
        "protocol": {
            "threshold_selection_split": "RAF-DB train",
            "frozen_evaluation_split": "RAF-DB test",
            "test_used_for_tuning": False,
            "selection_target": args.target_selective_accuracy,
        },
        "artifact": str(args.artifact.resolve()),
        "calibration": _evaluate(calibration, thresholds, float(margin["threshold"])),
        "test": _evaluate(test, thresholds, float(margin["threshold"])),
    }
    (args.output_dir / "confidence_policy_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    class_rows = []
    for label in CLASSES:
        class_rows.append({
            "class": label,
            "confidence_threshold": thresholds[label],
            **{f"calibration_{key}": value for key, value in report["calibration"]["per_true_class"][label].items()},
            **{f"test_{key}": value for key, value in report["test"]["per_true_class"][label].items()},
        })
    with (args.output_dir / "class_selective_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    print(json.dumps({"artifact": str(args.artifact), "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
