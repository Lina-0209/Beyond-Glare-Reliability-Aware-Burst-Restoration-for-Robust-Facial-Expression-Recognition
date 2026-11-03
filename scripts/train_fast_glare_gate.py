#!/usr/bin/env python3
"""Train, calibrate, and freeze the three-way FastGlareGate.

The model is fitted on the configured training split.  Both routing thresholds
are selected on validation only.  The configured test split is loaded only
after the artifact has been frozen and is used for a final generalization
report; it does not influence model or threshold selection.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.glare_burst_v3_dataset import CombinedGlareBurstDataset
from vision_module.glare_gate import glare_features


def rgb(tensor):
    return (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def extract(dataset):
    features, labels, source_ids, strengths = [], [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        bbox = sample["roi_bbox"].tolist()
        clean, glare = rgb(sample["target"]), rgb(sample["burst"][0])
        source_id = f"{sample['source_domain']}:{sample['source_id']}"
        features.extend([glare_features(clean, bbox), glare_features(glare, bbox)])
        labels.extend([0, 1])
        source_ids.extend([source_id, source_id])
        x0, y0, x1, y1 = bbox
        strength = float(
            np.abs(
                glare[y0:y1, x0:x1].astype(np.float32)
                - clean[y0:y1, x0:x1].astype(np.float32)
            ).mean()
        )
        strengths.extend([0.0, strength])
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int8),
        source_ids,
        np.asarray(strengths, dtype=np.float32),
    )


def select_thresholds(labels, probabilities, strengths, *, max_restore_fpr, max_strong_direct_rate):
    negative = labels == 0
    positive = labels == 1
    strong_cut = float(np.quantile(strengths[positive], 2 / 3))
    strong = positive & (strengths >= strong_cut)
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))

    restore_candidates = []
    for threshold in candidates:
        fpr = float(np.mean(probabilities[negative] >= threshold))
        recall = float(np.mean(probabilities[positive] >= threshold))
        strong_recall = float(np.mean(probabilities[strong] >= threshold))
        if fpr <= max_restore_fpr:
            restore_candidates.append((strong_recall, recall, -fpr, float(threshold)))
    if not restore_candidates:
        raise RuntimeError("No restore threshold satisfies the configured validation FPR")
    restore_threshold = max(restore_candidates)[-1]

    direct_candidates = []
    for threshold in candidates:
        strong_direct_rate = float(np.mean(probabilities[strong] <= threshold))
        clean_direct_rate = float(np.mean(probabilities[negative] <= threshold))
        if strong_direct_rate <= max_strong_direct_rate and threshold < restore_threshold:
            direct_candidates.append((clean_direct_rate, -strong_direct_rate, float(threshold)))
    direct_threshold = max(direct_candidates)[-1] if direct_candidates else 0.0
    return direct_threshold, restore_threshold, strong_cut


def routing_metrics(labels, probabilities, strengths, direct_threshold, restore_threshold, strong_cut):
    routes = np.where(
        probabilities <= direct_threshold,
        "direct",
        np.where(probabilities >= restore_threshold, "restore", "abstain"),
    )
    negative, positive = labels == 0, labels == 1
    strong = positive & (strengths >= strong_cut)
    result = {
        "num_examples": int(len(labels)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "direct_threshold": float(direct_threshold),
        "restore_threshold": float(restore_threshold),
    }
    for name, mask in (("clean", negative), ("glare", positive), ("strong", strong)):
        result[name] = {
            route: float(np.mean(routes[mask] == route))
            for route in ("direct", "abstain", "restore")
        }
    restore = routes == "restore"
    result["restore_precision"] = float(np.mean(labels[restore] == 1)) if restore.any() else None
    result["restore_false_positive_rate"] = float(np.mean(restore[negative]))
    result["glare_restore_recall"] = float(np.mean(restore[positive]))
    result["strong_restore_recall"] = float(np.mean(restore[strong]))
    result["non_abstain_coverage"] = float(np.mean(routes != "abstain"))
    result["abstain_rate"] = float(np.mean(routes == "abstain"))
    return result


def inference_latency(model, features, *, runs=1000):
    subset = features[: min(len(features), 32)]
    for _ in range(20):
        model.predict_proba(subset)
    values = []
    for _ in range(runs):
        started = time.perf_counter()
        model.predict_proba(subset)
        values.append((time.perf_counter() - started) * 1000.0 / len(subset))
    return {
        "batch_size": int(len(subset)),
        "runs": runs,
        "mean_ms_per_roi": float(np.mean(values)),
        "p95_ms_per_roi": float(np.percentile(values, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/berr/combined.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/glare_gate")
    parser.add_argument("--max-restore-fpr", type=float, default=0.10)
    parser.add_argument("--max-strong-direct-rate", type=float, default=0.05)
    args = parser.parse_args()

    dataset_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["dataset"]
    roots = lambda key: {name: ROOT / path for name, path in dataset_config[key].items()}
    train = CombinedGlareBurstDataset(roots("train_roots"))
    validation = CombinedGlareBurstDataset(roots("val_roots"))
    test = CombinedGlareBurstDataset(roots("test_roots"))
    train_x, train_y, train_ids, train_strength = extract(train)
    val_x, val_y, val_ids, val_strength = extract(validation)
    test_x, test_y, test_ids, test_strength = extract(test)
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise RuntimeError("Source leakage among gate train/validation/test splits")

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=20260926,
    )
    model.fit(train_x, train_y)
    validation_probabilities = model.predict_proba(val_x)[:, 1]
    direct_threshold, restore_threshold, strong_cut = select_thresholds(
        val_y,
        validation_probabilities,
        val_strength,
        max_restore_fpr=args.max_restore_fpr,
        max_strong_direct_rate=args.max_strong_direct_rate,
    )
    validation_metrics = routing_metrics(
        val_y,
        validation_probabilities,
        val_strength,
        direct_threshold,
        restore_threshold,
        strong_cut,
    )

    payload = {
        "model_type": "hist_gradient_boosting",
        "feature_version": "enhanced_static_v2",
        "model": model,
        "direct_threshold": direct_threshold,
        "restore_threshold": restore_threshold,
        "threshold": restore_threshold,
        "validation_strong_cut": strong_cut,
        "calibration_split": "validation",
        "seed": 20260926,
        "validation_metrics": validation_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_artifact = args.output_dir / "fast_glare_gate.pkl"
    with frozen_artifact.open("wb") as handle:
        pickle.dump(payload, handle)
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    with (artifact_dir / "fast_glare_gate_v2.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    # Test is evaluated only after the calibrated artifact has been serialized.
    test_probabilities = model.predict_proba(test_x)[:, 1]
    test_metrics = routing_metrics(
        test_y,
        test_probabilities,
        test_strength,
        direct_threshold,
        restore_threshold,
        strong_cut,
    )
    report = {
        "protocol": {
            "model_fit_split": "train",
            "threshold_selection_split": "validation",
            "test_role": "frozen final evaluation only",
            "selection_rule": "maximize strong-glare restore recall subject to validation restore FPR <= 0.10; maximize clean direct rate subject to strong-glare direct rate <= 0.05",
            "feature_version": "enhanced_static_v2",
            "model_type": "hist_gradient_boosting",
            "train_sources": len(set(train_ids)),
            "validation_sources": len(set(val_ids)),
            "test_sources": len(set(test_ids)),
            "source_overlap": 0,
        },
        "validation": validation_metrics,
        "test_frozen": test_metrics,
        "model_probability_latency": inference_latency(model, test_x),
    }
    (args.output_dir / "gate_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
