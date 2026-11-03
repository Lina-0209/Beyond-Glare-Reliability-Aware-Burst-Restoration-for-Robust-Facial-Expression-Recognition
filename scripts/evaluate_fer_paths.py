#!/usr/bin/env python3
"""Portable four-path FER evaluation without private historical result folders."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEVERITIES = ("mild", "medium", "strong")
PATHS = ("glare", "nafnet", "mean_nafnet", "berr", "resize_only", "aligned_mean")
CONDITIONS = ("clean",) + tuple(f"{s}_{p}" for s in SEVERITIES for p in PATHS)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    import numpy as np
    from PIL import Image
    import torch
    from datasets.face_geometry import extract_landmarks_with_fallback
    from datasets.labelled_fer_dataset import LabelledFERDataset
    from vision_module.berr import BERRRestorer
    from vision_module.nafnet import FrozenNAFNetRestorer
    from vision_module.config import load_vision_config, resolve_project_path
    from vision_module.aligned_mean import aligned_mean_rgb
    from vision_module.glare_stress import synthesize_glare_burst
    from vision_module.preprocessing import _resize_rgb

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new, empty output directory; do not mix experiment runs")
    torch.set_num_threads(1)
    cfg = load_vision_config(args.vision_config)
    data = LabelledFERDataset(args.manifest, mapping_config=args.mapping_config,
                             dataset_name=args.dataset_name, split=args.split)
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        requested = sum(r["split"].strip().lower() == args.split.lower() for r in csv.DictReader(stream))
    records = data.records[:args.max_samples] if args.max_samples else data.records
    b = cfg["berr"]
    restorer = BERRRestorer(resolve_project_path(b["checkpoint"]), device=args.device,
                           config=b["config"], image_size=b["image_size"],
                           num_frames=4, face_landmarker_path=resolve_project_path(b["face_landmarker"]))
    nafnet = FrozenNAFNetRestorer(args.nafnet_checkpoint, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete, failures = [], []
    for source in records:
        item = {"sample_id": source["sample_id"], "label": source["label"], "images": {}}
        try:
            with Image.open(source["image_path"]) as image:
                clean = image.convert("RGB")
            points, method = extract_landmarks_with_fallback(np.asarray(clean), restorer.face_landmarker_path)
            if points is None:
                raise ValueError("no_face")
            token = hashlib.sha256(source["sample_id"].encode()).hexdigest()
            folder = args.output_dir / "cache" / token
            folder.mkdir(parents=True)

            def save(name, frame):
                relative = Path("cache") / token / f"{name}.png"
                Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(args.output_dir / relative)
                item["images"][name] = {"path": relative.as_posix(),
                                        "sha256": digest(args.output_dir / relative)}

            save("clean", np.asarray(clean))
            for severity in SEVERITIES:
                stress = synthesize_glare_burst(clean, source["sample_id"], severity, landmarks=points)
                anchor = stress["frames"][0]
                resized = _resize_rgb(anchor, restorer.image_size)
                mean = aligned_mean_rgb(stress["frames"], stress["landmarks"],
                                        image_size=restorer.image_size, device=args.device)
                frames = {"glare": anchor, "resize_only": resized, "aligned_mean": mean,
                          "nafnet": nafnet.restore(resized)["restored_frame"],
                          "mean_nafnet": nafnet.restore(mean)["restored_frame"],
                          "berr": restorer.restore(stress["frames"], landmarks=stress["landmarks"],
                                                   roi_bbox=stress["roi_bbox"])["restored_frame"]}
                for name, frame in frames.items():
                    save(f"{severity}_{name}", frame)
            item["detector_method"] = method
            item["source_sha256"] = digest(source["image_path"])
            complete.append(item)
        except Exception as error:
            failures.append({"sample_id": source["sample_id"], "label": source["label"],
                             "error_type": type(error).__name__, "reason": str(error)})
        print(f"Prepared {len(complete)} complete sources; {len(failures)} failures", flush=True)
    manifest = {"protocol": "portable manifest-based evaluation; not a replay of private historical cohorts",
                "requested_split_rows": requested, "manifest_eligible": len(data),
                "manifest_exclusions": dict(data.excluded), "selected_sources": len(records),
                "max_samples": args.max_samples, "source_manifest_sha256": digest(args.manifest),
                "mapping_sha256": digest(args.mapping_config), "conditions": CONDITIONS,
                "berr_sha256": digest(restorer.checkpoint_path),
                "nafnet_sha256": digest(nafnet.checkpoint_path),
                "geometry": "clean-derived synthetic landmarks; not a detector-under-glare benchmark",
                "fer_rejection": False, "records": complete, "failures": failures}
    write_json(args.output_dir / "cache_manifest.json", manifest)


def infer(args):
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(1)
    cache = json.loads((args.output_dir / "cache_manifest.json").read_text())
    target = args.output_dir / f"predictions_{args.backend}.json"
    if target.exists():
        raise FileExistsError(target)
    if args.backend == "enet":
        from vision_module.emotion import EmotionRecognizer
        backend = EmotionRecognizer(device="cpu")
        def predict(path):
            with Image.open(path) as image:
                return backend.predict(np.asarray(image.convert("RGB")))
    else:
        if not args.poster_root or not args.poster_checkpoint:
            raise ValueError("POSTER requires --poster-root and --poster-checkpoint")
        from vision_module.fer_backends.poster_backend import PosterV2Backend
        backend = PosterV2Backend(args.poster_root, args.poster_checkpoint, device=args.device)
        def predict(path):
            p = backend.predict(path)
            return {"emotion": p.emotion, "confidence": p.emotion_confidence,
                    "probabilities": p.emotion_probs}
    predictions, failures = [], []
    for record in cache["records"]:
        row = {"sample_id": record["sample_id"], "label": record["label"]}
        try:
            for condition in CONDITIONS:
                image = record["images"][condition]
                path = args.output_dir / image["path"]
                if digest(path) != image["sha256"]:
                    raise ValueError(f"Cache hash mismatch: {condition}")
                row[condition] = predict(path)
            predictions.append(row)
        except Exception as error:
            failures.append({"sample_id": record["sample_id"], "error_type": type(error).__name__,
                             "reason": str(error)})
    weights = {}
    if args.backend == "poster":
        for path in [args.poster_checkpoint, args.poster_root / "models/pretrain/ir50.pth",
                     args.poster_root / "models/pretrain/mobilefacenet_model_best.pth.tar"]:
            weights[path.name] = digest(path)
    else:
        weights = {p.name: digest(p) for p in (Path.home() / ".emotiefflib").glob("*enet_b2_7*.onnx")}
    write_json(target, {"backend": args.backend, "cache_manifest_sha256": digest(args.output_dir / "cache_manifest.json"),
                        "weights_sha256": weights, "records": predictions, "failures": failures})


def analyze(args):
    from evaluation.metrics import metrics, paired_bootstrap, mcnemar_exact
    cache_path = args.output_dir / "cache_manifest.json"
    cache = json.loads(cache_path.read_text())
    expected = {r["sample_id"]: r["label"] for r in cache["records"]}
    tables = {}
    failures = {}
    for name in args.backends:
        payload = json.loads((args.output_dir / f"predictions_{name}.json").read_text())
        if payload["cache_manifest_sha256"] != digest(cache_path):
            raise ValueError("Predictions use a different cache manifest")
        rows = payload["records"]
        table = {r["sample_id"]: r for r in rows}
        if len(table) != len(rows) or not set(table) <= set(expected):
            raise ValueError("Duplicate or unexpected source IDs")
        for key, row in table.items():
            if row["label"] != expected[key] or not all(c in row for c in CONDITIONS):
                raise ValueError("Label mismatch or incomplete source prediction")
        tables[name] = table
        failures[name] = payload["failures"]
    common = sorted(set(expected).intersection(*(set(t) for t in tables.values())))
    if not common:
        raise ValueError("No common complete sources; inspect recorded failures")
    report = {"complete_paired_sources": len(common), "source_ids": common,
              "requested_split_rows": cache["requested_split_rows"],
              "coverage_full_split": len(common) / max(cache["requested_split_rows"], 1),
              "max_samples": cache["max_samples"], "manifest_exclusions": cache["manifest_exclusions"],
              "preparation_failures": cache["failures"], "inference_failures": failures,
              "bootstrap_trials": args.bootstrap_trials,
              "uncertainty": "paired source percentile 95% CIs; per-comparison, not multiplicity-adjusted",
              "metrics": {}, "comparisons": {}}
    pairs = (("glare", "berr"), ("nafnet", "berr"), ("mean_nafnet", "berr"),
             ("glare", "nafnet"), ("glare", "mean_nafnet"))
    for backend, table in tables.items():
        rows = [table[key] for key in common]
        report["metrics"][backend] = {c: metrics(rows, c) for c in CONDITIONS}
        report["comparisons"][backend] = {}
        for si, severity in enumerate(SEVERITIES):
            comparisons = {}
            for pi, (first, second) in enumerate(pairs):
                a, b = f"{severity}_{first}", f"{severity}_{second}"
                result = paired_bootstrap(rows, a, b, trials=args.bootstrap_trials,
                                          seed=20261021 + 100 * si + pi)
                for metric in ("accuracy", "macro_f1"):
                    result[metric + "_difference"] = metrics(rows, b)[metric] - metrics(rows, a)[metric]
                result["mcnemar"] = mcnemar_exact(rows, a, b)
                comparisons[f"{second}_minus_{first}"] = result
            report["comparisons"][backend][severity] = comparisons
    # Holm within each comparison family over the selected backends x severities.
    for first, second in pairs:
        key = f"{second}_minus_{first}"
        tests = [(backend, severity, report["comparisons"][backend][severity][key]["mcnemar"])
                 for backend in tables for severity in SEVERITIES]
        running = 0.
        for rank, (_, _, result) in enumerate(sorted(tests, key=lambda t: t[2]["exact_p_value"])):
            running = max(running, min(1., (len(tests) - rank) * result["exact_p_value"]))
            result["holm_p"] = running
            result["holm_family_size"] = len(tests)
    write_json(args.output_dir / "summary.json", report)
    print(f"Analyzed {len(common)} paired sources across {len(tables)} FER backends")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "infer", "analyze"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mapping-config", type=Path, default=ROOT / "configs/rafdb_label_mapping.yaml")
    parser.add_argument("--dataset-name", default="rafdb_basic")
    parser.add_argument("--split", default="test")
    parser.add_argument("--vision-config", type=Path, default=ROOT / "configs/vision.yaml")
    parser.add_argument("--nafnet-checkpoint", type=Path, default=ROOT / "checkpoints/nafnet.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--backend", choices=["enet", "poster"], default="enet")
    parser.add_argument("--poster-root", type=Path)
    parser.add_argument("--poster-checkpoint", type=Path)
    parser.add_argument("--backends", nargs="+", choices=["enet", "poster"], default=["enet", "poster"])
    parser.add_argument("--bootstrap-trials", type=int, default=5000)
    args = parser.parse_args()
    if args.action == "prepare" and args.manifest is None:
        parser.error("prepare requires --manifest")
    if args.bootstrap_trials < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("bootstrap-trials and max-samples must be positive")
    {"prepare": prepare, "infer": infer, "analyze": analyze}[args.action](args)


if __name__ == "__main__":
    main()
