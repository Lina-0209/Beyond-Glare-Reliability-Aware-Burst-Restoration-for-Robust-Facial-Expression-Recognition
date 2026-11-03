"""Seven-class metrics and paired comparisons, extracted from the validated runner."""
from collections import Counter
import math
import numpy as np

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def _eligible(rows, condition):
    return [row for row in rows if condition in row]

def metrics(rows, condition):
    selected = _eligible(rows, condition)
    if not selected:
        return {
            "accuracy": None, "macro_f1": None, "mean_confidence": None,
            "num_samples": 0, "per_class_counts": {c: 0 for c in CLASSES},
            "per_class_recall": {c: None for c in CLASSES},
            "confusion": [[0] * len(CLASSES) for _ in CLASSES],
        }
    y = [row["label"] for row in selected]
    p = [row[condition]["emotion"] for row in selected]
    f1, recalls = [], {}
    for label in CLASSES:
        tp = sum(a == label and b == label for a, b in zip(y, p))
        fp = sum(a != label and b == label for a, b in zip(y, p))
        fn = sum(a == label and b != label for a, b in zip(y, p))
        support = sum(a == label for a in y)
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        recalls[label] = tp / support if support else None
    return {
        "accuracy": float(np.mean([a == b for a, b in zip(y, p)])),
        "macro_f1": float(np.mean(f1)),
        "mean_confidence": float(np.mean([row[condition]["confidence"] for row in selected])),
        "num_samples": len(selected),
        "per_class_counts": dict(Counter(y)),
        "per_class_recall": recalls,
        "confusion": [[sum(a == ca and b == cb for a, b in zip(y, p)) for cb in CLASSES] for ca in CLASSES],
    }

def paired_bootstrap(rows, first, second, *, trials=2000, seed=20260920):
    """Paired bootstrap CIs for second minus first."""
    paired = [row for row in rows if first in row and second in row]
    if not paired:
        return {"num_pairs": 0, "accuracy_difference_95ci": None, "macro_f1_difference_95ci": None}
    rng = np.random.RandomState(seed)
    class_index = {label: index for index, label in enumerate(CLASSES)}
    truth = np.array([class_index[row["label"]] for row in paired], dtype=np.int16)
    first_prediction = np.array([class_index[row[first]["emotion"]] for row in paired], dtype=np.int16)
    second_prediction = np.array([class_index[row[second]["emotion"]] for row in paired], dtype=np.int16)
    accuracy_differences, f1_differences = np.empty(trials), np.empty(trials)
    batch_size = 100
    for start in range(0, trials, batch_size):
        stop = min(start + batch_size, trials)
        indices = rng.randint(0, len(paired), size=(stop - start, len(paired)))
        sampled_truth = truth[indices]
        sampled_first = first_prediction[indices]
        sampled_second = second_prediction[indices]
        accuracy_differences[start:stop] = (
            (sampled_second == sampled_truth).mean(axis=1)
            - (sampled_first == sampled_truth).mean(axis=1)
        )
        macro_f1 = []
        for prediction in (sampled_first, sampled_second):
            per_class = []
            for class_id in range(len(CLASSES)):
                tp = ((sampled_truth == class_id) & (prediction == class_id)).sum(axis=1)
                fp = ((sampled_truth != class_id) & (prediction == class_id)).sum(axis=1)
                fn = ((sampled_truth == class_id) & (prediction != class_id)).sum(axis=1)
                denominator = 2 * tp + fp + fn
                per_class.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0))
            macro_f1.append(np.stack(per_class, axis=1).mean(axis=1))
        f1_differences[start:stop] = macro_f1[1] - macro_f1[0]
    return {
        "num_pairs": len(paired),
        "accuracy_difference_95ci": [float(x) for x in np.percentile(accuracy_differences, [2.5, 97.5])],
        "macro_f1_difference_95ci": [float(x) for x in np.percentile(f1_differences, [2.5, 97.5])],
    }

def transition_metrics(rows, glare_key="glare", restored_key="berr"):
    paired = [row for row in rows if all(key in row for key in ("clean", glare_key, restored_key))]
    if not paired:
        return {"num_pairs": 0}
    clean_ok = np.array([row["clean"]["emotion"] == row["label"] for row in paired])
    glare_ok = np.array([row[glare_key]["emotion"] == row["label"] for row in paired])
    restored_ok = np.array([row[restored_key]["emotion"] == row["label"] for row in paired])
    broken = clean_ok & ~glare_ok
    drop = float(clean_ok.mean() - glare_ok.mean())
    gain = float(restored_ok.mean() - glare_ok.mean())
    clean_metrics = metrics(paired, "clean")
    glare_metrics = metrics(paired, glare_key)
    restored_metrics = metrics(paired, restored_key)
    return {
        "num_pairs": len(paired),
        "glare_accuracy_damage": drop,
        "glare_macro_f1_damage": clean_metrics["macro_f1"] - glare_metrics["macro_f1"],
        "restoration_accuracy_gain": gain,
        "restoration_macro_f1_gain": restored_metrics["macro_f1"] - glare_metrics["macro_f1"],
        "recovery_ratio": gain / drop if drop > 0 else None,
        "prediction_flip_rate": float(np.mean([row["clean"]["emotion"] != row[glare_key]["emotion"] for row in paired])),
        "harmful_glare_flip_rate": float(broken.sum() / max(clean_ok.sum(), 1)),
        "correct_recovery_rate": float((restored_ok & broken).sum() / max(broken.sum(), 1)),
        "correct_recovery_count": int((restored_ok & broken).sum()),
        "clean_correct_glare_wrong_count": int(broken.sum()),
    }

def mcnemar_exact(rows, first, second):
    paired = [row for row in rows if first in row and second in row]
    b = sum(row[first]["emotion"] == row["label"] and row[second]["emotion"] != row["label"] for row in paired)
    c = sum(row[first]["emotion"] != row["label"] and row[second]["emotion"] == row["label"] for row in paired)
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
        p_value = min(1.0, 2.0 * tail)
    return {"num_pairs": len(paired), "first_correct_second_wrong": b, "first_wrong_second_correct": c, "exact_p_value": p_value}
