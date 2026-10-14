"""Fixed small-model comparison on processed scores, without deployment.

No parameter search or threshold tuning. Same-day results are exploratory.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

from train_presence import FEATURES, dataset, evaluate, fit


def candidates():
    return {
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=60, max_depth=1, min_samples_leaf=2, learning_rate=0.05, random_state=42),
    }


def weights(rows):
    counts = Counter(r["recording"] for r in rows)
    labels = {r["recording"]: r["label"] for r in rows}
    classes = Counter(labels.values())
    if set(classes) != {0, 1}:
        raise ValueError("Every training fold must contain both empty and occupied recordings")
    return np.array([0.5 / classes[r["label"]] / counts[r["recording"]] for r in rows])


def train(estimator, rows):
    estimator.fit([r["features"] for r in rows], [r["label"] for r in rows], sample_weight=weights(rows))
    return estimator


def export_forest(estimator, rows):
    """Export tree probabilities to JSON for dependency-free live inference."""
    if not isinstance(estimator, (ExtraTreesClassifier, RandomForestClassifier)):
        raise ValueError("Only Extra Trees and Random Forest support this export")
    occupied_index = list(estimator.classes_).index(1)
    trees = []
    for member in estimator.estimators_:
        tree = member.tree_
        values = tree.value[:, 0, :]
        probabilities = values[:, occupied_index] / values.sum(axis=1)
        trees.append({"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                      "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                      "occupied_probability": probabilities.tolist()})
    return {"task": "occupied_vs_empty", "model_type": "tree_ensemble",
            "algorithm": type(estimator).__name__, "feature_names": FEATURES, "trees": trees,
            "window_seconds": 30, "trim_seconds": 10, "threshold": 0.5,
            "required_detector": "high_accuracy", "max_gap_ms": 1000, "calibrated": False,
            "input_precision": "float32", "sklearn_version": sklearn.__version__,
            "training_recordings": sorted({r["recording"] for r in rows}),
            "status": "experimental occupancy evidence; not integrated into FSM"}


def metrics(predictions):
    tp = sum(r["label"] == 1 and r["predicted_label"] == 1 for r in predictions)
    fn = sum(r["label"] == 1 and r["predicted_label"] == 0 for r in predictions)
    fp = sum(r["label"] == 0 and r["predicted_label"] == 1 for r in predictions)
    tn = sum(r["label"] == 0 and r["predicted_label"] == 0 for r in predictions)
    return {"true_occupied": tp, "missed_occupied": fn, "false_occupied": fp, "true_empty": tn,
            "occupied_recall": tp / (tp + fn), "empty_specificity": tn / (tn + fp),
            "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp))}


def tree_evaluation(rows, name, group):
    predictions = []
    for held_group in sorted({r[group] for r in rows}):
        training = [r for r in rows if r[group] != held_group]
        held = [r for r in rows if r[group] == held_group]
        estimator = train(candidates()[name], training)
        class_index = list(estimator.classes_).index(1)
        probabilities = estimator.predict_proba([r["features"] for r in held])[:, class_index]
        for recording in sorted({r["recording"] for r in held}):
            members = [(r, p) for r, p in zip(held, probabilities) if r["recording"] == recording]
            score = float(np.mean([p for _, p in members]))
            first = members[0][0]
            predictions.append({"recording": recording, "label": first["label"], "posture": first["posture"],
                                "condition": first["condition"], "windows": len(members),
                                "mean_occupied_score": score, "min_occupied_score": float(min(p for _, p in members)),
                                "max_occupied_score": float(max(p for _, p in members)), "predicted_label": int(score >= 0.5)})
    return metrics(predictions), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("breathing-research/data"))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows, skipped = dataset(args.data)
        sessions = {r["session_id"] for r in rows}
        if None in sessions:
            raise ValueError("Session IDs are required")
        group = "session_id" if len(sessions) > 1 else "recording"
        excluded = set()
        if args.selection:
            excluded = {r["recording"] for r in json.loads(args.selection.read_text())["excluded"]}
        if excluded - {r["recording"] for r in rows}:
            raise ValueError("Selection contains unknown recordings")
        sets = {"full": rows}
        if excluded:
            sets["selected_exploratory"] = [r for r in rows if r["recording"] not in excluded]
        results, predictions, artifacts = [], {}, {}
        for subset, data in sets.items():
            baseline, output = evaluate(data, group_key=group)
            results.append({"subset": subset, "model": "logistic_baseline", **baseline})
            predictions[f"{subset}-logistic_baseline"] = output
            artifacts[f"{subset}-logistic_baseline"] = fit(data)
            for name in candidates():
                result, output = tree_evaluation(data, name, group)
                results.append({"subset": subset, "model": name, **result})
                predictions[f"{subset}-{name}"] = output
                estimator = train(candidates()[name], data)
                artifacts[f"{subset}-{name}"] = {"estimator": estimator, "feature_names": FEATURES,
                                                 "window_seconds": 30, "trim_seconds": 10, "threshold": 0.5,
                                                 "calibrated": False, "sklearn_version": sklearn.__version__,
                                                 "training_recordings": sorted({r["recording"] for r in data})}
        args.output.mkdir(parents=True, exist_ok=False)
        report = {"holdout_group": group, "sessions": sorted(sessions), "sklearn_version": sklearn.__version__,
                  "skipped_recordings": skipped, "excluded_recordings": sorted(excluded), "results": results,
                  "warning": "Fixed exploratory comparisons, not an independent model-selection test. No automatic promotion; live reader remains unchanged."}
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        with (args.output / "comparison.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
        for name, values in predictions.items():
            with (args.output / f"{name}-predictions.csv").open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        for name, artifact in artifacts.items():
            if "estimator" in artifact:
                joblib.dump(artifact, args.output / f"{name}.joblib")
                if isinstance(artifact["estimator"], (ExtraTreesClassifier, RandomForestClassifier)):
                    subset = name.rsplit("-", 1)[0]
                    export = export_forest(artifact["estimator"], sets[subset])
                    (args.output / f"{name}.json").write_text(json.dumps(export, indent=2) + "\n")
            else:
                (args.output / f"{name}.json").write_text(json.dumps(artifact, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
