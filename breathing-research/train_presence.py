"""Train an exploratory occupied/empty model from processed ESPectre scores.

No extra dependencies, synthetic data, firmware changes, or FSM integration.
Leave-one-recording-out evaluation is not a substitute for held-out days.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from score_research import segments


FEATURES = ["log_mean", "log_median", "log_p95", "log_std", "zero_fraction", "fraction_above_0.001"]


def features(points):
    values = sorted(score for _, score in points)
    stats = [statistics.mean(values), statistics.median(values),
             values[int(0.95 * (len(values) - 1))], statistics.pstdev(values)]
    return [math.log10(max(value, 1e-12)) for value in stats] + [
        sum(value == 0 for value in values) / len(values),
        sum(value > 0.001 for value in values) / len(values)]


def dataset(directory, window_seconds=30, trim_seconds=10):
    rows, skipped = [], []
    for path in sorted(directory.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        session = next((r for r in records if r.get("kind") == "session"), None)
        if session is None:
            raise ValueError(f"Missing session metadata: {path.name}")
        # Only explicit empty posture is negative: empty-with-fan is also empty.
        posture = session.get("posture")
        if posture not in {"empty", "standing", "sitting", "lying_back", "lying_left", "lying_right"}:
            raise ValueError(f"Ambiguous presence label in {path.name}: {posture}")
        sensing = session.get("metadata", {}).get("sensing", {})
        if sensing.get("detector") != "high_accuracy":
            raise ValueError(f"Expected High Accuracy in {path.name}")
        groups = segments(records, max_gap_ms=1000)
        count = 0
        for group in groups:
            # Trim both ends of every contiguous, unchanged sensing segment.
            start = group[0][0] + trim_seconds * 1000
            stop = group[-1][0] - trim_seconds * 1000
            while start + window_seconds * 1000 <= stop:
                end = start + window_seconds * 1000
                points = [(t, s) for t, s in group if start <= t < end]
                if len(points) >= 20:
                    rows.append({"recording": path.name, "session_id": session.get("session_id"),
                                 "posture": posture, "condition": session.get("condition"),
                                 "label": int(posture != "empty"), "start_ms": start,
                                 "end_ms": end, "features": features(points)})
                    count += 1
                start = end
        if not count:
            skipped.append(path.name)
    if not rows:
        raise ValueError("No usable recordings")
    return rows, skipped


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def fit(rows, epochs=800, regularization=0.1, feature_names=None):
    names = FEATURES if feature_names is None else feature_names
    if not names or any(len(row["features"]) != len(names) for row in rows):
        raise ValueError("Feature dimensions do not match model definition")
    counts = {}
    labels = {}
    for row in rows:
        key = row["recording"]
        counts[key] = counts.get(key, 0) + 1
        labels[key] = row["label"]
    class_counts = {label: sum(y == label for y in labels.values()) for label in (0, 1)}
    if not all(class_counts.values()):
        raise ValueError("Both empty and occupied recordings are required")
    # Each class receives half the weight; each recording within a class is equal.
    weights = [0.5 / class_counts[r["label"]] / counts[r["recording"]] for r in rows]
    means = [sum(w * r["features"][j] for w, r in zip(weights, rows)) for j in range(len(names))]
    scales = [max(1e-6, math.sqrt(sum(w * (r["features"][j] - means[j]) ** 2
                                    for w, r in zip(weights, rows)))) for j in range(len(names))]
    xs = [[(x - mean) / scale for x, mean, scale in zip(r["features"], means, scales)] for r in rows]
    coefficients, intercept = [0.0] * len(names), 0.0
    for _ in range(epochs):
        gradients = [regularization * coefficient for coefficient in coefficients]
        bias_gradient = 0.0
        for x, row, weight in zip(xs, rows, weights):
            residual = weight * (sigmoid(intercept + sum(a * b for a, b in zip(coefficients, x))) - row["label"])
            bias_gradient += residual
            for j, value in enumerate(x):
                gradients[j] += residual * value
        intercept -= 0.1 * bias_gradient
        coefficients = [c - 0.1 * g for c, g in zip(coefficients, gradients)]
    return {"feature_names": names, "means": means, "scales": scales,
            "coefficients": coefficients, "intercept": intercept,
            "epochs": epochs, "regularization": regularization}


def predict(model, values):
    xs = [(x - mean) / scale for x, mean, scale in zip(values, model["means"], model["scales"])]
    return sigmoid(model["intercept"] + sum(a * b for a, b in zip(model["coefficients"], xs)))


def evaluate(rows, feature_names=None, group_key="recording"):
    predictions = []
    for name in sorted({r["recording"] for r in rows}):
        held_out = [r for r in rows if r["recording"] == name]
        held_group = held_out[0][group_key]
        train = [r for r in rows if r[group_key] != held_group]
        model = fit(train, feature_names=feature_names)
        scores = [predict(model, r["features"]) for r in held_out]
        score = statistics.mean(scores)
        predictions.append({"recording": name, "label": held_out[0]["label"],
                            "posture": held_out[0]["posture"], "condition": held_out[0]["condition"],
                            "windows": len(scores), "mean_occupied_score": score,
                            "min_occupied_score": min(scores), "max_occupied_score": max(scores),
                            "predicted_label": int(score >= 0.5)})
    tp = sum(r["label"] == 1 and r["predicted_label"] == 1 for r in predictions)
    fn = sum(r["label"] == 1 and r["predicted_label"] == 0 for r in predictions)
    fp = sum(r["label"] == 0 and r["predicted_label"] == 1 for r in predictions)
    tn = sum(r["label"] == 0 and r["predicted_label"] == 0 for r in predictions)
    metrics = {"true_occupied": tp, "missed_occupied": fn, "false_occupied": fp, "true_empty": tn,
               "occupied_recall": tp / (tp + fn), "empty_specificity": tn / (tn + fp),
               "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp))}
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("breathing-research/data"))
    parser.add_argument("--selection", type=Path, help="Exploratory exclusion manifest; full-data results always included")
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory; never overwrites previous runs")
    args = parser.parse_args()
    try:
        rows, skipped = dataset(args.data)
        exclusions = set()
        if args.selection:
            exclusions = {r["recording"] for r in json.loads(args.selection.read_text())["excluded"]}
        known = {r["recording"] for r in rows}
        if exclusions - known:
            raise ValueError(f"Unknown excluded recordings: {sorted(exclusions - known)}")
        selected = [r for r in rows if r["recording"] not in exclusions]
        full_metrics, full_predictions = evaluate(rows)
        selected_metrics, selected_predictions = evaluate(selected) if exclusions else (full_metrics, full_predictions)
        model = fit(selected)
        sessions = sorted({r["session_id"] for r in rows}, key=str)
        model.update(task="occupied_vs_empty", threshold=0.5, calibrated=False,
                     window_seconds=30, step_seconds=30, trim_seconds=10,
                     required_detector="high_accuracy", max_gap_ms=1000,
                     training_recordings=sorted({r["recording"] for r in selected}),
                     status="experimental; positive evidence only; not integrated into FSM")
        report = {"evaluation": "leave-one-recording-out; scores averaged per held-out recording",
                  "warning": "Same-session evaluation does not establish transfer to another day, person, or room. Scores are not calibrated probabilities.",
                  "sessions": sessions, "skipped_recordings": skipped, "excluded_recordings": sorted(exclusions),
                  "full_data": full_metrics, "selected_data_exploratory": selected_metrics,
                  "excluded_recording_predictions_from_selected_model": [
                      {"recording": name, "label": next(r["label"] for r in rows if r["recording"] == name),
                       "mean_occupied_score": statistics.mean(predict(model, r["features"]) for r in rows if r["recording"] == name)}
                      for name in sorted(exclusions)]}
        args.output.mkdir(parents=True, exist_ok=False)
        for name, obj in (("model.json", model), ("evaluation.json", report)):
            (args.output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
        for name, predictions in (("full-predictions.csv", full_predictions), ("selected-predictions.csv", selected_predictions)):
            with (args.output / name).open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
        print(json.dumps(report, indent=2))
        print(f"Saved model and evaluation to {args.output}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
