"""Train a raw-CSI baseline and compare matched score recordings offline.

Manifest paths are relative to the manifest file. Raw and score captures are
separate trials under matched conditions, not synchronized fusion samples.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_presence import FEATURES, dataset, evaluate, fit


RAW_FEATURES = ["log_profile_temporal_std_mean", "log_profile_temporal_std_p95",
                "log_profile_step_median", "log_profile_step_p95",
                "channel_cv_mean", "channel_cv_std"]
# Centered HT20 bins; also exclude +1, which first_word_invalid can zero.
LIVE_BINS = list(range(4, 32)) + list(range(34, 61))


def raw_features(csi, times_us):
    # Dataset order is [Q0, I0, Q1, I1, ...]. Widen before arithmetic.
    values = csi.astype(np.float64)
    amplitude = np.hypot(values[:, 0::2], values[:, 1::2])[:, LIVE_BINS]
    means = amplitude.mean(axis=1)
    if np.any(means <= 0):
        raise ValueError("Zero live-band amplitude: invalid capture window")
    profile = amplitude / means[:, None]
    temporal_std = profile.std(axis=0)
    valid_pairs = (np.diff(times_us) > 0) & (np.diff(times_us) <= 20000)
    displacement = np.abs(np.diff(profile, axis=0)).mean(axis=1)[valid_pairs]
    if len(displacement) < 20:
        raise ValueError("Insufficient adjacent packet pairs")
    cv = profile.std(axis=1)
    stats = [temporal_std.mean(), np.quantile(temporal_std, 0.95),
             np.median(displacement), np.quantile(displacement, 0.95)]
    return [float(np.log10(max(x, 1e-12))) for x in stats] + [float(cv.mean()), float(cv.std())]


def raw_windows(path, entry):
    with np.load(path, allow_pickle=False) as data:
        csi = data["csi_data"]
        times = data["device_ticks_us"].astype(np.float64)
        label = str(data["label"].item())
        if entry["posture"] == "empty" and label != "empty":
            raise ValueError(f"Manifest says empty but NPZ label is {label}: {path.name}")
        if entry["posture"] != "empty" and label not in {"static_presence", "motion"}:
            raise ValueError(f"Manifest says occupied but NPZ label is {label}: {path.name}")
        if csi.ndim != 2 or csi.shape[1] != 128 or csi.dtype != np.int8:
            raise ValueError("Expected normalized HT20 int8 CSI with 64 complex subcarriers")
        if times.ndim != 1 or len(times) != len(csi) or len(times) < 2 or not np.isfinite(times).all():
            raise ValueError("Valid device_ticks_us required for each packet")
        rate = float(data["requested_pps"].item()) if "requested_pps" in data else float(data["nominal_packet_rate"].item())
        if not 95 <= rate <= 105:
            raise ValueError("This first raw baseline requires captures at 100 pps")
        if "channel_width" in data and not np.all(data["channel_width"].astype(str) == "20"):
            raise ValueError("Only 20 MHz captures are supported")
        if "ltf_type" in data and not np.all(data["ltf_type"].astype(str) == "ht-ltf"):
            raise ValueError("This raw baseline requires HT-LTF")
    breaks = np.flatnonzero((np.diff(times) <= 0) | (np.diff(times) > 1_000_000)) + 1
    rows, quality = [], []
    for indices in np.split(np.arange(len(times)), breaks):
        if len(indices) < 2:
            continue
        local = times[indices]
        # One nearest-center packet per 10 ms slot; never synthesize missing CSI.
        slots = np.floor((local - local[0]) / 10000).astype(np.int64)
        distances = np.abs(local - (local[0] + slots * 10000 + 5000))
        ordering = np.lexsort((distances, slots))
        _, first = np.unique(slots[ordering], return_index=True)
        chosen = indices[ordering[first]]
        start, stop = times[chosen[0]] + 10_000_000, times[chosen[-1]] - 10_000_000
        while start + 30_000_000 <= stop:
            selected = chosen[(times[chosen] >= start) & (times[chosen] < start + 30_000_000)]
            coverage = len(selected) / 3000
            quality.append({"start_us": float(start), "coverage": coverage, "accepted": coverage >= 0.85})
            if coverage >= 0.85:
                rows.append({"recording": path.name, "session_id": entry["session_id"],
                             "experiment_id": entry["experiment_id"], "posture": entry["posture"],
                             "condition": entry["condition"], "label": int(entry["posture"] != "empty"),
                             "start_ms": float(start / 1000), "end_ms": float((start + 30_000_000) / 1000),
                             "features": raw_features(csi[selected], times[selected])})
            start += 30_000_000
    if not rows:
        raise ValueError(f"No usable 30-second windows in {path.name}; inspect duration and slot coverage")
    return rows, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory")
    args = parser.parse_args()
    try:
        entries = json.loads(args.manifest.read_text())["recordings"]
        raw, scores, quality, paired = [], [], {}, []
        score_cache = {}
        used_raw, used_score, used_ids = set(), set(), set()
        for entry in entries:
            if entry["posture"] not in {"empty", "standing", "sitting", "lying_back", "lying_left", "lying_right"}:
                raise ValueError("Use an explicit supported posture for presence labels")
            if entry["experiment_id"] in used_ids:
                raise ValueError("Experiment IDs must be unique")
            used_ids.add(entry["experiment_id"])
            path = (args.manifest.parent / entry["raw_file"]).resolve()
            if path.name in used_raw:
                raise ValueError("Duplicate raw recording basename")
            used_raw.add(path.name)
            rows, quality[path.name] = raw_windows(path, entry)
            raw.extend(rows)
            if entry.get("score_file"):
                score_path = (args.manifest.parent / entry["score_file"]).resolve()
                if score_path.name in used_score:
                    raise ValueError("Score recording cannot be reused for multiple raw trials")
                used_score.add(score_path.name)
                if score_path.parent not in score_cache:
                    score_cache[score_path.parent] = dataset(score_path.parent)[0]
                selected = [dict(r) for r in score_cache[score_path.parent] if r["recording"] == score_path.name]
                if not selected:
                    raise ValueError(f"No score windows in {score_path.name}")
                for row in selected:
                    if row["posture"] != entry["posture"] or row["condition"] != entry["condition"] or row["session_id"] != entry["session_id"]:
                        raise ValueError(f"Score labels/session differ from manifest: {score_path.name}")
                    row["experiment_id"] = entry["experiment_id"]
                scores.extend(selected)
                paired.append((entry["experiment_id"], path.name, score_path.name))
        session_count = len({r["session_id"] for r in raw})
        group = "session_id" if session_count > 1 else "experiment_id"
        metrics, predictions = evaluate(raw, RAW_FEATURES, group)
        model = fit(raw, feature_names=RAW_FEATURES)
        model.update(task="occupied_vs_empty_raw", calibrated=False, threshold=0.5,
                     window_seconds=30, capture_pps=100, trim_seconds=10,
                     live_bins=LIVE_BINS, minimum_slot_coverage=0.85,
                     preprocessing="per-frame live-band mean amplitude normalization",
                     status="experimental raw-only model; not compatible with score live reader")
        report = {"evaluation_group": group, "raw_metrics": metrics, "quality": quality,
                  "warning": "Matched conditions captured separately; no synchronized fusion or calibrated probabilities. Single-day results are exploratory."}
        paired_rows = []
        if scores:
            if len(paired) != len(entries):
                raise ValueError("Supply score_file for every trial, or omit it for every trial, for a matched comparison")
            score_metrics, score_predictions = evaluate(scores, FEATURES, group)
            report["score_metrics"] = score_metrics
            a = {r["recording"]: r for r in predictions}
            b = {r["recording"]: r for r in score_predictions}
            for experiment, raw_name, score_name in paired:
                paired_rows.append({"experiment_id": experiment, "label": a[raw_name]["label"],
                                    "raw_occupied_score": a[raw_name]["mean_occupied_score"],
                                    "processed_occupied_score": b[score_name]["mean_occupied_score"],
                                    "raw_prediction": a[raw_name]["predicted_label"],
                                    "processed_prediction": b[score_name]["predicted_label"]})
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "raw-model.json").write_text(json.dumps(model, indent=2) + "\n")
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        if scores:
            score_model = fit(scores)
            score_model.update(task="occupied_vs_empty", calibrated=False, threshold=0.5,
                               window_seconds=30, step_seconds=30, trim_seconds=10,
                               required_detector="high_accuracy", max_gap_ms=1000,
                               training_recordings=sorted({r["recording"] for r in scores}),
                               status="experimental; positive evidence only; not integrated into FSM")
            (args.output / "score-model.json").write_text(json.dumps(score_model, indent=2) + "\n")
        for name, values in (("raw-predictions.csv", predictions), ("cross-examination.csv", paired_rows)):
            if values:
                with (args.output / name).open("w", newline="") as target:
                    writer = csv.DictWriter(target, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        print(json.dumps({k: v for k, v in report.items() if k != "quality"}, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
