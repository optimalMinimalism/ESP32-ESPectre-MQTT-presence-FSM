"""Display rolling presence evidence from processed ESPectre motion scores.

Read-only SSE client. No raw CSI, device changes, or FSM transitions.
Use --replay to exercise the same window logic with an existing recording.
"""

import argparse
import json
import math
import queue
import struct
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

from score_research import DIRECT_ORIGIN, api_base, positive, read_json, sse_events, utc_now
from train_presence import FEATURES, features, predict


DEFAULT_MODEL = Path(__file__).parent / "reports/presence-extra-trees/model.json"


def load_model(path):
    model = json.loads(path.read_text())
    if model.get("feature_names") != FEATURES or model.get("task") != "occupied_vs_empty":
        raise ValueError("Model task or feature names do not match this detector")
    model_type = model.get("model_type", "logistic_regression")
    if model_type == "tree_ensemble":
        validate_trees(model)
    elif model_type == "logistic_regression":
        for name in ("means", "scales", "coefficients"):
            values = model[name]
            if len(values) != len(FEATURES) or not all(math.isfinite(float(x)) for x in values):
                raise ValueError(f"Invalid model {name}")
        if any(x <= 0 for x in model["scales"]) or not math.isfinite(model["intercept"]):
            raise ValueError("Invalid model scales or intercept")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    for name in ("window_seconds", "max_gap_ms"):
        if not math.isfinite(model[name]) or model[name] <= 0:
            raise ValueError(f"Invalid model {name}")
    if not 0 <= model["threshold"] <= 1:
        raise ValueError("Invalid model threshold")
    if not math.isfinite(model.get("trim_seconds", 0)) or model.get("trim_seconds", 0) < 0:
        raise ValueError("Invalid model settling time")
    return model


def validate_trees(model):
    if not model.get("trees") or model.get("input_precision") != "float32":
        raise ValueError("Model must contain trees and declare float32 input")
    for tree in model["trees"]:
        count = len(tree["left"])
        if not count or any(len(tree[key]) != count for key in ("right", "feature", "threshold", "occupied_probability")):
            raise ValueError("Inconsistent tree dimensions")
        for node in range(count):
            left, right = tree["left"][node], tree["right"][node]
            probability, threshold = tree["occupied_probability"][node], tree["threshold"][node]
            if not math.isfinite(threshold) or not 0 <= probability <= 1:
                raise ValueError("Invalid tree probability or threshold")
            if left == right == -1:
                continue
            if not (isinstance(left, int) and isinstance(right, int) and node < left < count and node < right < count):
                raise ValueError("Invalid or cyclic tree child indices")
            if not isinstance(tree["feature"][node], int) or not 0 <= tree["feature"][node] < len(FEATURES):
                raise ValueError("Invalid tree feature index")


def model_score(model, values):
    if model.get("model_type") != "tree_ensemble":
        return predict(model, values)
    # sklearn tree prediction converts inputs to IEEE float32 before routing.
    values = [struct.unpack("f", struct.pack("f", value))[0] for value in values]
    total = 0.0
    for tree in model["trees"]:
        node = 0
        while tree["left"][node] != -1:
            node = tree["left"][node] if values[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
        total += tree["occupied_probability"][node]
    return total / len(model["trees"])


def sensing_signature(payload):
    keys = ("enabled", "ready", "calibrating", "mode", "derived_events_paused", "detector",
            "threshold", "csi_target_pps", "motion_on_hits", "motion_off_hits", "traffic_generator_mode")
    return tuple(payload.get(key) for key in keys)


class LiveDetector:
    def __init__(self, model, sensing, update_seconds=5):
        self.model = model
        self.update_ms = update_seconds * 1000
        self.sensing = sensing
        self.signature = sensing_signature(sensing)
        self.reset()

    def reset(self):
        self.points = deque()
        self.started_ms = None
        self.last_ms = None
        self.last_received = None
        self.last_evaluation_ms = None
        self.last_status = None

    def status(self, state, reason, **values):
        self.last_status = state
        return {"received_utc": utc_now(), "status": state, "reason": reason, **values}

    def valid_sensing(self):
        return (self.sensing.get("detector") == self.model["required_detector"]
                and self.sensing.get("enabled", True) and self.sensing.get("ready", False)
                and not self.sensing.get("calibrating", False)
                and self.sensing.get("mode", "sensing") == "sensing"
                and not self.sensing.get("derived_events_paused", False))

    def expire(self, now):
        if self.last_received is not None and now - self.last_received > self.model["max_gap_ms"] / 1000:
            self.reset()
            return self.status("UNKNOWN", "motion stream stale; previous evidence cleared")
        return None

    def event(self, event, payload, received):
        if event == "sensing":
            signature = sensing_signature(payload)
            self.sensing = payload
            if signature != self.signature:
                self.signature = signature
                self.reset()
                return self.status("WARMING_UP" if self.valid_sensing() else "UNKNOWN", "sensing changed; history cleared")
            return None
        if event == "fault" or (event == "health" and payload.get("online") is False):
            self.reset()
            self.sensing = {**self.sensing, "ready": False}
            return self.status("UNKNOWN", "device fault or offline; waiting for ready sensing")
        if event != "motion":
            return None
        if not self.valid_sensing():
            if self.last_status != "UNKNOWN":
                return self.status("UNKNOWN", "sensing not ready or detector differs from model")
            return None
        timestamp, score = float(payload["timestamp_ms"]), float(payload["score"])
        if not all(math.isfinite(x) for x in (timestamp, score, received)) or not 0 <= score <= 1:
            raise ValueError("Invalid motion timestamp or score")
        if self.last_ms is not None and (timestamp <= self.last_ms
                or timestamp - self.last_ms > self.model["max_gap_ms"]
                or received < self.last_received
                or received - self.last_received > self.model["max_gap_ms"] / 1000):
            self.reset()
        if self.started_ms is None:
            self.started_ms = timestamp
        self.last_ms, self.last_received = timestamp, received
        elapsed = timestamp - self.started_ms
        settling_ms = self.model.get("trim_seconds", 0) * 1000
        window_ms = self.model["window_seconds"] * 1000
        if elapsed >= settling_ms:
            self.points.append((timestamp, score))
        while self.points and self.points[0][0] < timestamp - window_ms:
            self.points.popleft()
        if elapsed < settling_ms + window_ms:
            if self.last_evaluation_ms is None or timestamp - self.last_evaluation_ms >= self.update_ms:
                self.last_evaluation_ms = timestamp
                return self.status("WARMING_UP", "settling and filling score window",
                                   seconds_remaining=round((settling_ms + window_ms - elapsed) / 1000, 1))
            return None
        if self.last_evaluation_ms is not None and timestamp - self.last_evaluation_ms < self.update_ms:
            return None
        self.last_evaluation_ms = timestamp
        # Same half-open window convention as training: [start, end).
        window = [(t, s) for t, s in self.points if t < timestamp]
        if len(window) < 20:
            return self.status("UNKNOWN", "insufficient samples in score window")
        occupied_score = model_score(self.model, features(window))
        return self.status("OCCUPIED_EVIDENCE" if occupied_score >= self.model["threshold"] else "NO_OCCUPIED_EVIDENCE",
                           "experimental model score; low score does not establish an empty room",
                           occupied_score=occupied_score, threshold=self.model["threshold"],
                           window_start_ms=timestamp - window_ms, window_end_ms=timestamp,
                           samples=len(window))


def display(result, json_output):
    if result is None:
        return
    if json_output:
        print(json.dumps(result, allow_nan=False), flush=True)
    else:
        detail = (f"score={result['occupied_score']:.3f} threshold={result['threshold']:.2f} samples={result['samples']}"
                  if "occupied_score" in result else result["reason"])
        if "seconds_remaining" in result:
            detail += f" ({result['seconds_remaining']:g}s remaining)"
        print(f"{result['status']}: {detail}", flush=True)


def replay(args, model):
    detector = None
    with args.replay.open() as source:
        for line in source:
            record = json.loads(line)
            if record.get("kind") == "session":
                detector = LiveDetector(model, record["metadata"]["sensing"], args.update_seconds)
            elif record.get("kind") == "event":
                if detector is None:
                    raise ValueError("Recording has no initial sensing metadata")
                display(detector.event(record["event"], record["payload"], record["received_monotonic"]), args.json)
    if detector is None:
        raise ValueError("Recording has no session metadata")
    return 0


def live(args, model):
    base = api_base(args.endpoint)
    sensing = read_json(base + "/sensing")
    capabilities = read_json(base + "/capabilities")
    if "motion" not in capabilities.get("events", []):
        raise ValueError("Firmware does not advertise SSE motion events")
    if sensing.get("detector") != model["required_detector"]:
        raise ValueError("Select High Accuracy in the device panel; this reader never changes settings")
    detector = LiveDetector(model, sensing, args.update_seconds)
    messages = queue.Queue(maxsize=2048)
    stop = threading.Event()

    def receive():
        try:
            request = urllib.request.Request(base + "/events", headers={"Accept": "text/event-stream", "Origin": DIRECT_ORIGIN})
            with urllib.request.urlopen(request, timeout=15) as response:
                for event, payload in sse_events(response):
                    if stop.is_set():
                        return
                    messages.put_nowait((event, payload, time.monotonic()))
            messages.put(("connection_error", {"message": "SSE stream closed; restart to reconnect"}, time.monotonic()))
        except Exception as exc:
            messages.put(("connection_error", {"message": str(exc) or type(exc).__name__}, time.monotonic()))

    threading.Thread(target=receive, daemon=True).start()
    deadline = time.monotonic() + args.duration if args.duration else math.inf
    display(detector.status("WARMING_UP" if detector.valid_sensing() else "UNKNOWN", "waiting for processed motion scores"), args.json)
    try:
        while time.monotonic() < deadline:
            display(detector.expire(time.monotonic()), args.json)
            try:
                event, payload, received = messages.get(timeout=0.25)
            except queue.Empty:
                continue
            if event == "connection_error":
                detector.reset()
                display(detector.status("UNKNOWN", payload["message"]), args.json)
                return 1
            if time.monotonic() - received > model["max_gap_ms"] / 1000:
                detector.reset()
                if detector.last_status != "UNKNOWN":
                    display(detector.status("UNKNOWN", "queued events stale; history cleared"), args.json)
                continue
            display(detector.event(event, payload, received), args.json)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--endpoint", help="S3 Direct API address, including port and /espectre/v1")
    source.add_argument("--replay", type=Path, help="Replay recorded events offline, without real-time waits")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--update-seconds", type=positive, default=5, help="Rolling prediction interval; default 5 seconds")
    parser.add_argument("--duration", type=positive, help="Stop live mode after this many seconds; otherwise Ctrl+C")
    parser.add_argument("--json", action="store_true", help="Machine-readable output; still no FSM integration")
    args = parser.parse_args()
    try:
        model = load_model(args.model)
        return replay(args, model) if args.replay else live(args, model)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
