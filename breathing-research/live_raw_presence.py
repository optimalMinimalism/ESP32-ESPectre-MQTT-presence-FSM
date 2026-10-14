"""Run the raw-only presence model through ESPectre's canonical CSI receiver.

This mode pauses the board's ordinary motion events. No FSM updates are sent.
"""

import argparse
import json
import queue
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from live_presence import display, model_score, validate_trees
from raw_presence import RAW_FEATURES, raw_features, selected_slot_indices
from score_research import api_base, positive, read_json, utc_now
from train_presence import predict

ROOT = Path(__file__).resolve().parent.parent / "espectre"
for directory in (ROOT, ROOT / "src/python", ROOT / "src/python/micro_espectre"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools.lib.csi_io import DirectRawCSIReceiver
from tools.ha_traffic_generator_addon.espectre_traffic_generator import ExternalTrafficGenerator


def load_raw_model(path):
    model = json.loads(path.read_text())
    if model.get("task") != "occupied_vs_empty_raw" or model.get("feature_names") != RAW_FEATURES:
        raise ValueError("Expected a raw presence model, not a processed-score model")
    if model.get("model_type") == "tree_ensemble":
        validate_trees(model)
    if model.get("capture_pps") != 100 or model.get("window_seconds") != 30:
        raise ValueError("This reader requires a 100 pps, 30-second raw model")
    return model


class RawDetector:
    def __init__(self, model, update_seconds=5):
        self.model = model
        self.update_us = update_seconds * 1_000_000
        self.reset()

    def reset(self):
        self.points = deque()
        self.origin = None
        self.last_time = None
        self.last_received = None
        self.last_output = None

    def unknown(self, reason):
        self.reset()
        return {"received_utc": utc_now(), "status": "UNKNOWN", "reason": reason}

    def expire(self, now):
        if self.last_received is not None and now - self.last_received > 1:
            return self.unknown("raw stream stale; evidence cleared")
        return None

    def process(self, iq, timestamp, received):
        if self.last_time is not None and (timestamp <= self.last_time or timestamp - self.last_time > 1_000_000
                                           or received - self.last_received > 1 or received < self.last_received):
            self.reset()
        if self.origin is None:
            self.origin = timestamp
        self.last_time, self.last_received = timestamp, received
        if timestamp - self.origin >= 10_000_000:
            self.points.append((timestamp, np.asarray(iq, dtype=np.int8).copy()))
        while self.points and self.points[0][0] < timestamp - 30_000_000:
            self.points.popleft()
        if self.last_output is not None and timestamp - self.last_output < self.update_us:
            return None
        self.last_output = timestamp
        if timestamp - self.origin < 40_000_000:
            return {"received_utc": utc_now(), "status": "WARMING_UP", "reason": "settling and filling raw window",
                    "seconds_remaining": round((40_000_000 - timestamp + self.origin) / 1e6, 1)}
        points = [(t, iq) for t, iq in self.points if t < timestamp]
        times = np.array([t for t, _ in points], dtype=np.float64)
        if not len(times):
            return self.unknown("no raw samples")
        indices = selected_slot_indices(times, self.origin)
        coverage = len(indices) / 3000
        if coverage < self.model["minimum_slot_coverage"]:
            return {"received_utc": utc_now(), "status": "UNKNOWN", "reason": "raw window coverage too low", "coverage": coverage}
        matrix = np.stack([iq for _, iq in points])[indices]
        values = raw_features(matrix, times[indices])
        score = model_score(self.model, values) if self.model.get("model_type") == "tree_ensemble" else predict(self.model, values)
        return {"received_utc": utc_now(), "status": "OCCUPIED_EVIDENCE" if score >= self.model["threshold"] else "NO_OCCUPIED_EVIDENCE",
                "reason": "raw-only experimental model; low score does not establish emptiness",
                "occupied_score": score, "threshold": self.model["threshold"], "samples": len(indices), "coverage": coverage}


def run_live(args, model):
    base = api_base(args.endpoint)
    sensing = read_json(base + "/sensing")
    if not sensing.get("enabled") or sensing.get("mode") != "sensing":
        raise ValueError("Enable ordinary sensing and close other raw sessions first")
    if sensing.get("csi_target_pps") != 100:
        raise ValueError("Set CSI target rate to 100 pps in the panel before this experiment")
    detector = RawDetector(model, args.update_seconds)
    messages = queue.Queue(maxsize=4000)
    receiver = DirectRawCSIReceiver(base, derive_complex=False)
    generator = None
    if sensing.get("traffic_generator_mode") == "external":
        host = socket.gethostbyname(urlsplit(base).hostname)
        generator = ExternalTrafficGenerator([host], rate_pps=100, port=sensing.get("csi_traffic_udp_port", 5555))
    def packet_received(packet):
        if packet.phy_mode != "ht" or packet.ltf_type != "ht-ltf" or packet.channel_width != "20":
            raise ValueError("Raw model requires HT20 HT-LTF CSI")
        messages.put_nowait((packet.iq_raw.copy(), int(packet.device_ticks_us), time.monotonic()))
    receiver.add_callback(packet_received)
    def receive():
        try:
            receiver.run()
            messages.put((None, "raw stream ended", 0))
        except Exception as exc:
            messages.put((None, str(exc), 0))
    deadline = time.monotonic() + args.duration if args.duration else float("inf")
    print("Raw mode pauses the board's normal motion events. Stop other traffic generators before running.", file=sys.stderr)
    try:
        if generator:
            generator.start()
        threading.Thread(target=receive, daemon=True).start()
        while time.monotonic() < deadline:
            display(detector.expire(time.monotonic()), args.json)
            try:
                iq, timestamp, received = messages.get(timeout=0.25)
            except queue.Empty:
                continue
            if iq is None:
                display(detector.unknown(timestamp), args.json)
                return 1
            if time.monotonic() - received > 1:
                display(detector.unknown("queued raw samples stale"), args.json)
                continue
            display(detector.process(iq, timestamp, received), args.json)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        if generator:
            generator.stop()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--endpoint")
    source.add_argument("--replay", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "reports/raw-presence-01/raw-model.json")
    parser.add_argument("--update-seconds", type=positive, default=5)
    parser.add_argument("--duration", type=positive)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        model = load_raw_model(args.model)
        if args.replay:
            detector = RawDetector(model, args.update_seconds)
            with np.load(args.replay, allow_pickle=False) as data:
                for timestamp, iq in zip(data["device_ticks_us"], data["csi_data"]):
                    display(detector.process(iq, int(timestamp), int(timestamp) / 1e6), args.json)
            return 0
        return run_live(args, model)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
