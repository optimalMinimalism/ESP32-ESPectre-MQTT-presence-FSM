"""Record processed ESPectre motion events and inspect periodicity offline.

Uses only the Python standard library. Run with --help for commands.
Periodicity is an exploratory measurement, not a breathing classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import statistics
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def api_base(endpoint):
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
        if ":" not in endpoint.split("//", 1)[1]:
            endpoint += ":62587"
    return endpoint.rstrip("/") if endpoint.rstrip("/").endswith("/espectre/v1") else endpoint.rstrip("/") + "/espectre/v1"


def read_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def sse_events(lines):
    """Decode SSE framing, including multiline data and comments."""
    event, data = "message", []
    for line in lines:
        line = line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].lstrip(" ")
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))


def record(args):
    base = api_base(args.endpoint)
    metadata = {name: read_json(base + "/" + name) for name in ("device", "sensing", "capabilities")}
    if "motion" not in metadata["capabilities"].get("events", []):
        raise ValueError("This firmware does not advertise motion events over SSE")
    if not metadata["sensing"].get("ready"):
        raise ValueError("Sensing is not ready; check the device panel before recording")
    if metadata["sensing"].get("detector") != "high_accuracy":
        raise ValueError("Select High Accuracy in the device panel before this experiment")
    messages = queue.Queue(maxsize=2048)
    stop = threading.Event()

    def receive():
        try:
            request = urllib.request.Request(base + "/events", headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(request, timeout=15) as response:
                for event, payload in sse_events(response):
                    if stop.is_set():
                        return
                    messages.put_nowait({"kind": "event", "event": event, "payload": payload,
                                        "received_utc": utc_now(), "received_monotonic": time.monotonic()})
            messages.put_nowait({"kind": "error", "message": "SSE connection closed; events are not replayed"})
        except Exception as exc:
            # If the queue is full, report failure as soon as space is available.
            messages.put({"kind": "error", "message": str(exc) or type(exc).__name__})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count, interrupted, failure = 0, False, None
    with args.output.open("x", encoding="utf-8") as output:
        def write(value):
            output.write(json.dumps(value, allow_nan=False) + "\n")
            output.flush()
        write({"kind": "session", "created_utc": utc_now(), "endpoint": base,
               "participant": args.participant, "session_id": args.session_id,
               "posture": args.posture, "condition": args.condition,
               "position": args.position, "notes": args.notes, "metadata": metadata})
        threading.Thread(target=receive, daemon=True).start()
        deadline = time.monotonic() + args.duration
        print(f"Recording to {args.output} for {args.duration:g} seconds", flush=True)
        try:
            while time.monotonic() < deadline:
                try:
                    item = messages.get(timeout=min(1, max(0.001, deadline - time.monotonic())))
                except queue.Empty:
                    continue
                write(item)
                if item["kind"] == "error":
                    failure = item["message"]
                    break
                if item["event"] == "motion":
                    count += 1
        except KeyboardInterrupt:
            interrupted = True
        finally:
            stop.set()
            write({"kind": "end", "motion_events": count, "interrupted": interrupted,
                   "error": failure, "finished_utc": utc_now()})
    print(f"Saved {count} motion events. " + (f"Capture error: {failure}" if failure else ""))
    return 1 if failure or not count else 0


def segments(records, max_gap_ms):
    """Never bridge device resets, network gaps, or sensing configuration changes."""
    result, current = [], []
    previous_received = None
    for record in records:
        if record.get("kind") != "event":
            continue
        if record.get("event") == "sensing":
            if current:
                result.append(current)
            current, previous_received = [], None
            continue
        if record.get("event") != "motion":
            continue
        payload = record["payload"]
        timestamp, score = float(payload["timestamp_ms"]), float(payload["score"])
        received = float(record["received_monotonic"])
        if not all(math.isfinite(x) for x in (timestamp, score, received)) or not 0 <= score <= 1:
            raise ValueError("Invalid timestamp or score in motion event")
        gap = current and (timestamp <= current[-1][0] or timestamp - current[-1][0] > max_gap_ms
                           or received - previous_received > max_gap_ms / 1000
                           or received < previous_received)
        if gap:
            result.append(current)
            current = []
        current.append((timestamp, score))
        previous_received = received
    if current:
        result.append(current)
    return result


def periodicity(samples, low_hz=0.1, high_hz=0.6):
    """Fit candidate sinusoids to detrended scores at actual device timestamps.

    The explained fraction measures fit strength, not biological specificity.
    No interpolation over missing events and no trained breathing threshold.
    """
    times = [(t - samples[0][0]) / 1000 for t, _ in samples]
    scores = [s for _, s in samples]
    mean_t, mean_y = statistics.mean(times), statistics.mean(scores)
    centered_t = [t - mean_t for t in times]
    denominator = sum(t * t for t in centered_t)
    slope = sum(t * (y - mean_y) for t, y in zip(centered_t, scores)) / denominator if denominator else 0
    ys = [y - mean_y - slope * t for t, y in zip(centered_t, scores)]
    energy = sum(y * y for y in ys)
    duration = times[-1]
    result = {"mean_score": mean_y, "score_std": statistics.pstdev(scores),
              "candidate_frequency_hz": None, "candidate_cycles_per_minute": None,
              "sinusoid_explained_fraction": 0.0}
    if energy < 1e-12 or duration <= 0:
        return result
    best, best_hz = 0.0, None
    step = 1 / (4 * duration)
    for index in range(int((high_hz - low_hz) / step) + 1):
        hz = low_hz + index * step
        sin = [math.sin(2 * math.pi * hz * t) for t in times]
        cos = [math.cos(2 * math.pi * hz * t) for t in times]
        # Remove mean/trend from basis as well: compare nested least-squares fits.
        for basis in (sin, cos):
            mean = statistics.mean(basis)
            trend = sum(t * (b - mean) for t, b in zip(centered_t, basis)) / denominator
            basis[:] = [b - mean - trend * t for b, t in zip(basis, centered_t)]
        ss, cc = sum(s * s for s in sin), sum(c * c for c in cos)
        sc = sum(s * c for s, c in zip(sin, cos))
        sy, cy = sum(s * y for s, y in zip(sin, ys)), sum(c * y for c, y in zip(cos, ys))
        det = ss * cc - sc * sc
        if det <= 1e-12:
            continue
        a, b = (sy * cc - cy * sc) / det, (cy * ss - sy * sc) / det
        explained = max(0, min(1, (a * sy + b * cy) / energy))
        if explained > best:
            best, best_hz = explained, hz
    result.update(candidate_frequency_hz=best_hz,
                  candidate_cycles_per_minute=60 * best_hz if best_hz else None,
                  sinusoid_explained_fraction=best)
    return result


def analyze(args):
    with args.input.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    groups = segments(records, args.max_gap_ms)
    session = next((r for r in records if r.get("kind") == "session"), {})
    rows = []
    for group_id, group in enumerate(groups):
        start = group[0][0]
        while start + args.window_seconds * 1000 <= group[-1][0]:
            end = start + args.window_seconds * 1000
            samples = [(t, s) for t, s in group if start <= t < end]
            if len(samples) >= 20:
                rows.append({"segment": group_id, "start_device_ms": start,
                             "end_device_ms": end, "sample_count": len(samples),
                             "posture": session.get("posture", "unknown"),
                             "condition": session.get("condition", "unknown"),
                             **periodicity(samples, args.low_hz, args.high_hz)})
            start += args.step_seconds * 1000
    if not rows:
        raise ValueError("No usable windows; check recording length, timestamps, and gap limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} windows across {len(groups)} segments to {args.output}")
    print("Candidate periodicity only: not a breathing classification or confidence probability.")
    return 0


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("record", help="Read processed motion scores via SSE; makes no configuration changes")
    capture.add_argument("--endpoint", required=True, help="IP/hostname or full Direct API URL; bare host uses port 62587")
    capture.add_argument("--output", type=Path, required=True, help="New JSONL path; existing files are never overwritten")
    capture.add_argument("--duration", type=positive, default=300)
    capture.add_argument("--participant", required=True, help="Anonymous participant ID, or none for empty controls")
    capture.add_argument("--session-id", required=True, help="Group all conditions/nodes from the same experiment session")
    capture.add_argument("--posture", choices=["empty", "standing", "sitting", "lying_back", "lying_left", "lying_right", "other"], required=True)
    capture.add_argument("--condition", choices=["quiet", "empty", "fan", "typing", "movement", "other"], required=True)
    capture.add_argument("--position", required=True)
    capture.add_argument("--notes", default="", help="Reference recording identifier, orientation, disturbances, etc.")
    capture.set_defaults(run=record)
    inspect = commands.add_parser("analyze", help="Export exploratory periodicity measurements, not breathing decisions")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--window-seconds", type=positive, default=30)
    inspect.add_argument("--step-seconds", type=positive, default=5)
    inspect.add_argument("--max-gap-ms", type=positive, default=1000)
    inspect.add_argument("--low-hz", type=positive, default=0.1, help="Exploratory frequency search lower bound")
    inspect.add_argument("--high-hz", type=positive, default=0.6)
    inspect.set_defaults(run=analyze)
    args = parser.parse_args()
    if args.command == "analyze" and args.high_hz <= args.low_hz:
        parser.error("--high-hz must be greater than --low-hz")
    try:
        return args.run(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
