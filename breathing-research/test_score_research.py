"""Offline numerical and framing checks; these do not validate breathing."""

import math
import argparse
import csv
import io
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from score_research import analyze, api_base, periodicity, record, segments, sse_events


class ScoreResearchTests(unittest.TestCase):
    def test_periodicity_recovers_known_frequency_with_trend_and_jitter(self):
        samples = []
        for i in range(240):
            t = i * 0.25 + 0.015 * math.sin(i)
            samples.append((t * 1000, 0.3 + 0.001 * t + 0.05 * math.sin(2 * math.pi * 0.25 * t)))
        result = periodicity(samples)
        self.assertAlmostEqual(result["candidate_frequency_hz"], 0.25, delta=0.005)
        self.assertGreater(result["sinusoid_explained_fraction"], 0.98)

    def test_flat_score_has_no_frequency(self):
        result = periodicity([(i * 250, 0.1) for i in range(120)])
        self.assertIsNone(result["candidate_frequency_hz"])

    def test_gaps_resets_and_configuration_changes_split_history(self):
        def motion(timestamp, received):
            return {"kind": "event", "event": "motion", "received_monotonic": received,
                    "payload": {"timestamp_ms": timestamp, "score": 0.1}}
        records = [motion(1000, 1), motion(1250, 1.25), motion(3000, 3),
                   motion(0, 3.25), {"kind": "event", "event": "sensing"},
                   motion(250, 3.5), motion(500, 6)]
        self.assertEqual([len(g) for g in segments(records, 1000)], [2, 1, 1, 1, 1])

    def test_sse_heartbeat_and_multiline_json(self):
        events = list(sse_events([b": heartbeat\n", b"\n", b"event: motion\r\n",
                                  b'data: {"score": 0.1,\n', b'data: "timestamp_ms": 250}\n', b"\n"]))
        self.assertEqual(events, [("motion", {"score": 0.1, "timestamp_ms": 250})])

    def test_endpoint(self):
        self.assertEqual(api_base("192.168.1.10"), "http://192.168.1.10:62587/espectre/v1")
        self.assertEqual(api_base("http://sensor:62587/espectre/v1/"), "http://sensor:62587/espectre/v1")

    def test_record_and_analyze_preserve_partial_stream_and_use_only_get(self):
        resources = {"device": {"device_id": "test-s3"},
                     "sensing": {"ready": True, "detector": "high_accuracy"},
                     "capabilities": {"events": ["motion"]}}
        paths = []
        def open_response(request, timeout):
            self.assertIsInstance(request, urllib.request.Request)
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.get_header("Origin"), "https://test.espectre.dev")
            url = request.full_url
            resource = url.rsplit("/", 1)[1]
            paths.append(resource)
            if resource != "events":
                return io.BytesIO(json.dumps(resources[resource]).encode())
            self.assertEqual(request.get_header("Accept"), "text/event-stream")
            stream = []
            for i in range(240):
                payload = {"timestamp_ms": i * 250, "state": "idle",
                           "score": 0.3 + 0.05 * math.sin(2 * math.pi * 0.25 * i * 0.25)}
                stream.append("event: motion\ndata: " + json.dumps(payload) + "\n\n")
            return io.BytesIO("".join(stream).encode())

        with tempfile.TemporaryDirectory() as directory:
            capture_path = Path(directory) / "capture.jsonl"
            report_path = Path(directory) / "report.csv"
            args = argparse.Namespace(endpoint="test-s3", output=capture_path, duration=300,
                                      participant="p01", session_id="day01", posture="sitting",
                                      condition="quiet", position="desk", notes="synthetic fixture")
            with patch("score_research.urllib.request.urlopen", side_effect=open_response), patch("builtins.print"):
                # Premature disconnect is explicit, and the useful recording survives.
                self.assertEqual(record(args), 1)
                self.assertEqual(paths, ["device", "sensing", "capabilities", "events"])
                events = [json.loads(line) for line in capture_path.read_text().splitlines()]
                self.assertEqual(events[-1]["motion_events"], 240)
                self.assertIn("closed", events[-1]["error"])
                analysis_args = argparse.Namespace(input=capture_path, output=report_path,
                                                   max_gap_ms=1000, window_seconds=30,
                                                   step_seconds=5, low_hz=0.1, high_hz=0.6)
                self.assertEqual(analyze(analysis_args), 0)
            with report_path.open() as report:
                rows = list(csv.DictReader(report))
            self.assertGreater(len(rows), 0)
            self.assertAlmostEqual(float(rows[0]["candidate_cycles_per_minute"]), 15, delta=0.5)
            with self.assertRaises(FileExistsError), patch("score_research.urllib.request.urlopen", side_effect=open_response):
                record(args)


if __name__ == "__main__":
    unittest.main()
