"""Check live inference matches training and stale evidence is cleared."""

import argparse
import io
import json
import math
import unittest
from unittest.mock import patch

from live_presence import LiveDetector, live
from train_presence import FEATURES, features, predict


def model_fixture():
    return {"feature_names": FEATURES, "means": [0] * 6, "scales": [1] * 6,
            "coefficients": [0.1] * 6, "intercept": 0.2, "threshold": 0.5,
            "required_detector": "high_accuracy", "window_seconds": 30,
            "trim_seconds": 10, "max_gap_ms": 1000}


def sensing_fixture():
    return {"enabled": True, "ready": True, "detector": "high_accuracy", "threshold": 0.5}


class LivePresenceTests(unittest.TestCase):
    def test_live_window_matches_training_feature_convention(self):
        model = model_fixture()
        detector = LiveDetector(model, sensing_fixture())
        points = []
        predictions = []
        for i in range(161):
            timestamp = i * 250
            score = 0.3 + 0.05 * math.sin(i)
            points.append((timestamp, score))
            result = detector.event("motion", {"timestamp_ms": timestamp, "score": score}, i * 0.25)
            if result and "occupied_score" in result:
                predictions.append(result)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["samples"], 120)
        self.assertEqual(predictions[0]["window_start_ms"], 10000)
        self.assertAlmostEqual(predictions[0]["occupied_score"], predict(model, features(points[40:160])))

    def test_stale_input_clears_evidence_and_restarts_warmup(self):
        detector = LiveDetector(model_fixture(), sensing_fixture())
        for i in range(161):
            detector.event("motion", {"timestamp_ms": i * 250, "score": 0.1}, i * 0.25)
        self.assertEqual(detector.expire(42)["status"], "UNKNOWN")
        self.assertFalse(detector.points)
        resumed = detector.event("motion", {"timestamp_ms": 45000, "score": 0.1}, 45)
        self.assertEqual(resumed["status"], "WARMING_UP")
        self.assertEqual(resumed["seconds_remaining"], 40)

    def test_configuration_change_and_not_ready_block_prediction(self):
        detector = LiveDetector(model_fixture(), sensing_fixture())
        detector.event("motion", {"timestamp_ms": 1000, "score": 0.1}, 1)
        self.assertIsNone(detector.event("sensing", {**sensing_fixture(), "uptime": 2}, 2))
        self.assertEqual(len(detector.points), 0)  # Still in settling period.
        result = detector.event("sensing", {**sensing_fixture(), "ready": False}, 2)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(detector.event("motion", {"timestamp_ms": 2000, "score": 1}, 2))
        result = detector.event("sensing", {**sensing_fixture(), "threshold": 0.6}, 3)
        self.assertEqual(result["status"], "WARMING_UP")

    def test_device_reset_restarts_window(self):
        detector = LiveDetector(model_fixture(), sensing_fixture())
        detector.event("motion", {"timestamp_ms": 1000, "score": 0.1}, 1)
        result = detector.event("motion", {"timestamp_ms": 0, "score": 0.1}, 1.25)
        self.assertEqual(result["seconds_remaining"], 40)

    def test_live_sse_reads_only_and_disconnect_becomes_unknown(self):
        stream = []
        for i in range(181):
            payload = {"timestamp_ms": i * 250, "score": 0.1}
            stream.append("event: motion\ndata: " + json.dumps(payload) + "\n\n")
        def open_stream(request, timeout):
            self.assertEqual(request.get_method(), "GET")
            self.assertTrue(request.full_url.endswith("/events"))
            self.assertIn("Origin", request.headers)
            return io.BytesIO("".join(stream).encode())
        args = argparse.Namespace(endpoint="s3", update_seconds=5, duration=3, json=True)
        metadata = [sensing_fixture(), {"events": ["motion"]}]
        with patch("live_presence.read_json", side_effect=metadata), \
                patch("live_presence.urllib.request.urlopen", side_effect=open_stream), \
                patch("builtins.print") as output:
            self.assertEqual(live(args, model_fixture()), 1)
        results = [json.loads(call.args[0]) for call in output.call_args_list]
        self.assertTrue(any("occupied_score" in result for result in results))
        self.assertEqual(results[-1]["status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
