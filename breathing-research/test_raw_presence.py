"""Synthetic format and preprocessing checks; not presence validation."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from raw_presence import main, raw_features, raw_windows, selected_slot_indices
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "Raw tests require requirements-raw.txt")
class RawPresenceTests(unittest.TestCase):
    def test_regular_packet_jitter_does_not_create_false_missing_slots(self):
        times = np.arange(6000) * 10000 + 1000 * np.sin(np.arange(6000))
        self.assertEqual(len(selected_slot_indices(times)), 6000)

    def test_static_presence_first_ten_seconds_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occupied.npz"
            times = np.arange(6000) * 10000
            csi = np.ones((6000, 128), dtype=np.int8)
            csi[:1000] = 50
            np.savez(path, csi_data=csi, device_ticks_us=times, label="static_presence", requested_pps=100)
            entry = dict(posture="occupied", condition="quiet", session_id="day01", experiment_id="occupied01")
            with patch("raw_presence.raw_features", wraps=raw_features) as extract:
                rows, quality = raw_windows(path, entry)
            selected_csi, selected_times = extract.call_args.args
            self.assertGreaterEqual(selected_times.min(), 10_000_000)
            self.assertTrue(np.all(selected_csi == 1))
            self.assertEqual(rows[0]["start_ms"], 10000)

    def test_per_packet_uniform_gain_does_not_change_features(self):
        rng = np.random.default_rng(42)
        csi = rng.integers(1, 10, size=(3000, 128), dtype=np.int8)
        gains = rng.integers(1, 5, size=(3000, 1), dtype=np.int8)
        times = np.arange(3000) * 10000
        np.testing.assert_allclose(raw_features(csi, times), raw_features(csi * gains, times), atol=1e-12)

    def test_low_slot_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse.npz"
            times = np.arange(3000) * 20000
            np.savez(path, csi_data=np.ones((3000, 128), dtype=np.int8),
                     device_ticks_us=times, label="empty", requested_pps=100)
            entry = dict(posture="empty", condition="empty", session_id="day01", experiment_id="empty01")
            with self.assertRaisesRegex(ValueError, "No usable"):
                raw_windows(path, entry)

    def test_end_to_end_matched_comparison_holds_out_days(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for day in (1, 2):
                for occupied in (False, True):
                    name = f"day{day}-{'occupied' if occupied else 'empty'}"
                    posture = "sitting" if occupied else "empty"
                    condition = "quiet" if occupied else "empty"
                    times = np.arange(6000) * 10000
                    tone = np.arange(64)[None, :]
                    t = np.arange(6000)[:, None]
                    amplitudes = (20 + 5 * np.sin(t / 30 + tone / 3) if occupied else np.broadcast_to(20 + tone % 8, (6000, 64)))
                    csi = np.zeros((6000, 128), dtype=np.int8)
                    csi[:, 1::2] = amplitudes.astype(np.int8)
                    np.savez(root / f"{name}.npz", csi_data=csi, device_ticks_us=times,
                             label="static_presence" if occupied else "empty", requested_pps=100,
                             channel_width=np.full(6000, "20"), ltf_type=np.full(6000, "ht-ltf"))
                    session = {"kind": "session", "session_id": f"day{day}", "posture": posture,
                               "condition": condition, "metadata": {"sensing": {"detector": "high_accuracy"}}}
                    with (root / f"{name}.jsonl").open("w") as score:
                        score.write(json.dumps(session) + "\n")
                        for i in range(240):
                            score.write(json.dumps({"kind": "event", "event": "motion", "received_monotonic": i / 4,
                                                    "payload": {"timestamp_ms": i * 250, "score": 0.1 if occupied else 1e-8}}) + "\n")
                    entries.append({"experiment_id": name, "session_id": f"day{day}", "posture": posture,
                                    "condition": condition, "raw_file": f"{name}.npz", "score_file": f"{name}.jsonl"})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"recordings": entries}))
            output = root / "results"
            with patch("sys.argv", ["raw_presence.py", "--manifest", str(manifest), "--output", str(output)]), redirect_stdout(io.StringIO()):
                main()
            report = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(report["evaluation_group"], "session_id")
            self.assertIn("score_metrics", report)
            self.assertEqual(len((output / "cross-examination.csv").read_text().splitlines()), 5)
            self.assertEqual(json.loads((output / "raw-model.json").read_text())["task"], "occupied_vs_empty_raw")


if __name__ == "__main__":
    unittest.main()
