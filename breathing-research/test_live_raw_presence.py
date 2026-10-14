import unittest

try:
    import numpy as np
    from live_raw_presence import RawDetector
    from raw_presence import RAW_FEATURES, raw_features, selected_slot_indices
    from train_presence import predict
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "Raw live tests require requirements-raw.txt")
class LiveRawTests(unittest.TestCase):
    def test_live_window_matches_raw_training_and_stale_input_clears(self):
        model = {"feature_names": RAW_FEATURES, "means": [0] * 6, "scales": [1] * 6,
                 "coefficients": [0.01] * 6, "intercept": 0.2, "threshold": 0.5,
                 "minimum_slot_coverage": 0.85}
        detector = RawDetector(model)
        iq = np.ones(128, dtype=np.int8)
        for index in range(4001):
            result = detector.process(iq, index * 10000, index / 100)
        self.assertIn("occupied_score", result)
        points = np.ones((3000, 128), dtype=np.int8)
        expected = predict(model, raw_features(points, np.arange(1000, 4000) * 10000))
        self.assertAlmostEqual(result["occupied_score"], expected)
        self.assertEqual(result["samples"], 3000)
        self.assertEqual(detector.expire(42)["status"], "UNKNOWN")
        self.assertFalse(detector.points)
        restarted = detector.process(iq, 45_000_000, 45)
        self.assertEqual(restarted["seconds_remaining"], 40)


if __name__ == "__main__":
    unittest.main()
