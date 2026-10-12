"""Check weighting and feature conventions, not real-world presence accuracy."""

import unittest

from train_presence import features, fit, predict


class PresenceTrainingTests(unittest.TestCase):
    def test_flat_zero_features_are_finite_and_explicit(self):
        self.assertEqual(features([(0, 0), (250, 0)]), [-12, -12, -12, -12, 1, 0])

    def test_duplicating_windows_does_not_change_recording_weight(self):
        rows = [dict(recording="empty", label=0, features=[-8, -8, -8, -8, 0, 0]),
                dict(recording="occupied", label=1, features=[-3, -3, -3, -3, 0, 1])]
        original = fit(rows)
        duplicated = fit([rows[0]] * 10 + [rows[1]])
        for row in rows:
            self.assertAlmostEqual(predict(original, row["features"]), predict(duplicated, row["features"]), places=10)
        self.assertLess(predict(original, rows[0]["features"]), 0.5)
        self.assertGreater(predict(original, rows[1]["features"]), 0.5)

    def test_training_requires_both_labels(self):
        with self.assertRaises(ValueError):
            fit([dict(recording="empty", label=0, features=[0] * 6)])


if __name__ == "__main__":
    unittest.main()
