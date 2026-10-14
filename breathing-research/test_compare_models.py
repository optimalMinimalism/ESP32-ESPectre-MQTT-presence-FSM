import unittest

try:
    from compare_models import candidates, export_forest, train, weights, tree_evaluation
    from live_presence import load_model, model_score
except ModuleNotFoundError:
    weights = None


@unittest.skipIf(weights is None, "Model comparisons require requirements-models.txt")
class ComparisonTests(unittest.TestCase):
    def test_exported_extra_trees_matches_sklearn_predictions(self):
        import json
        import tempfile
        from pathlib import Path
        rows = [dict(recording=f"r{i}", label=int(i >= 4), features=[float(i + j) for j in range(6)]) for i in range(8)]
        estimator = train(candidates()["extra_trees"], rows)
        exported = export_forest(estimator, rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(exported))
            loaded = load_model(path)
        probes = [r["features"] for r in rows] + [[2.5 + j for j in range(6)]]
        expected = estimator.predict_proba(probes)[:, list(estimator.classes_).index(1)]
        for values, score in zip(probes, expected):
            self.assertAlmostEqual(model_score(loaded, values), score, places=12)

    def test_class_and_recording_weights_do_not_favor_long_captures(self):
        rows = [dict(recording="empty1", label=0)] * 10 + [dict(recording="empty2", label=0), dict(recording="occupied", label=1)]
        result = weights(rows)
        self.assertAlmostEqual(sum(result[:10]), 0.25)
        self.assertAlmostEqual(result[10], 0.25)
        self.assertAlmostEqual(result[11], 0.5)

    def test_session_fold_rejects_training_with_only_one_class(self):
        rows = [dict(recording="a", session_id="day1", label=0, features=[0] * 6),
                dict(recording="b", session_id="day2", label=1, features=[1] * 6)]
        with self.assertRaisesRegex(ValueError, "both empty and occupied"):
            tree_evaluation(rows, "random_forest", "session_id")


if __name__ == "__main__":
    unittest.main()
