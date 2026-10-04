import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pulserank.ml.data import build_dataset, seed_ml_population
from pulserank.ml.model import ModelBundle
from pulserank.ml.training import feature_matrix, train
from pulserank.drift import detect_drift
from pulserank.experiments import assign
from pulserank.recommender import recommend
from pulserank.registry import ModelRegistry, PromotionError
from pulserank.serving import ModelRuntime
from pulserank.storage import Store


class MachineLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.store = Store(str(cls.root / "training.db"))
        seed_ml_population(cls.store, users=100)
        cls.report, cls.bundle, cls.dataset = train(
            cls.store, cls.root / "model", dimensions=8, epochs=15, seed=17, track=False
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_population_seed_is_idempotent(self):
        result = seed_ml_population(self.store, users=100)
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(result["duplicates"], 2400)

    def test_temporal_split_has_disjoint_targets(self):
        dataset = build_dataset(self.store)
        for uid, train_items in dataset.train_positives.items():
            self.assertNotIn(dataset.validation_items[uid], train_items)
            self.assertNotIn(dataset.test_items[uid], train_items)
            self.assertNotEqual(dataset.validation_items[uid], dataset.test_items[uid])

    def test_training_reduces_pairwise_loss(self):
        training = self.report["training"]
        self.assertLess(training["final_bpr_loss"], training["initial_bpr_loss"] * 0.6)

    def test_learned_ranker_beats_popularity_baseline(self):
        learned = self.report["metrics"]["learned_ranker"]
        baseline = self.report["metrics"]["popularity"]
        self.assertGreater(learned["ndcg_at_5"], baseline["ndcg_at_5"])
        self.assertGreater(learned["recall_at_5"], baseline["recall_at_5"])
        self.assertGreaterEqual(learned["coverage_at_5"], baseline["coverage_at_5"])

    def test_saved_model_has_training_serving_parity(self):
        loaded = ModelBundle.load(self.root / "model")
        uid = 0
        candidates = np.arange(len(self.dataset.items), dtype=np.int64)
        features = feature_matrix(self.dataset, self.bundle.retrieval, uid, candidates)
        expected = self.bundle.ranker.scores(features)
        loaded_features = feature_matrix(self.dataset, loaded.retrieval, uid, candidates)
        actual = loaded.ranker.scores(loaded_features)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        self.assertEqual(loaded.version, self.report["model_version"])

    def test_challenger_uses_versioned_learned_model_online(self):
        runtime = ModelRuntime.load(self.root / "model")
        user = next(value for value in self.dataset.users if assign(value).variant == "challenger")
        _, assignment, _, recs, serving = recommend(
            self.store, user, limit=5, model_runtime=runtime, include_serving=True
        )
        self.assertEqual(assignment.variant, "challenger")
        self.assertEqual(serving.mode, "learned_challenger")
        self.assertEqual(serving.model_version, self.report["model_version"])
        self.assertEqual(len(recs), 5)

    def test_unknown_user_falls_back_without_failing_request(self):
        runtime = ModelRuntime.load(self.root / "model")
        user = next(f"unknown-{index}" for index in range(100) if assign(f"unknown-{index}").variant == "challenger")
        _, _, _, recs, serving = recommend(
            self.store, user, limit=5, model_runtime=runtime, include_serving=True
        )
        self.assertEqual(serving.mode, "heuristic_fallback")
        self.assertEqual(serving.fallback_reason, "unknown user")
        self.assertEqual(len(recs), 5)

    def test_missing_artifact_reports_unready_instead_of_crashing(self):
        runtime = ModelRuntime.load(self.root / "does-not-exist")
        self.assertFalse(runtime.status()["ready"])

    def test_registry_applies_quality_gates_and_promotes_atomically(self):
        registry = ModelRegistry(self.root / "registry")
        staged = registry.register(self.root / "model")
        self.assertTrue(staged["gates"]["passed"])
        status = registry.promote(self.report["model_version"])
        self.assertEqual(status["active_version"], self.report["model_version"])
        self.assertTrue(registry.active_path().is_dir())
        second_report, _, _ = train(
            self.store, self.root / "model-two", dimensions=8, epochs=16, seed=17, track=False
        )
        registry.register(self.root / "model-two")
        registry.promote(second_report["model_version"])
        rolled_back = registry.rollback()
        self.assertEqual(rolled_back["active_version"], self.report["model_version"])

    def test_registry_rejects_model_that_regresses_against_heuristic(self):
        candidate = self.root / "regressed-model"
        shutil.copytree(self.root / "model", candidate)
        report_path = candidate / "evaluation.json"
        report = json.loads(report_path.read_text())
        report["metrics"]["learned_ranker"]["ndcg_at_5"] = 0.0
        report_path.write_text(json.dumps(report))
        registry = ModelRegistry(self.root / "rejected-registry")
        staged = registry.register(candidate)
        self.assertFalse(staged["gates"]["passed"])
        with self.assertRaises(PromotionError):
            registry.promote(self.report["model_version"])

    def test_drift_report_is_versioned_and_uses_live_events(self):
        report = detect_drift(self.store, self.bundle)
        self.assertEqual(report["model_version"], self.report["model_version"])
        self.assertGreater(report["observed_events"], 0)
        self.assertGreaterEqual(report["item_distribution_js"], 0)


if __name__ == "__main__":
    unittest.main()
