import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

from pulserank.demo import seed
from pulserank.experiments import assign
from pulserank.features import user_features
from pulserank.models import Event
from pulserank.recommender import recommend
from pulserank.storage import ConflictError, Store


class PulseRankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "test.db"))
        seed(self.store, now=10_000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_event_ingestion_is_idempotent_and_conflict_safe(self):
        event = Event("new-1", "casey", "title-01", "like", 11_000, 0.8)
        self.assertEqual(self.store.ingest([event]), (1, 0))
        self.assertEqual(self.store.ingest([event]), (0, 1))
        with self.assertRaises(ConflictError):
            self.store.ingest([Event("new-1", "casey", "title-02", "skip", 11_000, 0.1)])

    def test_features_are_point_in_time_correct(self):
        before = user_features(self.store, "casey", 11_000)
        self.store.ingest([Event("future", "casey", "title-01", "like", 12_000, 1.0)])
        unchanged = user_features(self.store, "casey", 11_000)
        after = user_features(self.store, "casey", 12_000)
        self.assertEqual(before, unchanged)
        self.assertNotEqual(after.genre_affinity, before.genre_affinity)

    def test_assignments_are_stable_and_isolated_by_experiment(self):
        self.assertEqual(assign("maya"), assign("maya"))
        self.assertNotEqual(assign("maya", "ranker-v1").bucket, assign("maya", "ranker-v2").bucket)

    def test_recommendations_exclude_recently_consumed_items(self):
        _, _, features, recs = recommend(self.store, "maya", limit=10, as_of=10_001)
        returned = {r.item.item_id for r in recs}
        self.assertFalse(returned.intersection(features.recent_items))
        self.assertEqual(len(recs), 10)

    def test_reranker_returns_a_mixed_genre_slate(self):
        _, _, _, recs = recommend(self.store, "maya", limit=8, as_of=10_001)
        genres = {g for rec in recs for g in rec.item.genres}
        self.assertGreaterEqual(len(genres), 5)

    def test_dashboard_assets_are_packaged(self):
        html = files("pulserank.static").joinpath("index.html").read_text()
        self.assertIn("PulseRank", html)
        self.assertTrue(files("pulserank.static").joinpath("app.js").is_file())

    def test_demo_seed_is_exactly_replayable(self):
        stats_before = self.store.stats()
        seed(self.store, now=10_000)
        self.assertEqual(self.store.stats(), stats_before)


if __name__ == "__main__":
    unittest.main()
