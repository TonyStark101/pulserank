import tempfile
import unittest
from pathlib import Path

from pulserank.demo import seed
from pulserank.experimentation import analyze_experiment
from pulserank.models import Event
from pulserank.storage import Store


class ExperimentAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "experiment.db"))
        seed(self.store, now=900_000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cuped_readout_attributes_only_exposed_post_period_events(self):
        timestamp = 1_000_000
        rows = [
            ("control-1", "control-a", "title-01", 1, "ranker-v1", "control", .4, timestamp, None, "heuristic_control"),
            ("control-2", "control-b", "title-02", 1, "ranker-v1", "control", .4, timestamp, None, "heuristic_control"),
            ("challenger-1", "challenger-a", "title-03", 1, "ranker-v1", "challenger", .5, timestamp, "v1", "learned_challenger"),
            ("challenger-2", "challenger-b", "title-04", 1, "ranker-v1", "challenger", .5, timestamp, "v1", "learned_challenger"),
        ]
        self.store.record_exposures(rows)
        self.store.ingest([
            Event("outcome-c", "control-a", "title-01", "like", timestamp + 10, .8),
            Event("outcome-a", "challenger-a", "title-03", "complete", timestamp + 10, 1),
            Event("outcome-b", "challenger-b", "title-04", "like", timestamp + 20, .9),
            Event("unexposed", "control-b", "title-08", "like", timestamp + 10, .8),
        ])
        report = analyze_experiment(self.store, window_days=1, as_of=timestamp + 86_401)
        self.assertEqual(report["mature_requests"], 4)
        self.assertEqual(report["variants"]["control"]["conversion_rate"], .5)
        self.assertEqual(report["variants"]["challenger"]["conversion_rate"], 1.0)
        self.assertGreater(report["comparison"]["absolute_lift"], 0)

    def test_immature_exposures_are_not_peeked(self):
        timestamp = 1_000_000
        self.store.record_exposures([
            ("r1", "u1", "title-01", 1, "ranker-v1", "control", .4,
             timestamp, None, "heuristic_control")
        ])
        report = analyze_experiment(self.store, window_days=7, as_of=timestamp + 60)
        self.assertEqual(report["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
