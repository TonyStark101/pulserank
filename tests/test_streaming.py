import json
import tempfile
import unittest
from pathlib import Path

from pulserank.demo import seed
from pulserank.models import Event
from pulserank.storage import Store
from pulserank.streaming import EventTimePipeline, VerificationError


class StreamingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = Store(str(root / "source.db"))
        seed(self.store, now=10_000)
        self.pipeline = EventTimePipeline(self.store, root / "lake", watermark_seconds=7_200)

    def tearDown(self):
        self.tmp.cleanup()

    def test_medallion_pipeline_is_checkpointed_and_replay_verifiable(self):
        metrics = self.pipeline.run_all(batch_size=17)
        self.assertEqual(metrics["bronze_events"], 168)
        self.assertEqual(metrics["silver_events"], 168)
        self.assertEqual(metrics["late_events"], 0)
        self.assertGreater(metrics["gold_rows"], 20)
        self.assertTrue(self.pipeline.verify()["verified"])

        # A restart at the same source offset produces no duplicate sink rows.
        repeated = self.pipeline.run_all(batch_size=11)
        self.assertEqual(repeated, metrics)

    def test_checkpoint_resumes_from_only_new_source_offsets(self):
        before = self.pipeline.run_all()
        self.store.ingest([Event("after-checkpoint", "maya", "title-01", "like", 10_100, 0.8)])
        after = self.pipeline.run_all()
        self.assertEqual(after["bronze_events"], before["bronze_events"] + 1)
        self.assertEqual(after["silver_events"], before["silver_events"] + 1)

    def test_too_late_event_is_quarantined_then_repaired(self):
        strict = EventTimePipeline(self.store, Path(self.tmp.name) / "strict", watermark_seconds=600)
        self.store.ingest([
            Event("clock-advance", "maya", "title-01", "play", 20_000, 0.5),
            Event("arrived-late", "maya", "title-02", "like", 1_000, 1.0),
        ])
        metrics = strict.run_all()
        self.assertGreater(metrics["late_events"], 0)
        silver_before = metrics["silver_events"]
        repaired = strict.repair_late()
        self.assertEqual(repaired["repaired"], repaired["late_events"])
        self.assertEqual(repaired["silver_events"], silver_before + repaired["late_events"])
        self.assertTrue(strict.verify()["verified"])

    def test_verification_detects_materialized_view_corruption(self):
        self.pipeline.run_all()
        gold_path = self.pipeline.paths["gold"]
        payload = json.loads(gold_path.read_text())
        payload["features"][0]["affinity"] = 999
        gold_path.write_text(json.dumps(payload))
        with self.assertRaises(VerificationError):
            self.pipeline.verify()


if __name__ == "__main__":
    unittest.main()
