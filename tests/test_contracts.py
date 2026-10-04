import json
import unittest
from pathlib import Path

import pulserank.spark_pipeline as spark_pipeline


ROOT = Path(__file__).resolve().parents[1]


class DeploymentContractTests(unittest.TestCase):
    def test_versioned_event_schema_has_strict_required_fields(self):
        schema = json.loads((ROOT / "schemas" / "viewing-event-v1.json").read_text())
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(set(schema["required"]), {
            "schema_version", "event_id", "user_id", "item_id",
            "action", "event_time", "watch_pct",
        })
        self.assertEqual(set(schema["properties"]["action"]["enum"]), {
            "impression", "play", "complete", "like", "skip",
        })

    def test_distributed_module_is_safe_to_import_without_spark(self):
        self.assertTrue(callable(spark_pipeline.build_spark))
        self.assertTrue(callable(spark_pipeline.start_pipeline))

    def test_compose_pins_broker_and_console_images(self):
        compose = (ROOT / "infra" / "compose.yaml").read_text()
        self.assertIn("redpanda:v26.2.3", compose)
        self.assertIn("console:v3.12.0", compose)
        self.assertIn("viewing-events.v1", compose)


if __name__ == "__main__":
    unittest.main()
