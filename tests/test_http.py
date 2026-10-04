import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from pulserank.api import Handler
from pulserank.demo import seed
from pulserank.storage import Store


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        store = Store(str(Path(cls.tmp.name) / "http.db"))
        seed(store, now=10_000)
        Handler.store = store
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.tmp.cleanup()

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as response:
            return response.status, response.headers.get_content_type(), response.read()

    def test_dashboard_and_assets_are_served(self):
        status, content_type, body = self.get("/")
        self.assertEqual((status, content_type), (200, "text/html"))
        self.assertIn(b"Your feed is a", body)
        self.assertEqual(self.get("/static/app.js")[:2], (200, "text/javascript"))

    def test_recommendation_response_has_traceable_decision_data(self):
        status, _, body = self.get("/api/recommendations?user_id=maya")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["assignment"]["experiment"], "ranker-v1")
        self.assertEqual(len(payload["recommendations"]), 12)
        self.assertIn("retrieval_score", payload["recommendations"][0])
        self.assertEqual(payload["serving"]["mode"], "heuristic")

    def test_prometheus_metrics_capture_serving_latency(self):
        self.get("/api/recommendations?user_id=maya")
        status, content_type, body = self.get("/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/plain")
        self.assertIn(b'pulserank_recommendation_requests_total{mode="heuristic"}', body)
        self.assertIn(b"pulserank_recommendation_latency_seconds_count", body)


if __name__ == "__main__":
    unittest.main()
