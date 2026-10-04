import json
import mimetypes
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from .models import Event
from .observability import ServingMetrics
from .recommender import recommend
from .storage import ConflictError


class Handler(BaseHTTPRequestHandler):
    store = None
    model_runtime = None
    metrics = ServingMetrics()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/api/recommendations":
            started = time.perf_counter()
            user = query.get("user_id", ["maya"])[0]
            request_id, assignment, features, recs, serving = recommend(
                self.store, user, model_runtime=self.model_runtime, include_serving=True
            )
            self.json({
                "request_id": request_id,
                "assignment": assignment.__dict__,
                "features": features.__dict__,
                "serving": self.model_runtime.serialize(serving) if self.model_runtime else serving.__dict__,
                "recommendations": [{
                    "item": rec.item.__dict__, "score": round(rec.score, 4),
                    "retrieval_score": round(rec.retrieval_score, 4),
                    "rank_score": round(rec.rank_score, 4), "reasons": rec.reasons,
                } for rec in recs],
            })
            self.metrics.observe(serving, time.perf_counter() - started)
        elif parsed.path == "/api/overview":
            model = self.model_runtime.status() if self.model_runtime else {"ready": False, "error": "no model configured"}
            self.json({"stats": self.store.stats(), "model": model,
                       "timestamp": time.time(), "status": "healthy"})
        elif parsed.path == "/api/model":
            model = self.model_runtime.status() if self.model_runtime else {"ready": False, "error": "no model configured"}
            self.json(model, 200 if model["ready"] else 503)
        elif parsed.path == "/api/drift":
            if not self.model_runtime or not self.model_runtime.bundle:
                self.json({"status": "unavailable", "error": "no model configured"}, 503)
            else:
                from .drift import detect_drift
                self.json(detect_drift(self.store, self.model_runtime.bundle))
        elif parsed.path == "/healthz":
            self.json({"status": "ok"})
        elif parsed.path == "/metrics":
            self.text(self.metrics.render(), "text/plain; version=0.0.4; charset=utf-8")
        elif parsed.path == "/" or parsed.path.startswith("/static/"):
            name = "index.html" if parsed.path == "/" else parsed.path[len("/static/"):]
            self.static(name)
        else:
            self.json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/events":
            return self.json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 1_000_000:
                return self.json({"error": "body must be between 1 byte and 1 MB"}, 413)
            data = json.loads(self.rfile.read(length))
            events = [validate_event(raw) for raw in data.get("events", [])]
            if not events or len(events) > 1000:
                raise ValueError("events must contain 1 to 1000 records")
            accepted, duplicates = self.store.ingest(events)
            self.json({"accepted": accepted, "duplicates": duplicates})
        except ConflictError as exc:
            self.json({"error": str(exc)}, 409)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.json({"error": str(exc)}, 400)

    def static(self, name):
        if name not in {"index.html", "app.js", "styles.css"}:
            return self.json({"error": "not found"}, 404)
        content = files("pulserank.static").joinpath(name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def json(self, body, status=200):
        data = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def text(self, body, content_type="text/plain; charset=utf-8", status=200):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def validate_event(raw):
    required = {"event_id", "user_id", "item_id", "action", "timestamp"}
    if not required.issubset(raw):
        raise ValueError(f"missing fields: {sorted(required - set(raw))}")
    if raw["action"] not in {"impression", "play", "complete", "like", "skip"}:
        raise ValueError("invalid action")
    watch_pct = float(raw.get("watch_pct", 0))
    if not 0 <= watch_pct <= 1:
        raise ValueError("watch_pct must be between 0 and 1")
    return Event(str(raw["event_id"]), str(raw["user_id"]), str(raw["item_id"]),
                 raw["action"], float(raw["timestamp"]), watch_pct)


def serve(store, host="127.0.0.1", port=8787, model_path="artifacts/model"):
    Handler.store = store
    Handler.metrics = ServingMetrics()
    if model_path and (Path(model_path) / "manifest.json").exists():
        try:
            from .serving import ModelRuntime
            Handler.model_runtime = ModelRuntime.load(model_path)
        except ImportError as exc:
            Handler.model_runtime = None
            print(f"Model serving unavailable; using heuristic fallback ({exc})")
    else:
        Handler.model_runtime = None
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"PulseRank is running at http://{host}:{port}")
    if Handler.model_runtime:
        status = Handler.model_runtime.status()
        print(f"Model serving: {'ready' if status['ready'] else 'fallback'} ({status.get('model_version') or status.get('error')})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
