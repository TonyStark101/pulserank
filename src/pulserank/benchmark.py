import json
import platform
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from .ml.data import seed_ml_population
from .recommender import recommend
from .serving import ModelRuntime
from .storage import Store


def benchmark_serving(model_path="artifacts/model", requests=1000, output=None):
    """Run a deterministic, local end-to-end serving workload."""
    runtime = ModelRuntime.load(model_path)
    if not runtime.status()["ready"]:
        raise ValueError(f"model is not ready: {runtime.status()['error']}")
    with tempfile.TemporaryDirectory() as directory:
        store = Store(str(Path(directory) / "benchmark.db"))
        synthetic_users = [value for value in runtime.bundle.user_ids if value.startswith("viewer-")]
        seed_ml_population(store, users=len(synthetic_users))
        users = runtime.bundle.user_ids
        durations, modes = [], Counter()
        started = time.perf_counter()
        for index in range(requests):
            before = time.perf_counter_ns()
            *_, serving = recommend(
                store, users[index % len(users)], limit=12,
                model_runtime=runtime, include_serving=True,
            )
            durations.append((time.perf_counter_ns() - before) / 1_000_000)
            modes[serving.mode] += 1
        elapsed = time.perf_counter() - started
    ordered = sorted(durations)

    def percentile(value):
        position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * value)))
        return ordered[position]

    result = {
        "workload": {"requests": requests, "slate_size": 12, "catalog_items": len(runtime.bundle.item_ids)},
        "model_version": runtime.version,
        "latency_ms": {
            "mean": statistics.mean(durations), "p50": percentile(0.50),
            "p95": percentile(0.95), "p99": percentile(0.99), "max": max(durations),
        },
        "throughput_requests_per_second": requests / elapsed,
        "serving_modes": dict(modes),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
