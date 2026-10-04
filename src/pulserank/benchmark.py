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


def benchmark_ann(items=100_000, queries=100, dimensions=64, neighbors=50,
                  output="benchmarks/ann-retrieval.json", seed=17):
    """Measure HNSW recall and latency against brute-force inner product."""
    import hnswlib
    import numpy as np

    rng = np.random.default_rng(seed)
    item_vectors = rng.normal(size=(items, dimensions)).astype(np.float32)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)
    query_vectors = rng.normal(size=(queries, dimensions)).astype(np.float32)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    index = hnswlib.Index(space="ip", dim=dimensions)
    before = time.perf_counter()
    index.init_index(max_elements=items, ef_construction=200, M=32, random_seed=seed)
    index.add_items(item_vectors, np.arange(items), num_threads=1)
    build_seconds = time.perf_counter() - before
    ef_search = max(300, neighbors * 6)
    index.set_ef(ef_search)
    before = time.perf_counter()
    approximate, _ = index.knn_query(query_vectors, k=neighbors, num_threads=1)
    ann_seconds = time.perf_counter() - before
    before = time.perf_counter()
    exact = []
    for query in query_vectors:
        scores = item_vectors @ query
        candidates = np.argpartition(scores, -neighbors)[-neighbors:]
        exact.append(set(int(value) for value in candidates))
    exact_seconds = time.perf_counter() - before
    recalls = [len(set(map(int, approximate[index])) & exact[index]) / neighbors for index in range(queries)]
    result = {
        "workload": {"items": items, "queries": queries, "dimensions": dimensions,
                     "neighbors": neighbors, "seed": seed},
        "hnsw": {"m": 32, "ef_construction": 200, "ef_search": ef_search},
        "build_seconds": build_seconds,
        "recall_at_k": float(np.mean(recalls)),
        "ann_latency_ms_per_query": ann_seconds * 1000 / queries,
        "exact_latency_ms_per_query": exact_seconds * 1000 / queries,
        "speedup": exact_seconds / ann_seconds,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


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
