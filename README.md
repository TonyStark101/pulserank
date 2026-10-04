# PulseRank

**A real-time, learned recommendation and experimentation platform you can inspect.**

PulseRank turns viewing behavior into a personalized content feed, then exposes every stage behind the result: point-in-time features, candidate retrieval, experiment assignment, ranking scores, diversity reranking, and exposure logging. It spans the path from idempotent ingestion and event-time lakehouse processing through two-stage ML ranking, versioned online serving, controlled experiments, and production metrics.

## Run in 30 seconds

Requires Python 3.9+ and a modern browser.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pulserank demo
pulserank serve
```

Open **http://127.0.0.1:8787**. Choose a viewer and Like, Watch, or Skip a title. PulseRank ingests that signal, rebuilds the user's feature vector, reranks the candidate set, and shows the change immediately.

The deterministic demo includes 24 fictional titles, four behavioral personas, and 168 timestamped events. It needs no API keys or cloud account.

## Implemented

- **Atomic, idempotent ingestion.** Exact event retries are safe. Reusing an ID with different content returns a conflict and rolls back the transaction.
- **Point-in-time-correct features.** Every feature computation has an `as_of` boundary; future behavior cannot leak into historical training examples.
- **Multi-stage recommendations.** Taste and popularity retrieve candidates; a variant-aware ranker blends relevance, quality, and freshness; a greedy constrained reranker limits genre repetition.
- **Cold-start behavior.** Unknown viewers receive quality- and freshness-weighted discovery results rather than an empty feed.
- **Deterministic experimentation.** SHA-256 assignment maps users into 10,000 stable buckets, isolated by experiment name and ready for controlled traffic ramps.
- **Exposure logging.** Every served item records request, user, position, variant, score, and timestamp—the join key required for valid experiment analysis.
- **Inspectable UI.** The dashboard shows live features, ranking scores, experiment membership, pipeline stages, and system counters.
- **Regression tests.** Tests cover idempotency, conflict rollback, time-travel correctness, stable assignment, consumed-item filtering, and slate diversity.
- **Event-time lakehouse pipeline.** A checkpointed bronze/silver/gold reference pipeline handles out-of-order data, watermarks, explicit late-event quarantine, repair, and deterministic replay verification.
- **Distributed deployment boundary.** A pinned Redpanda broker/console stack, versioned JSON Schema contract, and Spark Structured Streaming + Delta Lake implementation mirror the tested local semantics.
- **Learned retrieval and ranking.** A BPR-trained two-tower model retrieves candidates; a regularized ranker blends collaborative relevance with quality, freshness, popularity, and taste features.
- **Leakage-safe evaluation.** Per-user temporal train/validation/test splits measure Recall@5, NDCG@5, MRR, catalog coverage, diversity, and novelty against popularity and heuristic baselines.
- **MLflow experiment tracking.** Training parameters, split metadata, metrics, model version, and serialized artifacts are recorded in a local SQLite-backed MLflow experiment.
- **Versioned online model serving.** Challenger traffic uses the serialized two-tower and ranker artifact, while control traffic remains on the heuristic baseline. Unknown users, stale artifacts, and catalog mismatches fail safely to a traceable fallback.
- **Experiment analysis.** Exposure-to-outcome attribution respects a fixed maturity window, applies CUPED variance reduction, and reports a conservative sequential harm guardrail.
- **Operational telemetry.** A Prometheus-compatible endpoint exports request counts by serving mode, fallback reasons, and latency histograms. Model readiness is separately inspectable.

## API

### Record behavior

```sh
curl -X POST http://127.0.0.1:8787/api/events \
  -H 'Content-Type: application/json' \
  -d '{"events":[{"event_id":"evt-101","user_id":"maya","item_id":"title-06","action":"like","timestamp":1770000000,"watch_pct":0.8}]}'
```

Actions are `impression`, `play`, `complete`, `like`, and `skip`.

### Request a ranked slate

```sh
curl 'http://127.0.0.1:8787/api/recommendations?user_id=maya'
```

The response includes the experiment assignment, model version, serving mode, and fallback reason. Inspect readiness at `/api/model` and Prometheus metrics at `/metrics`.

## Test

```sh
python3 -m unittest discover -s tests -v
```

## Process the event stream

The local reference pipeline is dependency-free and deliberately uses small batches so checkpoint/restart behavior is easy to inspect:

```sh
pulserank pipeline --batch-size 25 --watermark-minutes 120
pulserank verify-pipeline
```

Output lands under `data/lake` as append-only bronze and silver records, quarantined late events, an atomic gold feature snapshot, and a source-offset checkpoint. Late records can be incorporated through an explicit repair:

```sh
pulserank repair-late
pulserank verify-pipeline
```

The optional distributed path uses Redpanda, Spark Structured Streaming, and Delta Lake. It requires Python 3.10+ and Java 17:

```sh
docker compose -f infra/compose.yaml up -d
pip install -e ".[distributed]"
python -m pulserank.spark_pipeline
```

See [the streaming design](docs/streaming.md) for the event schema, watermark guarantee, checkpoint invariants, and correspondence between the reference and distributed implementations.

## Train the recommendation models

Requires the ML project extra, already included in the development environment:

```sh
pip install -e ".[ml]"
pulserank ml-demo --users 240
pulserank train --output artifacts/model --epochs 35
```

The canonical synthetic benchmark improved Recall@5 from **50.8%** for popularity and **54.9%** for the content heuristic to **62.3%** for the calibrated learned ranker. NDCG@5 improved to **38.0%**, versus 31.9% and 33.9%. These values demonstrate pipeline learning on planted synthetic structure, not production impact. See [the methodology](docs/machine-learning.md) and [raw benchmark](benchmarks/ml-evaluation.json).

Serve the trained artifact and reproduce the local load test:

```sh
pulserank serve --model artifacts/model
pulserank benchmark-serving --requests 1000
pulserank analyze-experiment --experiment ranker-v1 --window-days 7
```

On the recorded Apple Silicon development run, the full recommendation path—including SQLite reads, two-stage scoring, diversity reranking, and exposure writes—measured **11.8 ms p50**, **15.0 ms p95**, and **17.4 ms p99** across 1,000 sequential requests. See [the serving design](docs/serving.md) and [raw latency result](benchmarks/serving-latency.json).

## Roadmap

1. **Streaming foundation — implemented:** Redpanda event log, Spark Structured Streaming/Delta boundary, medallion contracts, watermarks, late-event repair, and replay verification.
2. **Production ML — in progress:** two-tower retrieval, learned ranker, temporal evaluation, MLflow tracking, serialization parity, versioned online serving, and safe fallback are implemented. Registry-driven promotion, large-catalog ANN retrieval, and drift detection remain.
3. **Experimentation and reliability — in progress:** deterministic A/B routing, outcome attribution, CUPED estimates, a sequential harm guardrail, Prometheus metrics, and reproducible latency benchmarks are implemented. Shadow routing, failure injection, and automated rollback remain.

See [the architecture notes](docs/architecture.md) for current correctness contracts and the distributed component mapping.

## Honest scope

This is a local reference system, not evidence of internet-scale operation. SQLite stands in for the online stores, the catalog is intentionally small, and the learned benchmark uses planted synthetic structure rather than real production behavior. Every performance claim is paired with a reproducible workload and raw result.
