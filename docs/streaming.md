# Event-time streaming design

PulseRank has two implementations of one behavioral contract:

1. `pulserank.streaming.EventTimePipeline` is the deterministic reference implementation. It runs anywhere Python runs and is used for failure, replay, and differential tests.
2. `pulserank.spark_pipeline` is the distributed implementation. It consumes the same schema from Redpanda through the Kafka API and persists bronze and silver Delta tables with independent checkpoints.

## Data contract

`schemas/viewing-event-v1.json` is the versioned wire contract. Producers send UTC RFC 3339 event time separately from broker/processing time. The event ID is globally unique and immutable.

| Layer | Contract |
| --- | --- |
| Bronze | Append the original payload plus source topic, partition, offset, and broker timestamp. Never silently discard a consumed record. |
| Silver | Parse v1 events, enforce action and range constraints, normalize event time, watermark state, and deduplicate by event ID. |
| Quarantine | Preserve events that arrive behind the reference watermark, including the watermark and measured lateness. |
| Gold | Materialize deterministic user/genre affinities from silver events. Record a content digest for replay comparison. |

## Watermark semantics

The reference watermark is:

```text
maximum event time observed − allowed lateness
```

An event older than that boundary is retained in bronze and copied to quarantine, but excluded from the low-latency silver/gold path. `repair-late` replays quarantined data in event-time order and rebuilds the affected materialization. Keeping repair explicit prevents a surprise old event from rewriting online features without an audit trail.

Spark's guarantee is deliberately phrased differently: events less delayed than the configured watermark threshold are guaranteed not to be dropped, while older records may or may not be processed. The local reference pipeline uses the stricter deterministic rule so tests never depend on engine timing.

## Checkpoint and replay invariants

- Source offsets advance only after the corresponding sink append is flushed.
- Sink identity keys make a replay after a crash idempotent.
- Checkpoints are replaced atomically and never edited in place.
- A clean replay of silver must reproduce the gold rows and SHA-256 digest exactly.
- Corrupt or duplicate silver state makes verification fail loudly.

## Run the local pipeline

```sh
pulserank demo
pulserank pipeline --batch-size 25 --watermark-minutes 120
pulserank verify-pipeline
```

To exercise the late-data path, ingest an event behind the active watermark, run `pipeline`, inspect `data/lake/quarantine`, then run:

```sh
pulserank repair-late
pulserank verify-pipeline
```

## Run the distributed boundary

This path requires Docker for Redpanda, plus Python 3.10+ and Java 17 for current Spark/Delta releases.

```sh
docker compose -f infra/compose.yaml up -d
pip install -e ".[distributed]"
python -m pulserank.spark_pipeline
```

The launcher pins Spark's driver and Python workers to the active interpreter, avoiding training/serving code running under different Python installations. On Apple Silicon it discovers Homebrew's keg-only Java 17 automatically; other platforms should set `JAVA_HOME` normally.

Redpanda Console is available at `http://localhost:8080`. The broker is exposed at `localhost:19092`, and startup creates a six-partition `viewing-events.v1` topic.

The distributed path pins Spark 4.1.1 with Delta Lake 4.3.1. Delta 4.3.1's package metadata constrains Spark to `>=4.0.1, <=4.1.1`. Its state-store partition count is checkpoint-sensitive and must not be changed under an existing query checkpoint.
