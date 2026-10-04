# Online serving and experimentation

PulseRank serves a deterministic control and a learned challenger behind the same recommendation contract. Stable hashing assigns each user before scoring; this makes the exposure log the source of truth for later causal analysis.

## Request path

1. Build point-in-time user features and remove recently consumed items.
2. Resolve the stable `ranker-v1` assignment.
3. For eligible challenger users, retrieve candidates with the two-tower embeddings and score them with the calibrated ranker.
4. Greedily rerank the slate with a genre-repetition penalty.
5. Atomically log request, position, variant, model version, serving mode, and score.

Control users remain on the heuristic baseline. A challenger falls back to that path when the artifact is absent or invalid, the user is unknown, or the live catalog differs from the training catalog. Every response makes this decision visible.

## Artifact contract

The version-2 artifact contains user/item embeddings, item bias, ranker normalization and weights, the validation-selected ensemble weight, and immutable online feature state. Its version is a SHA-256 digest of learned weights and calibration. Loading validates the serving feature contract before the artifact becomes ready.

The current retrieval pass is exact NumPy scoring because the reference catalog contains 24 items. The boundary is deliberately isolated so an ANN index can replace it without changing ranking, experimentation, or exposure contracts.

## Observability

`GET /api/model` exposes readiness, artifact version, dimensions, and cardinalities. `GET /metrics` exposes Prometheus text metrics for:

- requests partitioned by learned, control, and fallback modes;
- fallbacks partitioned by reason;
- end-to-end recommendation latency histogram, sum, and count.

The benchmark command builds an isolated deterministic workload, performs the complete request path, and writes raw results to `benchmarks/serving-latency.json`.

## Experiment correctness

`pulserank analyze-experiment` groups exposures at request level, waits for a fixed attribution window to mature, joins only subsequent positive actions on exposed items, and uses pre-period engagement as a CUPED covariate. The readout reports adjusted conversion, uncertainty, lift, and a 99% one-sided harm guardrail after both variants reach 30 mature requests.

The guardrail is intentionally not a claim of a fully general sequential-testing framework. Repeated looks, user-level clustering, multiple metrics, and power planning would need explicit treatment before production use.
