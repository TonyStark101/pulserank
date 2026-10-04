# Machine learning system

PulseRank trains two stages from implicit viewing feedback and evaluates every model against reproducible baselines.

## Labels and temporal split

Likes and completions are positive labels. Plays become positive only after 60% watch progress; short plays, impressions, and skips supply negatives. Each eligible user's interactions are ordered by event time:

- everything before the final two positive items is training data;
- the penultimate positive item is validation data;
- the final positive item is the untouched test target.

Model fitting never consumes validation or test labels. The validation item selects the learned/heuristic ensemble weight. Test metrics are computed once after that choice.

## Retrieval model

The first stage is a two-tower implicit-feedback model. User and item ID towers produce embeddings whose dot product scores a candidate. Training uses Bayesian Personalized Ranking: every update asks a known positive item to outrank a sampled unobserved item. L2 regularization and seeded sampling make the run deterministic.

The implementation is intentionally direct NumPy rather than a framework wrapper. The optimization boundary, negative sampling, gradients, and artifact layout remain inspectable during interviews.

## Ranking model

The second stage is a regularized logistic ranker over:

- retrieval score;
- item quality and freshness;
- training-only popularity;
- user/genre taste match;
- a relevance-quality interaction.

An ensemble weight between the learned scorer and the content heuristic is selected from eleven candidates using validation NDCG@5. This protects the serving model from accepting a learned component merely because it exists.

## Evaluation

For every test user, all non-training and non-validation items are ranked. PulseRank reports:

- Recall@5 and NDCG@5;
- mean reciprocal rank;
- catalog coverage;
- intra-list genre diversity;
- popularity-adjusted novelty.

The committed [benchmark result](../benchmarks/ml-evaluation.json) compares popularity, the original heuristic, raw two-tower retrieval, and the calibrated learned ranker. The synthetic generator plants genre and hidden collaborative preferences. Its numbers verify learning and evaluation behavior, not real-world business impact.

## Reproduce

```sh
source .venv/bin/activate
pulserank ml-demo --users 240
pulserank train --output artifacts/model --epochs 35
```

The run writes `manifest.json`, compressed model weights, and `evaluation.json`. MLflow stores parameters, metrics, tags, and artifacts in `data/mlflow.db` under the `pulserank-ranking` experiment.

To inspect runs:

```sh
mlflow server --backend-store-uri sqlite:///data/mlflow.db --host 127.0.0.1 --port 5000
```

## Artifact contract

Model artifacts contain an explicit format version, content-derived model version, ordered user/item vocabularies, feature names, embedding dimensions, normalization statistics, ensemble weight, and compressed weights. A round-trip parity test requires predictions before and after serialization to match within `1e-12`.

