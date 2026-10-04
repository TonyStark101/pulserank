import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .data import build_dataset, temporal_split_manifest
from .model import ModelBundle, RankerModel, TwoTowerModel


FEATURE_NAMES = [
    "retrieval_score", "item_quality", "item_freshness",
    "train_popularity", "genre_taste_match", "quality_x_relevance",
]


def feature_matrix(dataset, retrieval, uid, item_ids):
    item_ids = np.asarray(item_ids, dtype=np.int64)
    retrieval_scores = retrieval.scores(uid)[item_ids]
    quality = np.array([dataset.items[iid].quality for iid in item_ids])
    freshness = np.array([dataset.items[iid].freshness for iid in item_ids])
    popularity = dataset.popularity[item_ids]
    taste = dataset.genre_matrix[item_ids] @ dataset.taste[uid]
    return np.column_stack([
        retrieval_scores, quality, freshness, popularity, taste,
        quality * retrieval_scores,
    ])


def ranker_examples(dataset, retrieval, seed=31):
    rng = np.random.default_rng(seed)
    features, labels = [], []
    all_items = set(range(len(dataset.items)))
    for uid, positives in dataset.train_positives.items():
        explicit = sorted(dataset.train_negatives[uid])
        fallback = sorted(all_items - positives - {dataset.validation_items[uid], dataset.test_items[uid]})
        for positive in sorted(positives):
            features.append(feature_matrix(dataset, retrieval, uid, [positive])[0])
            labels.append(1.0)
        negatives = explicit
        if not negatives:
            sample_size = min(len(positives), len(fallback))
            negatives = list(rng.choice(fallback, size=sample_size, replace=False))
        for negative in negatives:
            features.append(feature_matrix(dataset, retrieval, uid, [negative])[0])
            labels.append(0.0)
    return np.asarray(features), np.asarray(labels)


def _ranking_metrics(dataset, score_function, cutoff=5, target_kind="test"):
    reciprocal_ranks, recalls, ndcgs, novelties = [], [], [], []
    recommended = set()
    diversity_values = []
    total_train = sum(len(values) for values in dataset.train_positives.values())
    for uid in range(len(dataset.users)):
        if target_kind == "validation":
            excluded = dataset.train_positives[uid]
            target = dataset.validation_items[uid]
        else:
            excluded = dataset.train_positives[uid] | {dataset.validation_items[uid]}
            target = dataset.test_items[uid]
        candidates = np.array([iid for iid in range(len(dataset.items)) if iid not in excluded], dtype=np.int64)
        scores = score_function(uid, candidates)
        ranking = candidates[np.argsort(-scores, kind="stable")]
        rank = int(np.where(ranking == target)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        recalls.append(1.0 if rank <= cutoff else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0)
        top = ranking[:cutoff]
        recommended.update(int(iid) for iid in top)
        for iid in top:
            frequency = sum(iid in values for values in dataset.train_positives.values())
            novelties.append(-math.log2((frequency + 1) / (total_train + len(dataset.items))))
        pairwise = []
        for left in range(len(top)):
            left_genres = set(dataset.items[int(top[left])].genres)
            for right in range(left + 1, len(top)):
                right_genres = set(dataset.items[int(top[right])].genres)
                pairwise.append(1.0 - len(left_genres & right_genres) / len(left_genres | right_genres))
        diversity_values.append(float(np.mean(pairwise)) if pairwise else 0.0)
    return {
        f"recall_at_{cutoff}": float(np.mean(recalls)),
        f"ndcg_at_{cutoff}": float(np.mean(ndcgs)),
        "mrr": float(np.mean(reciprocal_ranks)),
        f"coverage_at_{cutoff}": len(recommended) / len(dataset.items),
        f"diversity_at_{cutoff}": float(np.mean(diversity_values)),
        f"novelty_at_{cutoff}": float(np.mean(novelties)),
    }


def evaluate(dataset, retrieval, ranker):
    quality = np.array([item.quality for item in dataset.items])
    freshness = np.array([item.freshness for item in dataset.items])

    def popularity_score(uid, candidates):
        return dataset.popularity[candidates]

    def heuristic_score(uid, candidates):
        taste = dataset.genre_matrix[candidates] @ dataset.taste[uid]
        return 0.55 * taste + 0.25 * quality[candidates] + 0.20 * freshness[candidates]

    def retrieval_score(uid, candidates):
        return retrieval.scores(uid)[candidates]

    def ranker_score(uid, candidates):
        return ranker.scores(feature_matrix(dataset, retrieval, uid, candidates))

    return {
        "popularity": _ranking_metrics(dataset, popularity_score),
        "heuristic": _ranking_metrics(dataset, heuristic_score),
        "two_tower": _ranking_metrics(dataset, retrieval_score),
        "learned_ranker": _ranking_metrics(dataset, ranker_score),
    }


def calibrate_blend(dataset, retrieval, ranker):
    """Choose ensemble weight using validation data only."""
    best = None
    for alpha in np.linspace(0.0, 1.0, 11):
        ranker.blend = float(alpha)
        metrics = _ranking_metrics(
            dataset,
            lambda uid, candidates: ranker.scores(feature_matrix(dataset, retrieval, uid, candidates)),
            target_kind="validation",
        )
        objective = (metrics["ndcg_at_5"], metrics["mrr"], metrics["recall_at_5"])
        if best is None or objective > best[0]:
            best = (objective, float(alpha), metrics)
    ranker.blend = best[1]
    return {"selected_blend": best[1], "validation_metrics": best[2]}


def train(store, output="artifacts/model", dimensions=16, epochs=35, seed=17,
          tracking_uri=None, track=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(store)
    retrieval = TwoTowerModel.initialize(len(dataset.users), len(dataset.items), dimensions, seed)
    losses = retrieval.fit(dataset.train_positives, epochs=epochs, seed=seed)
    features, labels = ranker_examples(dataset, retrieval, seed=seed + 1)
    ranker = RankerModel.fit(features, labels)
    calibration = calibrate_blend(dataset, retrieval, ranker)
    version_hash = hashlib.sha256()
    for array in (
        retrieval.user_embeddings, retrieval.item_embeddings, retrieval.item_bias,
        ranker.weights, np.array([ranker.bias]), ranker.mean, ranker.scale,
        np.array([ranker.blend]), dataset.popularity, dataset.taste, dataset.genre_matrix,
        np.array([item.quality for item in dataset.items]),
        np.array([item.freshness for item in dataset.items]),
    ):
        version_hash.update(np.ascontiguousarray(array).tobytes())
    version_hash.update(json.dumps({
        "users": dataset.users, "items": [item.item_id for item in dataset.items],
        "features": FEATURE_NAMES,
    }, sort_keys=True, separators=(",", ":")).encode())
    digest = version_hash.hexdigest()[:12]
    bundle = ModelBundle(
        retrieval, ranker, dataset.users, [item.item_id for item in dataset.items],
        FEATURE_NAMES, digest, dataset.popularity, dataset.taste, dataset.genre_matrix,
        np.array([item.quality for item in dataset.items]),
        np.array([item.freshness for item in dataset.items]),
    )
    bundle.save(output)
    metrics = evaluate(dataset, retrieval, ranker)
    split = temporal_split_manifest(dataset)
    report = {
        "model_version": digest,
        "configuration": {"dimensions": dimensions, "epochs": epochs, "seed": seed},
        "split": split,
        "training": {
            "initial_bpr_loss": losses[0], "final_bpr_loss": losses[-1],
            "ranker_examples": len(labels), "positive_rate": float(labels.mean()),
            "selected_blend": calibration["selected_blend"],
        },
        "validation": calibration["validation_metrics"],
        "metrics": metrics,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if track:
        _track_run(output, report, tracking_uri)
    return report, bundle, dataset


def _track_run(output, report, tracking_uri):
    import mlflow

    if tracking_uri is None:
        database = (Path("data") / "mlflow.db").resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{database}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("pulserank-ranking")
    with mlflow.start_run(run_name=f"ranker-{report['model_version']}") as run:
        mlflow.log_params(report["configuration"])
        mlflow.log_params({f"split_{key}": value for key, value in report["split"].items()})
        flat_metrics = {}
        for model_name, values in report["metrics"].items():
            for metric_name, value in values.items():
                flat_metrics[f"{model_name}_{metric_name}"] = value
        flat_metrics.update(report["training"])
        mlflow.log_metrics(flat_metrics)
        mlflow.set_tags({"model_version": report["model_version"], "task": "implicit-feedback-ranking"})
        mlflow.log_artifacts(str(output), artifact_path="model")
        report["mlflow"] = {"run_id": run.info.run_id, "tracking_uri": tracking_uri}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
