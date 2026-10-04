import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


@dataclass
class TwoTowerModel:
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    item_bias: np.ndarray

    @classmethod
    def initialize(cls, users, items, dimensions, seed):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(dimensions)
        return cls(
            rng.normal(0, scale, (users, dimensions)),
            rng.normal(0, scale, (items, dimensions)),
            np.zeros(items, dtype=np.float64),
        )

    def fit(self, train_positives, epochs=30, learning_rate=0.035, regularization=0.002, seed=17):
        rng = np.random.default_rng(seed)
        all_items = np.arange(self.item_embeddings.shape[0])
        pairs = np.array([(uid, iid) for uid, values in train_positives.items() for iid in values], dtype=np.int64)
        losses = []
        for _ in range(epochs):
            rng.shuffle(pairs)
            epoch_loss = 0.0
            for uid, positive in pairs:
                blocked = train_positives[int(uid)]
                negative = int(rng.choice(all_items))
                while negative in blocked:
                    negative = int(rng.choice(all_items))
                user = self.user_embeddings[uid].copy()
                pos = self.item_embeddings[positive].copy()
                neg = self.item_embeddings[negative].copy()
                margin = np.dot(user, pos - neg) + self.item_bias[positive] - self.item_bias[negative]
                gradient = float(_sigmoid(-margin))
                epoch_loss += float(np.logaddexp(0.0, -margin))
                self.user_embeddings[uid] += learning_rate * (gradient * (pos - neg) - regularization * user)
                self.item_embeddings[positive] += learning_rate * (gradient * user - regularization * pos)
                self.item_embeddings[negative] += learning_rate * (-gradient * user - regularization * neg)
                self.item_bias[positive] += learning_rate * (gradient - regularization * self.item_bias[positive])
                self.item_bias[negative] += learning_rate * (-gradient - regularization * self.item_bias[negative])
            losses.append(epoch_loss / max(1, len(pairs)))
        return losses

    def scores(self, user_index):
        return self.item_embeddings @ self.user_embeddings[user_index] + self.item_bias


@dataclass
class RankerModel:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    blend: float = 1.0

    @classmethod
    def fit(cls, features, labels, epochs=500, learning_rate=0.08, regularization=0.002):
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalized = (features - mean) / scale
        weights = np.zeros(features.shape[1], dtype=np.float64)
        bias = 0.0
        for _ in range(epochs):
            predictions = _sigmoid(normalized @ weights + bias)
            error = predictions - labels
            weights -= learning_rate * ((normalized.T @ error) / len(labels) + regularization * weights)
            bias -= learning_rate * float(error.mean())
        return cls(weights, bias, mean, scale, 1.0)

    def scores(self, features):
        normalized = (features - self.mean) / self.scale
        learned = _sigmoid(normalized @ self.weights + self.bias)
        heuristic = 0.55 * features[:, 4] + 0.25 * features[:, 1] + 0.20 * features[:, 2]
        return self.blend * learned + (1.0 - self.blend) * heuristic


@dataclass
class ModelBundle:
    retrieval: TwoTowerModel
    ranker: RankerModel
    user_ids: list
    item_ids: list
    feature_names: list
    version: str
    popularity: Optional[np.ndarray] = None
    taste: Optional[np.ndarray] = None
    genre_matrix: Optional[np.ndarray] = None
    item_quality: Optional[np.ndarray] = None
    item_freshness: Optional[np.ndarray] = None

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            user_embeddings=self.retrieval.user_embeddings,
            item_embeddings=self.retrieval.item_embeddings,
            item_bias=self.retrieval.item_bias,
            ranker_weights=self.ranker.weights,
            ranker_bias=np.array([self.ranker.bias]),
            ranker_mean=self.ranker.mean,
            ranker_scale=self.ranker.scale,
            ranker_blend=np.array([self.ranker.blend]),
        )
        for name in ("popularity", "taste", "genre_matrix", "item_quality", "item_freshness"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        np.savez_compressed(directory / "weights.npz", **arrays)
        manifest = {
            "format_version": 2,
            "model_version": self.version,
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "feature_names": self.feature_names,
            "embedding_dimensions": int(self.retrieval.user_embeddings.shape[1]),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        arrays = np.load(directory / "weights.npz")
        retrieval = TwoTowerModel(arrays["user_embeddings"], arrays["item_embeddings"], arrays["item_bias"])
        blend = float(arrays["ranker_blend"][0]) if "ranker_blend" in arrays else 1.0
        ranker = RankerModel(arrays["ranker_weights"], float(arrays["ranker_bias"][0]),
                             arrays["ranker_mean"], arrays["ranker_scale"], blend)
        return cls(retrieval, ranker, manifest["user_ids"], manifest["item_ids"],
                   manifest["feature_names"], manifest["model_version"],
                   *[arrays[name] if name in arrays else None for name in (
                       "popularity", "taste", "genre_matrix", "item_quality", "item_freshness"
                   )])

    @property
    def serving_ready(self):
        return all(value is not None for value in (
            self.popularity, self.taste, self.genre_matrix,
            self.item_quality, self.item_freshness,
        ))

    def features(self, user_index, item_indices):
        if not self.serving_ready:
            raise ValueError("model artifact does not contain online feature state")
        item_indices = np.asarray(item_indices, dtype=np.int64)
        retrieval_scores = self.retrieval.scores(user_index)[item_indices]
        quality = self.item_quality[item_indices]
        freshness = self.item_freshness[item_indices]
        taste = self.genre_matrix[item_indices] @ self.taste[user_index]
        return np.column_stack([
            retrieval_scores, quality, freshness, self.popularity[item_indices], taste,
            quality * retrieval_scores,
        ])
