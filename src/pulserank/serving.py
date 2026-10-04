from dataclasses import asdict
from pathlib import Path

import numpy as np

from .ml.model import ModelBundle
from .models import ServingDecision


class ModelRuntime:
    """Validated, fail-safe boundary between a model artifact and the API."""

    def __init__(self, bundle=None, source=None, load_error=None):
        self.bundle = bundle
        self.source = str(source) if source is not None else None
        self.load_error = load_error
        self._users = {value: index for index, value in enumerate(bundle.user_ids)} if bundle else {}
        self._items = {value: index for index, value in enumerate(bundle.item_ids)} if bundle else {}
        self.ann = None
        if bundle:
            try:
                from .ann import HNSWIndex
                self.ann = HNSWIndex(bundle.retrieval)
            except ImportError:
                pass

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        try:
            bundle = ModelBundle.load(directory)
            if not bundle.serving_ready:
                raise ValueError("artifact predates the online-serving feature contract")
            return cls(bundle=bundle, source=directory)
        except (OSError, ValueError, KeyError) as exc:
            return cls(source=directory, load_error=str(exc))

    def status(self):
        if self.bundle is None:
            return {
                "ready": False, "model_version": None, "source": self.source,
                "error": self.load_error or "no model configured",
            }
        return {
            "ready": True, "model_version": self.bundle.version, "source": self.source,
            "users": len(self.bundle.user_ids), "items": len(self.bundle.item_ids),
            "embedding_dimensions": int(self.bundle.retrieval.user_embeddings.shape[1]),
            "retrieval_engine": "hnsw" if self.ann else "exact_numpy",
        }

    def decision(self, store, user_id, variant, candidate_count=0):
        if variant != "challenger":
            return ServingDecision("heuristic_control", self.version, None, candidate_count)
        if self.bundle is None:
            return ServingDecision("heuristic_fallback", None,
                                   self.load_error or "no model configured", candidate_count)
        if user_id not in self._users:
            return ServingDecision("heuristic_fallback", self.version, "unknown user", candidate_count)
        catalog = {item.item_id for item in store.items()}
        if catalog != set(self._items):
            return ServingDecision("heuristic_fallback", self.version, "catalog mismatch", candidate_count)
        return ServingDecision("learned_challenger", self.version, None, candidate_count)

    @property
    def version(self):
        return self.bundle.version if self.bundle else None

    def scores(self, user_id, item_ids):
        uid = self._users[user_id]
        indices = np.asarray([self._items[item_id] for item_id in item_ids], dtype=np.int64)
        retrieval = self.bundle.retrieval.scores(uid)[indices]
        rank = self.bundle.ranker.scores(self.bundle.features(uid, indices))
        return retrieval, rank

    def retrieve(self, user_id, allowed_item_ids, count):
        uid = self._users[user_id]
        allowed_indices = [self._items[item_id] for item_id in allowed_item_ids]
        if self.ann:
            indices = self.ann.query(uid, count, allowed_indices)
        else:
            all_scores = self.bundle.retrieval.scores(uid)
            indices = np.asarray(sorted(
                allowed_indices, key=lambda index: (-all_scores[index], self.bundle.item_ids[index])
            )[:count], dtype=np.int64)
        item_ids = [self.bundle.item_ids[int(index)] for index in indices]
        scores = self.bundle.retrieval.scores(uid)[indices]
        return item_ids, scores

    @staticmethod
    def serialize(decision):
        return asdict(decision)
