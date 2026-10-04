import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from ..demo import seed as seed_catalog
from ..models import Event, Item


POSITIVE_ACTIONS = {"like", "complete"}


@dataclass
class Dataset:
    users: List[str]
    items: List[Item]
    user_index: Dict[str, int]
    item_index: Dict[str, int]
    train_positives: Dict[int, Set[int]]
    train_negatives: Dict[int, Set[int]]
    validation_items: Dict[int, int]
    test_items: Dict[int, int]
    train_cutoffs: Dict[int, float]
    popularity: np.ndarray
    taste: np.ndarray
    genre_matrix: np.ndarray
    genres: List[str]


def seed_ml_population(store, users=240, random_seed=73):
    """Create a deterministic implicit-feedback population with latent tastes."""
    seed_catalog(store)
    items = store.items()
    genres = sorted({genre for item in items for genre in item.genres})
    events = []
    base_time = 1_767_312_000.0
    for user_number in range(users):
        user_id = f"viewer-{user_number:04d}"
        rng = random.Random(random_seed + user_number * 7_919)
        preferred = set(rng.sample(genres, 3))
        latent_cluster = user_number % 4
        order = list(items)
        rng.shuffle(order)
        raw_scores = []
        for item in order:
            overlap = len(preferred.intersection(item.genres))
            catalog_position = int(item.item_id.rsplit("-", 1)[-1]) - 1
            collaborative_signal = 1.35 if catalog_position % 4 == latent_cluster else 0.0
            raw_scores.append((
                1.35 * overlap + collaborative_signal + 0.65 * item.quality + rng.uniform(-0.55, 0.55),
                item,
            ))
        forced_positive = {item.item_id for _, item in sorted(raw_scores, reverse=True, key=lambda pair: pair[0])[:6]}
        for position, (latent_score, item) in enumerate(raw_scores):
            probability = 1.0 / (1.0 + math.exp(-(latent_score - 1.15)))
            positive = item.item_id in forced_positive or rng.random() < probability
            if positive:
                action = rng.choices(["like", "complete", "play"], weights=[4, 3, 1])[0]
                watch_pct = rng.uniform(0.72, 1.0)
            else:
                action = rng.choices(["skip", "impression", "play"], weights=[5, 3, 1])[0]
                watch_pct = rng.uniform(0.01, 0.38)
            events.append(Event(
                f"ml-{user_id}-{item.item_id}", user_id, item.item_id, action,
                base_time + user_number * 1_000 + position * 17,
                round(watch_pct, 4),
            ))
    accepted, duplicates = store.ingest(events)
    return {"users": users, "events": len(events), "accepted": accepted, "duplicates": duplicates}


def build_dataset(store, minimum_positives=5):
    items = store.items()
    item_index = {item.item_id: index for index, item in enumerate(items)}
    positives = defaultdict(dict)
    for row in store.events():
        is_positive = row["action"] in POSITIVE_ACTIONS or (
            row["action"] == "play" and row["watch_pct"] >= 0.6
        )
        if is_positive and row["item_id"] in item_index:
            positives[row["user_id"]][row["item_id"]] = max(
                row["timestamp"], positives[row["user_id"]].get(row["item_id"], float("-inf"))
            )

    eligible = {
        user: sorted(values.items(), key=lambda pair: (pair[1], pair[0]))
        for user, values in positives.items() if len(values) >= minimum_positives
    }
    users = sorted(eligible)
    if not users:
        raise ValueError("no users have enough positive interactions for a temporal split")
    user_index = {user: index for index, user in enumerate(users)}
    train, validation, test, cutoffs = {}, {}, {}, {}
    popularity = np.zeros(len(items), dtype=np.float64)
    for user in users:
        uid = user_index[user]
        ordered = eligible[user]
        train[uid] = {item_index[item_id] for item_id, _ in ordered[:-2]}
        validation[uid] = item_index[ordered[-2][0]]
        test[uid] = item_index[ordered[-1][0]]
        cutoffs[uid] = ordered[-3][1]
        for iid in train[uid]:
            popularity[iid] += 1
    popularity = np.log1p(popularity)
    if popularity.max() > 0:
        popularity /= popularity.max()

    genres = sorted({genre for item in items for genre in item.genres})
    genre_index = {genre: index for index, genre in enumerate(genres)}
    genre_matrix = np.zeros((len(items), len(genres)), dtype=np.float64)
    for iid, item in enumerate(items):
        for genre in item.genres:
            genre_matrix[iid, genre_index[genre]] = 1.0
        genre_matrix[iid] /= max(1.0, genre_matrix[iid].sum())
    taste = np.zeros((len(users), len(genres)), dtype=np.float64)
    for uid, item_ids in train.items():
        taste[uid] = genre_matrix[list(item_ids)].mean(axis=0)

    negatives = {uid: set() for uid in train}
    for row in store.events():
        uid = user_index.get(row["user_id"])
        iid = item_index.get(row["item_id"])
        if uid is None or iid is None or row["timestamp"] > cutoffs[uid]:
            continue
        is_negative = row["action"] in {"skip", "impression"} or (
            row["action"] == "play" and row["watch_pct"] < 0.4
        )
        if is_negative and iid not in train[uid]:
            negatives[uid].add(iid)

    return Dataset(users, items, user_index, item_index, train, negatives,
                   validation, test, cutoffs, popularity, taste, genre_matrix, genres)


def temporal_split_manifest(dataset):
    return {
        "users": len(dataset.users),
        "items": len(dataset.items),
        "train_interactions": sum(len(values) for values in dataset.train_positives.values()),
        "validation_interactions": len(dataset.validation_items),
        "test_interactions": len(dataset.test_items),
        "minimum_train_cutoff": min(dataset.train_cutoffs.values()),
        "maximum_train_cutoff": max(dataset.train_cutoffs.values()),
    }
