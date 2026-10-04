import math
import time
import uuid
from collections import Counter

from .experiments import assign
from .features import user_features
from .models import Recommendation, ServingDecision


def recommend(store, user_id: str, limit: int = 12, as_of: float = None,
              model_runtime=None, include_serving=False):
    as_of = time.time() if as_of is None else as_of
    features = user_features(store, user_id, as_of)
    assignment = assign(user_id)
    watched = set(features.recent_items)

    popularity = Counter(row["item_id"] for row in store.events(as_of=as_of) if row["action"] in ("play", "complete", "like"))
    max_pop = max(popularity.values(), default=1)
    available = [item for item in store.items() if item.item_id not in watched]
    serving = model_runtime.decision(store, user_id, assignment.variant, len(available)) if model_runtime else None
    learned_scores = {}
    if serving and serving.mode == "learned_challenger":
        item_ids = [item.item_id for item in available]
        retrieval_values, rank_values = model_runtime.scores(user_id, item_ids)
        candidate_limit = min(len(item_ids), max(20, limit * 4))
        selected_indices = sorted(
            range(len(item_ids)), key=lambda index: (-retrieval_values[index], item_ids[index])
        )[:candidate_limit]
        learned_scores = {
            item_ids[index]: (float(retrieval_values[index]), float(rank_values[index]))
            for index in selected_indices
        }
        available = [item for item in available if item.item_id in learned_scores]
        serving = model_runtime.decision(store, user_id, assignment.variant, len(available))

    candidates = []
    for item in available:
        if item.item_id in watched:
            continue
        taste = sum(features.genre_affinity.get(g, 0.0) for g in item.genres) / max(1, len(item.genres))
        pop = math.log1p(popularity[item.item_id]) / math.log1p(max_pop) if popularity[item.item_id] else 0.0
        retrieval = 0.70 * taste + 0.30 * pop
        if features.event_count == 0:
            retrieval = 0.65 * item.quality + 0.35 * item.freshness

        if item.item_id in learned_scores:
            retrieval, rank_score = learned_scores[item.item_id]
        elif assignment.variant == "challenger":
            rank_score = 0.55 * retrieval + 0.27 * item.quality + 0.18 * item.freshness
        else:
            rank_score = 0.62 * retrieval + 0.38 * item.quality
        reasons = _reasons(item, features, popularity[item.item_id])
        candidates.append(Recommendation(item, rank_score, reasons, retrieval, rank_score))

    # Greedy reranking penalizes repeated genres while retaining score relevance.
    selected, genre_counts = [], Counter()
    while candidates and len(selected) < limit:
        def adjusted(rec):
            repetition = sum(genre_counts[g] for g in rec.item.genres) / len(rec.item.genres)
            return rec.rank_score - 0.075 * repetition
        winner = max(candidates, key=adjusted)
        final_score = adjusted(winner)
        selected.append(Recommendation(winner.item, final_score, winner.reasons,
                                       winner.retrieval_score, winner.rank_score))
        genre_counts.update(winner.item.genres)
        candidates.remove(winner)

    request_id = uuid.uuid4().hex
    store.record_exposures([
        (request_id, user_id, rec.item.item_id, pos, assignment.experiment,
         assignment.variant, rec.score, as_of,
         serving.model_version if serving else None,
         serving.mode if serving else "heuristic")
        for pos, rec in enumerate(selected, 1)
    ])
    result = (request_id, assignment, features, selected)
    if include_serving:
        if serving is None:
            serving = ServingDecision("heuristic", None, "no model configured", len(available))
        return result + (serving,)
    return result


def _reasons(item, features, popularity):
    matched = sorted(item.genres, key=lambda g: features.genre_affinity.get(g, 0), reverse=True)
    reasons = []
    if matched and features.genre_affinity.get(matched[0], 0) > 0:
        reasons.append(f"Matches your interest in {matched[0]}")
    if item.freshness > 0.78:
        reasons.append("Fresh release")
    if popularity > 2:
        reasons.append("Trending now")
    if not reasons:
        reasons.append("Highly rated for discovery")
    return reasons[:2]
