from collections import defaultdict

from .models import UserFeatures


ACTION_WEIGHT = {
    "impression": 0.0,
    "play": 0.8,
    "complete": 2.2,
    "like": 3.0,
    "skip": -2.5,
}


def user_features(store, user_id: str, as_of: float) -> UserFeatures:
    """Build leakage-safe features using only events visible at ``as_of``."""
    rows = store.events(user_id=user_id, as_of=as_of)
    affinity = defaultdict(float)
    recent = []
    for row in rows:
        item = store.item(row["item_id"])
        if not item:
            continue
        weight = ACTION_WEIGHT[row["action"]]
        if row["action"] in ("play", "complete"):
            weight *= max(0.15, row["watch_pct"])
        for genre in item.genres:
            affinity[genre] += weight / len(item.genres)
        if row["action"] != "impression":
            recent.append(row["item_id"])
    scale = max(1.0, sum(abs(v) for v in affinity.values()))
    normalized = {genre: round(value / scale, 6) for genre, value in affinity.items()}
    return UserFeatures(user_id, as_of, normalized, recent[-12:], len(rows))

