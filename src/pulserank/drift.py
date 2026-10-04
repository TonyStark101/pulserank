from collections import Counter

import numpy as np


def _distribution(values, epsilon=1e-12):
    values = np.asarray(values, dtype=np.float64) + epsilon
    return values / values.sum()


def _jensen_shannon(left, right):
    left, right = _distribution(left), _distribution(right)
    midpoint = 0.5 * (left + right)
    divergence = 0.5 * np.sum(left * np.log2(left / midpoint))
    divergence += 0.5 * np.sum(right * np.log2(right / midpoint))
    return float(divergence)


def detect_drift(store, bundle, warning_threshold=0.10):
    """Compare live item and genre traffic with the artifact's training state."""
    item_index = {item_id: index for index, item_id in enumerate(bundle.item_ids)}
    live_items = np.zeros(len(bundle.item_ids), dtype=np.float64)
    action_counts = Counter()
    for row in store.events():
        index = item_index.get(row["item_id"])
        if index is not None:
            live_items[index] += 1
        action_counts[row["action"]] += 1
    training_items = bundle.popularity
    item_js = _jensen_shannon(training_items, live_items)
    observed = int(live_items.sum())
    status = "warning" if observed and item_js >= warning_threshold else "healthy"
    return {
        "status": status, "model_version": bundle.version, "observed_events": observed,
        "item_distribution_js": item_js, "warning_threshold": warning_threshold,
        "action_counts": dict(sorted(action_counts.items())),
    }
