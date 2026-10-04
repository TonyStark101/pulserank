import math
import time
from collections import defaultdict


POSITIVE_ACTIONS = {"like", "complete"}


def analyze_experiment(store, experiment="ranker-v1", window_days=7, as_of=None):
    """Attribute request-level outcomes and return a CUPED-adjusted experiment readout."""
    as_of = time.time() if as_of is None else as_of
    window = window_days * 86_400
    requests = {}
    for row in store.exposures(experiment):
        if row["timestamp"] + window > as_of:
            continue
        value = requests.setdefault(row["request_id"], {
            "user_id": row["user_id"], "variant": row["variant"],
            "timestamp": row["timestamp"], "items": set(),
        })
        value["items"].add(row["item_id"])

    user_events = defaultdict(list)
    for row in store.events(as_of=as_of):
        user_events[row["user_id"]].append(row)
    observations = []
    for request in requests.values():
        timestamp = request["timestamp"]
        events = user_events[request["user_id"]]
        pre_period = sum(
            row["action"] in POSITIVE_ACTIONS
            for row in events if timestamp - window <= row["timestamp"] < timestamp
        )
        converted = any(
            row["item_id"] in request["items"] and row["action"] in POSITIVE_ACTIONS
            and timestamp < row["timestamp"] <= timestamp + window
            for row in events
        )
        observations.append({"variant": request["variant"], "outcome": float(converted),
                             "pre_period_engagement": float(pre_period)})

    if not observations:
        return {"experiment": experiment, "status": "insufficient_data", "mature_requests": 0}
    outcomes = [row["outcome"] for row in observations]
    covariates = [row["pre_period_engagement"] for row in observations]
    mean_x = sum(covariates) / len(covariates)
    variance_x = sum((value - mean_x) ** 2 for value in covariates) / len(covariates)
    mean_y = sum(outcomes) / len(outcomes)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(covariates, outcomes)) / len(outcomes)
    theta = covariance / variance_x if variance_x > 0 else 0.0
    groups = defaultdict(list)
    for row in observations:
        adjusted = row["outcome"] - theta * (row["pre_period_engagement"] - mean_x)
        groups[row["variant"]].append((row["outcome"], adjusted))

    variants = {}
    for name, values in sorted(groups.items()):
        raw = [value[0] for value in values]
        adjusted = [value[1] for value in values]
        mean = sum(adjusted) / len(adjusted)
        variance = sum((value - mean) ** 2 for value in adjusted) / max(1, len(adjusted) - 1)
        variants[name] = {
            "requests": len(values), "conversion_rate": sum(raw) / len(raw),
            "cuped_conversion_rate": mean, "standard_error": math.sqrt(variance / len(adjusted)),
        }

    comparison = None
    if {"control", "challenger"}.issubset(variants):
        control, challenger = variants["control"], variants["challenger"]
        difference = challenger["cuped_conversion_rate"] - control["cuped_conversion_rate"]
        standard_error = math.sqrt(control["standard_error"] ** 2 + challenger["standard_error"] ** 2)
        z_score = difference / standard_error if standard_error else 0.0
        enough = min(control["requests"], challenger["requests"]) >= 30
        comparison = {
            "absolute_lift": difference,
            "relative_lift": difference / control["cuped_conversion_rate"] if control["cuped_conversion_rate"] else None,
            "z_score": z_score,
            "sequential_guardrail": "stop" if enough and difference < 0 and z_score <= -2.576 else "continue",
            "minimum_sample_reached": enough,
        }
    return {
        "experiment": experiment, "status": "ready", "mature_requests": len(observations),
        "attribution_window_days": window_days, "cuped_theta": theta,
        "variants": variants, "comparison": comparison,
    }
