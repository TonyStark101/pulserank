import threading
from collections import Counter


class ServingMetrics:
    """Small Prometheus-compatible metrics registry with no runtime dependency."""

    BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)

    def __init__(self):
        self._lock = threading.Lock()
        self.requests = Counter()
        self.fallbacks = Counter()
        self.latencies = []

    def observe(self, decision, latency_seconds):
        with self._lock:
            self.requests[decision.mode] += 1
            if decision.fallback_reason:
                self.fallbacks[decision.fallback_reason] += 1
            self.latencies.append(latency_seconds)

    def render(self):
        with self._lock:
            requests = self.requests.copy()
            fallbacks = self.fallbacks.copy()
            latencies = list(self.latencies)
        lines = [
            "# HELP pulserank_recommendation_requests_total Recommendation requests by serving mode.",
            "# TYPE pulserank_recommendation_requests_total counter",
        ]
        for mode, count in sorted(requests.items()):
            lines.append(f'pulserank_recommendation_requests_total{{mode="{mode}"}} {count}')
        lines.extend([
            "# HELP pulserank_model_fallbacks_total Learned-model fallbacks by reason.",
            "# TYPE pulserank_model_fallbacks_total counter",
        ])
        for reason, count in sorted(fallbacks.items()):
            safe = reason.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'pulserank_model_fallbacks_total{{reason="{safe}"}} {count}')
        lines.extend([
            "# HELP pulserank_recommendation_latency_seconds Recommendation latency.",
            "# TYPE pulserank_recommendation_latency_seconds histogram",
        ])
        for bucket in self.BUCKETS:
            count = sum(value <= bucket for value in latencies)
            lines.append(f'pulserank_recommendation_latency_seconds_bucket{{le="{bucket}"}} {count}')
        lines.append(f'pulserank_recommendation_latency_seconds_bucket{{le="+Inf"}} {len(latencies)}')
        lines.append(f"pulserank_recommendation_latency_seconds_sum {sum(latencies)}")
        lines.append(f"pulserank_recommendation_latency_seconds_count {len(latencies)}")
        return "\n".join(lines) + "\n"
