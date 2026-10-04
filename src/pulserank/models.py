from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Item:
    item_id: str
    title: str
    genres: List[str]
    year: int
    quality: float
    freshness: float
    color: str


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    item_id: str
    action: str
    timestamp: float
    watch_pct: float = 0.0


@dataclass(frozen=True)
class UserFeatures:
    user_id: str
    as_of: float
    genre_affinity: Dict[str, float]
    recent_items: List[str]
    event_count: int


@dataclass(frozen=True)
class Recommendation:
    item: Item
    score: float
    reasons: List[str]
    retrieval_score: float
    rank_score: float


@dataclass(frozen=True)
class ServingDecision:
    mode: str
    model_version: Optional[str]
    fallback_reason: Optional[str]
    candidate_count: int
