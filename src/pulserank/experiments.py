import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Assignment:
    experiment: str
    variant: str
    bucket: int


def assign(user_id: str, experiment: str = "ranker-v1", allocation: int = 50) -> Assignment:
    """Return a stable, reproducible experiment assignment.

    The experiment name is part of the hash so future experiments do not inherit
    old assignments. A 10,000 bucket space supports precise traffic ramps.
    """
    if not 0 <= allocation <= 100:
        raise ValueError("allocation must be between 0 and 100")
    digest = hashlib.sha256(f"pulserank:{experiment}:{user_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return Assignment(experiment, "challenger" if bucket < allocation * 100 else "control", bucket)

