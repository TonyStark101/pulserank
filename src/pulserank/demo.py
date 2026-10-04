import random

from .models import Event, Item


TITLES = [
    ("Neon Divide", ["Sci-Fi", "Thriller"]), ("After the Signal", ["Sci-Fi", "Drama"]),
    ("Last Light in Kyoto", ["Drama", "Romance"]), ("The Glass Algorithm", ["Thriller", "Mystery"]),
    ("Wild North", ["Documentary", "Nature"]), ("Red Horizon", ["Sci-Fi", "Action"]),
    ("Second Serve", ["Sports", "Drama"]), ("Midnight Circuit", ["Action", "Thriller"]),
    ("Small Gods", ["Fantasy", "Drama"]), ("The Long Table", ["Comedy", "Drama"]),
    ("Deep Current", ["Documentary", "Nature"]), ("Paper Kingdom", ["Fantasy", "Animation"]),
    ("Zero Day Summer", ["Comedy", "Romance"]), ("Borrowed Time", ["Mystery", "Drama"]),
    ("Ghost Frequency", ["Horror", "Mystery"]), ("Apex", ["Sports", "Documentary"]),
    ("Parallel Hearts", ["Romance", "Sci-Fi"]), ("The Quiet Heist", ["Crime", "Comedy"]),
    ("Fireline", ["Action", "Drama"]), ("Blue Hour", ["Mystery", "Thriller"]),
    ("Orbiters", ["Animation", "Sci-Fi"]), ("No Fixed Address", ["Comedy", "Drama"]),
    ("The Ninth Door", ["Horror", "Thriller"]), ("Velocity", ["Sports", "Action"]),
]
COLORS = ["#6957ff", "#ef5da8", "#00c2a8", "#ff9f43", "#2e86de", "#e74c3c"]


def seed(store, now=None):
    # A fixed event-time anchor makes `pulserank demo` exactly replayable: running
    # it twice produces idempotent duplicates instead of subtly different data.
    now = 1_767_225_600.0 if now is None else now
    rng = random.Random(42)
    items = [
        Item(f"title-{i:02d}", title, genres, 2021 + i % 6,
             round(0.58 + rng.random() * 0.39, 3),
             round(0.35 + rng.random() * 0.64, 3), COLORS[i % len(COLORS)])
        for i, (title, genres) in enumerate(TITLES, 1)
    ]
    store.put_items(items)
    events = []
    personas = {
        "maya": {"Sci-Fi", "Thriller", "Mystery"},
        "leo": {"Comedy", "Romance", "Drama"},
        "nora": {"Documentary", "Nature", "Sports"},
        "sam": {"Action", "Fantasy", "Animation"},
    }
    for user_idx, (user, preferred) in enumerate(personas.items()):
        for step in range(42):
            item = items[(step * 5 + user_idx * 3) % len(items)]
            relevant = bool(preferred.intersection(item.genres))
            action = rng.choice(["like", "complete", "play"]) if relevant else rng.choice(["skip", "play", "impression"])
            watch = rng.uniform(0.72, 1.0) if relevant else rng.uniform(0.02, 0.42)
            events.append(Event(f"seed-{user}-{step}", user, item.item_id, action,
                                now - 7200 + step * 113 + user_idx, round(watch, 3)))
    store.ingest(events)
    return {"items": len(items), "events": len(events), "personas": len(personas)}
