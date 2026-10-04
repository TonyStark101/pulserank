import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional

from .models import Event, Item


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS items (
  item_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  genres TEXT NOT NULL,
  year INTEGER NOT NULL,
  quality REAL NOT NULL,
  freshness REAL NOT NULL,
  color TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  item_id TEXT NOT NULL REFERENCES items(item_id),
  action TEXT NOT NULL CHECK(action IN ('impression','play','complete','like','skip')),
  timestamp REAL NOT NULL,
  watch_pct REAL NOT NULL CHECK(watch_pct >= 0 AND watch_pct <= 1),
  payload_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_user_time ON events(user_id, timestamp);
CREATE INDEX IF NOT EXISTS events_item_time ON events(item_id, timestamp);
CREATE TABLE IF NOT EXISTS exposures (
  request_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  item_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  experiment TEXT NOT NULL,
  variant TEXT NOT NULL,
  score REAL NOT NULL,
  timestamp REAL NOT NULL,
  model_version TEXT,
  serving_mode TEXT NOT NULL DEFAULT 'heuristic',
  PRIMARY KEY(request_id, item_id)
);
"""


class ConflictError(Exception):
    pass


class Store:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)
            columns = {row[1] for row in con.execute("PRAGMA table_info(exposures)")}
            if "model_version" not in columns:
                con.execute("ALTER TABLE exposures ADD COLUMN model_version TEXT")
            if "serving_mode" not in columns:
                con.execute("ALTER TABLE exposures ADD COLUMN serving_mode TEXT NOT NULL DEFAULT 'heuristic'")

    @contextmanager
    def connect(self):
        con = sqlite3.connect(str(self.path), timeout=10)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def put_items(self, items: Iterable[Item]) -> None:
        with self.connect() as con:
            con.executemany(
                "INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?,?)",
                [(i.item_id, i.title, json.dumps(i.genres), i.year, i.quality, i.freshness, i.color) for i in items],
            )

    def items(self) -> List[Item]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM items ORDER BY item_id").fetchall()
        return [self._item(row) for row in rows]

    def item(self, item_id: str) -> Optional[Item]:
        with self.connect() as con:
            row = con.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
        return self._item(row) if row else None

    def ingest(self, events: Iterable[Event]) -> tuple:
        accepted = duplicates = 0
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            for event in events:
                payload = json.dumps(event.__dict__, sort_keys=True, separators=(",", ":"))
                existing = con.execute(
                    "SELECT payload_hash FROM events WHERE event_id=?", (event.event_id,)
                ).fetchone()
                if existing:
                    if existing["payload_hash"] != payload:
                        raise ConflictError(f"event ID {event.event_id!r} was reused with different content")
                    duplicates += 1
                    continue
                con.execute(
                    "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                    (event.event_id, event.user_id, event.item_id, event.action,
                     event.timestamp, event.watch_pct, payload),
                )
                accepted += 1
        return accepted, duplicates

    def events(self, user_id: Optional[str] = None, as_of: Optional[float] = None):
        clauses, params = [], []
        if user_id is not None:
            clauses.append("user_id=?")
            params.append(user_id)
        if as_of is not None:
            clauses.append("timestamp<=?")
            params.append(as_of)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as con:
            return con.execute("SELECT * FROM events" + where + " ORDER BY timestamp, event_id", params).fetchall()

    def event_log(self, after_offset: int = 0, limit: int = 1000):
        """Read the accepted-event log in ingestion order.

        SQLite's rowid acts as the local source offset. The distributed pipeline
        maps this contract to Kafka partition/offset pairs.
        """
        with self.connect() as con:
            return con.execute(
                "SELECT rowid AS source_offset, * FROM events "
                "WHERE rowid>? ORDER BY rowid LIMIT ?",
                (after_offset, limit),
            ).fetchall()

    def record_exposures(self, rows) -> None:
        with self.connect() as con:
            con.executemany(
                "INSERT OR IGNORE INTO exposures "
                "(request_id,user_id,item_id,position,experiment,variant,score,timestamp,model_version,serving_mode) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def exposures(self, experiment=None):
        query, params = "SELECT * FROM exposures", []
        if experiment is not None:
            query += " WHERE experiment=?"
            params.append(experiment)
        query += " ORDER BY timestamp, request_id, position"
        with self.connect() as con:
            return con.execute(query, params).fetchall()

    def stats(self):
        with self.connect() as con:
            return {
                "items": con.execute("SELECT COUNT(*) FROM items").fetchone()[0],
                "events": con.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "users": con.execute("SELECT COUNT(DISTINCT user_id) FROM events").fetchone()[0],
                "exposures": con.execute("SELECT COUNT(*) FROM exposures").fetchone()[0],
            }

    @staticmethod
    def _item(row) -> Item:
        return Item(row["item_id"], row["title"], json.loads(row["genres"]), row["year"],
                    row["quality"], row["freshness"], row["color"])
