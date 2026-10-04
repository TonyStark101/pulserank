"""Deterministic event-time reference pipeline.

This module models the semantics expected from the Spark/Delta deployment while
remaining runnable in the dependency-free laptop demo. JSONL files represent
append-only bronze/silver/dead-letter tables and an atomic JSON snapshot
represents the materialized gold feature table.
"""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from .features import ACTION_WEIGHT


class VerificationError(Exception):
    pass


class EventTimePipeline:
    def __init__(self, store, lake_path, watermark_seconds=600):
        if watermark_seconds < 0:
            raise ValueError("watermark_seconds cannot be negative")
        self.store = store
        self.root = Path(lake_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.watermark_seconds = float(watermark_seconds)
        self.paths = {
            "bronze": self.root / "bronze" / "events.jsonl",
            "silver": self.root / "silver" / "events.jsonl",
            "late": self.root / "quarantine" / "late_events.jsonl",
            "gold": self.root / "gold" / "user_genre_features.json",
            "checkpoint": self.root / "_checkpoints" / "events.json",
        }
        for path in self.paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, batch_size=1000):
        checkpoint = self._checkpoint()
        rows = self.store.event_log(checkpoint["source_offset"], batch_size)
        if not rows:
            return self.metrics()

        bronze_offsets = self._existing(self.paths["bronze"], "source_offset")
        silver_ids = self._existing(self.paths["silver"], "event_id")
        late_ids = self._existing(self.paths["late"], "event_id")
        max_event_time = checkpoint["max_event_time"]

        for row in rows:
            record = {key: row[key] for key in (
                "source_offset", "event_id", "user_id", "item_id", "action", "timestamp", "watch_pct"
            )}
            if record["source_offset"] not in bronze_offsets:
                self._append(self.paths["bronze"], record)

            candidate_max = record["timestamp"] if max_event_time is None else max(max_event_time, record["timestamp"])
            watermark = candidate_max - self.watermark_seconds
            if record["timestamp"] < watermark:
                if record["event_id"] not in late_ids:
                    self._append(self.paths["late"], {
                        **record, "watermark": watermark,
                        "lateness_seconds": round(watermark - record["timestamp"], 6),
                    })
            elif record["event_id"] not in silver_ids:
                self._append(self.paths["silver"], {**record, "repaired": False})

            max_event_time = candidate_max
            checkpoint = {
                "source_offset": record["source_offset"],
                "max_event_time": max_event_time,
                "watermark": max_event_time - self.watermark_seconds,
            }
            self._atomic_json(self.paths["checkpoint"], checkpoint)

        self._rebuild_gold()
        return self.metrics()

    def run_all(self, batch_size=1000):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        while True:
            before = self._checkpoint()["source_offset"]
            result = self.run(batch_size=batch_size)
            if result["source_offset"] == before:
                return result

    def repair_late(self):
        silver_ids = self._existing(self.paths["silver"], "event_id")
        repaired = 0
        for record in sorted(self._read(self.paths["late"]), key=lambda r: (r["timestamp"], r["event_id"])):
            if record["event_id"] not in silver_ids:
                clean = {key: record[key] for key in (
                    "source_offset", "event_id", "user_id", "item_id", "action", "timestamp", "watch_pct"
                )}
                self._append(self.paths["silver"], {**clean, "repaired": True})
                silver_ids.add(record["event_id"])
                repaired += 1
        self._rebuild_gold()
        return {"repaired": repaired, **self.metrics()}

    def verify(self):
        expected = self._gold_rows()
        actual = self._read_json(self.paths["gold"], default={"features": []})["features"]
        if expected != actual:
            raise VerificationError("gold snapshot does not match deterministic replay of silver events")
        event_ids = [row["event_id"] for row in self._read(self.paths["silver"])]
        if len(event_ids) != len(set(event_ids)):
            raise VerificationError("silver table contains duplicate event IDs")
        return {
            "verified": True,
            "silver_events": len(event_ids),
            "gold_rows": len(actual),
            "gold_sha256": self._digest(actual),
        }

    def metrics(self):
        checkpoint = self._checkpoint()
        gold = self._read_json(self.paths["gold"], default={"features": []})["features"]
        return {
            "source_offset": checkpoint["source_offset"],
            "max_event_time": checkpoint["max_event_time"],
            "watermark": checkpoint.get("watermark"),
            "bronze_events": self._count(self.paths["bronze"]),
            "silver_events": self._count(self.paths["silver"]),
            "late_events": self._count(self.paths["late"]),
            "gold_rows": len(gold),
            "gold_sha256": self._digest(gold),
        }

    def _rebuild_gold(self):
        features = self._gold_rows()
        self._atomic_json(self.paths["gold"], {
            "schema_version": 1,
            "source": "silver/events",
            "features": features,
            "sha256": self._digest(features),
        })

    def _gold_rows(self):
        totals = defaultdict(float)
        for row in sorted(self._read(self.paths["silver"]), key=lambda r: (r["timestamp"], r["event_id"])):
            item = self.store.item(row["item_id"])
            if not item:
                continue
            weight = ACTION_WEIGHT[row["action"]]
            if row["action"] in ("play", "complete"):
                weight *= max(0.15, row["watch_pct"])
            for genre in item.genres:
                totals[(row["user_id"], genre)] += weight / len(item.genres)
        return [
            {"user_id": user, "genre": genre, "affinity": round(value, 6)}
            for (user, genre), value in sorted(totals.items())
        ]

    def _checkpoint(self):
        return self._read_json(self.paths["checkpoint"], default={
            "source_offset": 0, "max_event_time": None, "watermark": None,
        })

    @staticmethod
    def _append(path, record):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _read(path):
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _existing(self, path, key):
        return {row[key] for row in self._read(path)}

    def _count(self, path):
        return len(self._read(path))

    @staticmethod
    def _read_json(path, default):
        if not path.exists():
            return default
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _atomic_json(path, payload):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))

    @staticmethod
    def _digest(value):
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()
