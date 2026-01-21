# cupid/db.py
from __future__ import annotations
import sqlite3
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class CacheRecord:
    key: str
    best_config_json: str
    best_us: float
    created_at: float


class TuningDB:
    """
    Very small SQLite cache for:
      (kernel_name, device, dtype, shape_bucket, extra_key) -> best_config
    Similar spirit to Triton-DejaVu: save tuned results to disk. :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, path: str = "cupid_tuning.sqlite"):
        self.path = path
        self.conn = sqlite3.connect(self.path)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tuning_cache (
              key TEXT PRIMARY KEY,
              best_config_json TEXT NOT NULL,
              best_us REAL NOT NULL,
              created_at REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[CacheRecord]:
        cur = self.conn.cursor()
        cur.execute("SELECT key, best_config_json, best_us, created_at FROM tuning_cache WHERE key = ?;", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        return CacheRecord(key=row[0], best_config_json=row[1], best_us=float(row[2]), created_at=float(row[3]))

    def put(self, key: str, best_config: Dict[str, Any], best_us: float):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO tuning_cache(key, best_config_json, best_us, created_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              best_config_json=excluded.best_config_json,
              best_us=excluded.best_us,
              created_at=excluded.created_at;
            """,
            (key, json.dumps(best_config), float(best_us), time.time()),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
