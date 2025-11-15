"""
PredictionLogger

- Lightweight logger that appends prediction records to a CSV file.
- Thread-safe-ish (simple append mode). For production use, switch to a DB (Postgres) or queue.
- Methods:
    - log(symbol, timestamp, features, prediction, sentiment, extra)
    - read_recent(n=20) -> list of dict rows
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_LOG_PATH = "logs/predictions.csv"


class PredictionLogger:
    def __init__(self, filepath: str = DEFAULT_LOG_PATH):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        # Ensure header exists
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=[
                    "timestamp", "symbol", "prediction", "sentiment", "features", "extra"
                ])
                writer.writeheader()

    def log(self, symbol: str, timestamp: Optional[datetime], features: Optional[Dict[str, Any]], prediction: float, sentiment: float, extra: Optional[Dict[str, Any]] = None):
        ts = (timestamp or datetime.utcnow()).isoformat()
        row = {
            "timestamp": ts,
            "symbol": symbol,
            "prediction": float(prediction) if prediction is not None else "",
            "sentiment": float(sentiment) if sentiment is not None else "",
            "features": str(features) if features is not None else "",
            "extra": str(extra) if extra is not None else ""
        }
        with open(self.filepath, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["timestamp", "symbol", "prediction", "sentiment", "features", "extra"])
            writer.writerow(row)

    def read_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        rows = []
        try:
            with open(self.filepath, "r", newline="", encoding="utf-8") as fh:
                reader = list(csv.DictReader(fh))
                for row in reader[-n:]:
                    rows.append(row)
        except FileNotFoundError:
            return []
        return rows