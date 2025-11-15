from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
except Exception:
    requests = None  # requests is optional; fallback will be used

logger = logging.getLogger(__name__)


class RealTimeDataCollector:
    BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
    BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"

    def __init__(self, session=None):
        # session allows injection for tests; uses requests if available
        self.session = session or (requests.Session() if requests else None)

    def _query_binance_klines(self, symbol: str, interval: str, limit: int = 500) -> Optional[pd.DataFrame]:
        if self.session is None:
            return None
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            resp = self.session.get(self.BINANCE_KLINES, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # Kline format: [openTime, open, high, low, close, volume, closeTime, ...]
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for c in numeric_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            return df
        except Exception as e:
            logger.warning("Binance klines request failed: %s", e)
            return None

    def _query_binance_ticker(self, symbol: str) -> Optional[dict]:
        if self.session is None:
            return None
        try:
            resp = self.session.get(self.BINANCE_TICKER, params={"symbol": symbol}, timeout=8)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Binance ticker request failed: %s", e)
            return None

    def get_historical_data(self, symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 500) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        If external fetch fails, returns synthetic data with realistic structure.
        """
        # Try Binance public API
        df = self._query_binance_klines(symbol=symbol, interval=interval, limit=limit)
        if df is not None and len(df) >= min(30, limit):
            logger.info("Fetched %d historical rows for %s (%s)", len(df), symbol, interval)
            return df

        # Fallback synthetic generator
        logger.info("Falling back to synthetic historical data for %s", symbol)
        return self._generate_synthetic_history(days=min(limit, 365), interval=interval)

    def get_current_data(self, symbol: str = "BTCUSDT") -> dict:
        """
        Returns a small dict: { price: float, price_change: float (24h %), timestamp: datetime }.
        If API fails returns synthetic sample based on last-known synthetic price.
        """
        ticker = self._query_binance_ticker(symbol)
        if ticker:
            try:
                price = float(ticker.get("lastPrice", ticker.get("lastTradePrice", 0)))
                percent = float(ticker.get("priceChangePercent", 0))
                ts = datetime.utcnow()
                return {"price": price, "price_change": percent, "timestamp": ts}
            except Exception:
                pass

        # fallback synthetic current data (deterministic-ish)
        now = datetime.utcnow()
        df = self._generate_synthetic_history(days=60)
        last_close = float(df["close"].iloc[-1])
        last_prev = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
        pct = (last_close / last_prev - 1) * 100 if last_prev and not math.isnan(last_prev) else 0.0
        return {"price": float(last_close), "price_change": float(pct), "timestamp": now}

    def _generate_synthetic_history(self, days: int = 90, interval: str = "1d", base_price: float = 45000.0) -> pd.DataFrame:
        """
        Simple synthetic generator similar to the trainer's fallback. Produces 'days' rows.
        """
        rng = np.random.default_rng(42)
        freq = "1D" if interval.endswith("d") or interval == "1d" else "1min"
        dates = pd.date_range(end=datetime.utcnow(), periods=days, freq=freq)
        prices = [base_price]
        volumes = [1000.0]
        for i in range(1, len(dates)):
            phase = i % 30
            if phase < 10:
                trend = rng.normal(0.001, 0.02)
            elif phase < 20:
                trend = rng.normal(-0.0008, 0.015)
            else:
                trend = rng.normal(0.0001, 0.01)
            vol_mul = 1 + min(3.0, abs(trend) * 12.0)
            new_price = prices[-1] * (1 + trend)
            new_volume = float(rng.lognormal(7.0, 1.0) * vol_mul)
            prices.append(max(1.0, new_price))
            volumes.append(max(1.0, new_volume))

        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": [p * (1 + abs(rng.normal(0, 0.015))) for p in prices],
            "low": [p * (1 - abs(rng.normal(0, 0.015))) for p in prices],
            "close": prices,
            "volume": volumes
        })
        return df