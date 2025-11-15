#!/usr/bin/env python3
"""
model_training.py

Robust training pipeline for a RandomForest model predicting short-term crypto returns.

Usage:
    python model_training.py --symbol BTCUSDT --days 180 --interval 1d --out models/trained_model.pkl
"""
from __future__ import annotations

import argparse
import inspect
import joblib
import logging
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional: load environment variables (API keys etc.)
from dotenv import load_dotenv

load_dotenv()

# Try importing the project's data collector; fall back to None if not available
try:
    from utils.data_collector import RealTimeDataCollector  # type: ignore
except Exception:
    RealTimeDataCollector = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model_training")


class StockPredictorTrainer:
    """Trainer class responsible for data fetch, feature engineering, training and saving."""

    def __init__(self, model=None, random_state: int = 42):
        self.model = model or RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=random_state,
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=2,
        )
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.data_collector = self._get_data_collector()

    def _get_data_collector(self):
        if RealTimeDataCollector is None:
            logger.warning("RealTimeDataCollector not found; using internal synthetic generator.")
            return None
        try:
            return RealTimeDataCollector()
        except Exception as e:
            logger.warning("Failed to initialize RealTimeDataCollector: %s", e)
            return None

    def fetch_real_historical_data(self, symbol: str = "BTCUSDT", days: int = 90, interval: str = "1d") -> pd.DataFrame:
        """
        Attempt to fetch real historical data using whichever signature the collector provides.
        Falls back to synthetic data in case of failure.
        """
        logger.info("Fetching historical data for %s (days=%s, interval=%s)", symbol, days, interval)
        if self.data_collector is None:
            logger.info("No external data collector, generating synthetic data.")
            return self.generate_enhanced_synthetic_data(days=days)

        # Try a few commonly used signatures
        attempts = [
            ("get_historical_data", ["symbol", "interval", "limit"]),
            ("get_historical_data", ["symbol", "timeframe"]),
            ("get_historical_data", ["symbol", "limit"]),
            ("get_historical_data", ["symbol"]),
        ]

        for method_name, params in attempts:
            method = getattr(self.data_collector, method_name, None)
            if not callable(method):
                continue
            try:
                sig = inspect.signature(method)
                kwargs = {}
                if "symbol" in sig.parameters:
                    kwargs["symbol"] = symbol
                if "interval" in sig.parameters:
                    kwargs["interval"] = interval
                elif "timeframe" in sig.parameters:
                    kwargs["timeframe"] = interval if isinstance(interval, str) else "1d"
                if "limit" in sig.parameters:
                    kwargs["limit"] = min(days, 1000)
                df = method(**kwargs)
                if isinstance(df, pd.DataFrame) and len(df) >= 30:
                    logger.info("Fetched %d rows from data collector.", len(df))
                    return df
                logger.warning("Data collector returned insufficient data (len=%s).", None if df is None else len(df))
            except Exception as e:
                logger.debug("Calling %s with %s failed: %s", method_name, params, e)

        logger.warning("All data collector attempts failed; falling back to synthetic data.")
        return self.generate_enhanced_synthetic_data(days=days)

    def generate_enhanced_synthetic_data(self, days: int = 90, base_price: float = 45000.0) -> pd.DataFrame:
        """Generate synthetic time series data with regime structure (bull/bear/sideways)."""
        logger.info("Generating synthetic data for %d days (base_price=%s)", days, base_price)
        rng = np.random.default_rng(self.random_state)
        dates = pd.date_range(end=datetime.now(), periods=days, freq="1D")
        prices = [base_price]
        volumes = [1_000.0]

        for i in range(1, days):
            phase = i % 30
            if phase < 10:
                trend = rng.normal(0.001, 0.02)
            elif phase < 20:
                trend = rng.normal(-0.0008, 0.015)
            else:
                trend = rng.normal(0.0001, 0.01)
            vol_multiplier = 1 + min(3.0, abs(trend) * 12.0)
            new_price = prices[-1] * (1 + trend)
            new_volume = float(rng.lognormal(7.0, 1.0) * vol_multiplier)
            prices.append(max(1.0, new_price))
            volumes.append(max(1.0, new_volume))

        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": [p * (1 + abs(rng.normal(0, 0.015))) for p in prices],
            "low": [p * (1 - abs(rng.normal(0, 0.015))) for p in prices],
            "close": prices,
            "volume": volumes,
        })
        logger.info("Synthetic data generated (%d rows).", len(df))
        return df

    def create_advanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators and the future_return target."""
        logger.info("Creating technical features...")
        df = df.copy()

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["returns"] = df["close"].pct_change()
        df["price_trend"] = df["close"] / df["close"].rolling(window=10, min_periods=1).mean() - 1
        df["momentum"] = df["close"] - df["close"].shift(5)

        for window in [5, 10, 20, 50]:
            df[f"MA_{window}"] = df["close"].rolling(window=window, min_periods=1).mean()
            df[f"MA_ratio_{window}"] = df["close"] / df[f"MA_{window}"] - 1

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss.replace(0, np.nan))
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI"].fillna(50, inplace=True)

        df["BB_middle"] = df["close"].rolling(window=20, min_periods=1).mean()
        bb_std = df["close"].rolling(window=20, min_periods=1).std().fillna(0)
        df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
        df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
        df["BB_position"] = ((df["close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])).replace([np.inf, -np.inf], np.nan).fillna(0.5)

        df["volatility"] = df["returns"].rolling(window=20, min_periods=1).std().fillna(0)
        df["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs()
            )
        )
        df["atr"] = df["true_range"].rolling(window=14, min_periods=1).mean().fillna(0)

        df["volume_ma"] = df["volume"].rolling(window=10, min_periods=1).mean()
        df["volume_trend"] = df["volume"] / df["volume_ma"] - 1
        df["volume_volatility"] = df["volume"].pct_change().rolling(window=10, min_periods=1).std().fillna(0)

        df["price_range"] = (df["high"] - df["low"]) / df["close"]
        df["body_size"] = (df["close"] - df["open"]).abs() / df["close"].replace(0, np.nan)
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
        df["body_size"].fillna(0, inplace=True)
        df["price_position"].fillna(0.5, inplace=True)

        df["future_return_3"] = df["close"].shift(-3) / df["close"] - 1
        df["future_return_5"] = df["close"].shift(-5) / df["close"] - 1
        df["future_return"] = df["future_return_3"]

        df.dropna(subset=["future_return"], inplace=True)
        logger.info("Feature engineering complete; resulting shape: %s", df.shape)
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Select feature columns and return X, y, and the feature list."""
        logger.info("Preparing features for training.")
        feature_columns = [
            "returns", "price_trend", "momentum",
            "MA_ratio_5", "MA_ratio_10", "MA_ratio_20", "MA_ratio_50",
            "RSI", "volatility", "atr",
            "BB_position", "volume_trend", "volume_volatility",
            "price_range", "body_size", "price_position",
        ]
        available = [c for c in feature_columns if c in df.columns]
        if not available:
            raise ValueError("No matching features available in dataframe for training.")

        X = df[available].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)
        y = df["future_return"] * 100.0  # percent points

        logger.info("Selected %d features: %s", len(available), available)
        return X, y, available

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2):
        """Time-aware split training & evaluation."""
        logger.info("Training with %d samples.", len(X))
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info("Train samples: %d, Test samples: %d", len(X_train), len(X_test))
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        direction_accuracy = float(((y_pred > 0) == (y_test > 0)).mean())
        theoretical_profit = float(np.sum(np.where(y_pred > 0, y_test.values, 0.0)))

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "direction_accuracy": float(direction_accuracy),
            "theoretical_profit_pct": float(theoretical_profit),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        }

        logger.info("Training finished. MAE: %.4f; RMSE: %.4f; Direction Acc: %.2f%%", mae, rmse, direction_accuracy * 100)

        try:
            importances = getattr(self.model, "feature_importances_", None)
            if importances is not None:
                fi = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
                logger.info("Top features:\n%s", fi.head(10).to_string(index=False))
                metrics["feature_importances"] = fi
        except Exception as e:
            logger.debug("Could not extract feature importances: %s", e)

        return metrics

    def save_model(self, output_path: str = "models/trained_model.pkl", package_path: str = "models/trained_model_package.pkl", feature_names: list | None = None):
        """Save model and a metadata package to disk."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        logger.info("Saving model to %s", output_path)
        joblib.dump(self.model, output_path)

        logger.info("Saving model package to %s", package_path)
        package = {
            "model": self.model,
            "training_date": datetime.utcnow(),
            "feature_names": feature_names or [],
            "model_type": type(self.model).__name__,
        }
        joblib.dump(package, package_path)
        logger.info("Model files saved.")

    def run_full_training(self, symbol: str = "BTCUSDT", days: int = 90, interval: str = "1d", test_ratio: float = 0.2, output_path: str = "models/trained_model.pkl"):
        df = self.fetch_real_historical_data(symbol=symbol, days=days, interval=interval)
        df_feat = self.create_advanced_technical_features(df)

        if df_feat.empty or len(df_feat) < 30:
            raise RuntimeError("Not enough data after feature engineering to train a model.")

        X, y, features = self.prepare_features(df_feat)
        metrics = self.train_and_evaluate(X, y, test_ratio=test_ratio)

        pkg_path = output_path.replace(".pkl", "_package.pkl")
        self.save_model(output_path, pkg_path, feature_names=features)

        return metrics, features


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train RandomForest model for short-term crypto returns.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair (e.g. BTCUSDT)")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (1d, 1h, etc.)")
    parser.add_argument("--out", type=str, default="models/trained_model.pkl", help="Output model path (.pkl)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Holdout ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = StockPredictorTrainer(random_state=args.seed)
    logger.info("Starting training pipeline: symbol=%s days=%s interval=%s", args.symbol, args.days, args.interval)

    try:
        metrics, features = trainer.run_full_training(
            symbol=args.symbol,
            days=args.days,
            interval=args.interval,
            test_ratio=args.test_ratio,
            output_path=args.out,
        )

        logger.info("Training succeeded. Summary metrics: %s", {k: v for k, v in metrics.items() if k != "feature_importances"})
        logger.info("Trained model saved to %s and %s", args.out, args.out.replace(".pkl", "_package.pkl"))
        return 0

    except Exception as e:
        logger.exception("Training pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())