#!/usr/bin/env python3
"""
deployment_model_training.py

Deployment-ready training pipeline for a RandomForest model predicting short-term crypto returns.
This version does not require RealTimeDataCollector and always falls back to synthetic data.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging

# Optional: load environment variables (API keys etc.)
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deployment_model_training")


class StockPredictorTrainer:
    """Trainer class for synthetic-data-based RandomForest model."""

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

    def generate_synthetic_data(self, days: int = 180, base_price: float = 45000.0) -> pd.DataFrame:
        """Generate synthetic crypto data for deployment."""
        logger.info("Generating synthetic data for %d days.", days)
        rng = np.random.default_rng(self.random_state)
        dates = pd.date_range(end=datetime.now(), periods=days, freq="1D")
        prices = [base_price]
        volumes = [1000.0]

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

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for model."""
        df = df.copy()
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["returns"] = df["close"].pct_change()
        df["price_trend"] = df["close"] / df["close"].rolling(10, min_periods=1).mean() - 1
        df["momentum"] = df["close"] - df["close"].shift(5)

        for window in [5, 10, 20, 50]:
            df[f"MA_{window}"] = df["close"].rolling(window, min_periods=1).mean()
            df[f"MA_ratio_{window}"] = df["close"] / df[f"MA_{window}"] - 1

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI"].fillna(50, inplace=True)

        df["future_return_3"] = df["close"].shift(-3) / df["close"] - 1
        df["future_return"] = df["future_return_3"]
        df.dropna(subset=["future_return"], inplace=True)

        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        features = [
            "returns", "price_trend", "momentum",
            "MA_ratio_5", "MA_ratio_10", "MA_ratio_20", "MA_ratio_50",
            "RSI"
        ]
        available = [c for c in features if c in df.columns]
        X = df[available].fillna(0)
        y = df["future_return"] * 100.0
        return X, y, available

    def train_model(self, X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2):
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        direction_acc = ((y_pred > 0) == (y_test > 0)).mean()

        metrics = {"mae": mae, "rmse": rmse, "direction_acc": direction_acc}
        return metrics

    def save_model(self, output_path: str = "models/trained_model.pkl", features: list | None = None):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        joblib.dump(self.model, output_path)
        package = {
            "model": self.model,
            "training_date": datetime.utcnow(),
            "features": features or [],
            "model_type": type(self.model).__name__,
        }
        pkg_path = output_path.replace(".pkl", "_package.pkl")
        joblib.dump(package, pkg_path)
        logger.info("Model saved to %s and %s", output_path, pkg_path)

    def run(self, days: int = 180, test_ratio: float = 0.2, output_path: str = "models/trained_model.pkl"):
        df = self.generate_synthetic_data(days=days)
        df_feat = self.create_features(df)
        X, y, features = self.prepare_data(df_feat)
        metrics = self.train_model(X, y, test_ratio=test_ratio)
        self.save_model(output_path, features)
        logger.info("Training completed. Metrics: %s", metrics)
        return metrics, features


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Deployable RandomForest crypto trainer")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--out", type=str, default="models/trained_model.pkl")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = StockPredictorTrainer(random_state=args.seed)
    metrics, features = trainer.run(days=args.days, test_ratio=args.test_ratio, output_path=args.out)
    logger.info("Deployment-ready training finished successfully.")


if __name__ == "__main__":
    main()
