import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INTERVAL = "1h"

FEATURES = ["open", "high", "low", "close", "volume",
            "ma_7", "ma_25", "ma_99", "rsi",
            "bb_mid", "bb_upper", "bb_lower", "bb_width"]

HORIZONS = {
    "1h":  "target_1h",
    "6h":  "target_6h",
    "24h": "target_24h",
}

MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost":          XGBRegressor(n_estimators=100, random_state=42),
}

def evaluate(y_true, y_pred):
    return {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2":   r2_score(y_true, y_pred),
    }

def train_symbol(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}_{INTERVAL}_features.csv")
    df   = pd.read_csv(path)

    print(f"\n{'='*50}")
    print(f"Training untuk: {symbol}")

    for horizon, target_col in HORIZONS.items():
        X = df[FEATURES]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        print(f"\n  Horizon: {horizon} | Train: {len(X_train)} | Test: {len(X_test)}")

        mlflow.set_experiment(f"mlops-crypto-{symbol}-{horizon}")

        for name, model in MODELS.items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                y_pred  = model.predict(X_test)
                metrics = evaluate(y_test, y_pred)

                mlflow.log_param("model", name)
                mlflow.log_param("symbol", symbol)
                mlflow.log_param("horizon", horizon)
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("test_size", len(X_test))
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path="model")

                print(f"    {name:20s} → MAE: {metrics['mae']:,.2f} | R²: {metrics['r2']:.4f}")

if __name__ == "__main__":
    for symbol in SYMBOLS:
        train_symbol(symbol)
    print("\nSelesai!")
