import mlflow.sklearn

SYMBOLS  = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
HORIZONS = ["1h", "6h", "24h"]

for symbol in SYMBOLS:
    for horizon in HORIZONS:
        name  = f"crypto-predictor-{symbol}-{horizon}"
        model = mlflow.sklearn.load_model(f"models:/{name}/1")
        path  = f"models/{symbol}_{horizon}"
        mlflow.sklearn.save_model(model, path)
        print(f"Saved: {path}")
