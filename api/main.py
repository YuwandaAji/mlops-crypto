import mlflow.sklearn
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Crypto Price Predictor", version="2.0.0")

SYMBOLS  = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
HORIZONS = ["1h", "6h", "24h"]

# Load semua model saat startup
MODELS = {}
for symbol in SYMBOLS:
    MODELS[symbol] = {}
    for horizon in HORIZONS:
        name = f"crypto-predictor-{symbol}-{horizon}"
        MODELS[symbol][horizon] = mlflow.sklearn.load_model(f"models:/{name}/1")

class PredictRequest(BaseModel):
    symbol: str = "BTCUSDT"

def get_latest_features(symbol: str) -> dict:
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1h", "limit": 120}
    raw    = requests.get(url, params=params).json()

    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)

    df["ma_7"]     = df["close"].rolling(7).mean()
    df["ma_25"]    = df["close"].rolling(25).mean()
    df["ma_99"]    = df["close"].rolling(99).mean()
    delta          = df["close"].diff()
    gain           = delta.clip(lower=0)
    loss           = -delta.clip(upper=0)
    df["rsi"]      = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_upper"] = df["bb_mid"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["bb_mid"] - 2 * df["close"].rolling(20).std()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    df.dropna(inplace=True)
    latest = df.iloc[-1]

    return {
        "open": latest["open"], "high": latest["high"],
        "low": latest["low"],   "close": latest["close"],
        "volume": latest["volume"], "ma_7": latest["ma_7"],
        "ma_25": latest["ma_25"],   "ma_99": latest["ma_99"],
        "rsi": latest["rsi"],       "bb_mid": latest["bb_mid"],
        "bb_upper": latest["bb_upper"], "bb_lower": latest["bb_lower"],
        "bb_width": latest["bb_width"]
    }

@app.get("/")
def root():
    return {"message": "Crypto Price Predictor API", "status": "running"}

@app.post("/predict")
def predict(req: PredictRequest):
    if req.symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Symbol tidak valid. Pilih: {SYMBOLS}")

    features      = get_latest_features(req.symbol)
    current_price = features["close"]
    rsi           = features["rsi"]
    df            = pd.DataFrame([features])

    predictions = {}
    for horizon in HORIZONS:
        pred = MODELS[req.symbol][horizon].predict(df)[0]
        predictions[horizon] = round(float(pred), 2)

    trend = "Bullish" if predictions["1h"] > current_price else "Bearish"

    return {
        "symbol":        req.symbol,
        "current_price": round(current_price, 2),
        "predictions":   predictions,
        "rsi":           round(rsi, 2),
        "trend":         trend,
        "unit":          "USDT"
    }
