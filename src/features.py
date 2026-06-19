import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INTERVAL = "1h"

def add_features(df):
    # Moving Average
    df["ma_7"]  = df["close"].rolling(7).mean()
    df["ma_25"] = df["close"].rolling(25).mean()
    df["ma_99"] = df["close"].rolling(99).mean()

    # RSI
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs        = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_upper"] = df["bb_mid"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["bb_mid"] - 2 * df["close"].rolling(20).std()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    # Target: multi-horizon
    df["target_1h"]  = df["close"].shift(-1)
    df["target_6h"]  = df["close"].shift(-6)
    df["target_24h"] = df["close"].shift(-24)

    # Hapus baris yang ada nilai kosong
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

if __name__ == "__main__":
    for symbol in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{symbol}_{INTERVAL}.csv")
        df   = pd.read_csv(path)
        df   = add_features(df)

        out  = os.path.join(DATA_DIR, f"{symbol}_{INTERVAL}_features.csv")
        df.to_csv(out, index=False)

        print(f"{symbol}: {df.shape} → disimpan ke {out}")
        print(f"  Kolom: {list(df.columns)}\n")
