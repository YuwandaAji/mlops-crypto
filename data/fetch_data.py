import requests
import pandas as pd
import time
import os

SYMBOLS    = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INTERVAL   = "1h"
LIMIT      = 1000
TOTAL_ROWS = 5000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL   = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol, interval, limit, end_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def fetch_all(symbol, interval, total):
    all_candles = []
    end_time = None
    fetched = 0

    print(f"\nFetching {total} candles untuk {symbol} ({interval})...")

    while fetched < total:
        batch = min(LIMIT, total - fetched)
        raw = fetch_klines(symbol, interval, batch, end_time)
        if not raw:
            break
        all_candles = raw + all_candles
        end_time = raw[0][0] - 1
        fetched += len(raw)
        print(f"  {fetched}/{total} candles...", end="\r")
        time.sleep(0.3)

    print(f"\nSelesai! Total: {len(all_candles)} candles")
    return parse(all_candles)

def parse(raw):
    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_volume","trades",
            "taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for c in ["open","high","low","close","volume","quote_volume"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    df.drop(columns=["ignore"], inplace=True)
    df.drop_duplicates(subset="open_time", inplace=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    for symbol in SYMBOLS:
        df = fetch_all(symbol, INTERVAL, TOTAL_ROWS)
        print(f"Shape : {df.shape}")
        print(f"Range : {df['open_time'].min()} → {df['open_time'].max()}")
        print(df.head(3))
        output = os.path.join(OUTPUT_DIR, f"{symbol}_{INTERVAL}.csv")
        df.to_csv(output, index=False)
        print(f"Disimpan ke: {output}")
