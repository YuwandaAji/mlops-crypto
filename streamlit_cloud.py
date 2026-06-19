import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mlflow.sklearn
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="",
    layout="wide"
)

SYMBOLS = {
    "Bitcoin (BTC)":  "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "BNB (BNB)":      "BNBUSDT",
    "Solana (SOL)":   "SOLUSDT",
}
HORIZONS = ["1h", "6h", "24h"]

@st.cache_resource
def load_models():
    models = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]:
        models[symbol] = {}
        for horizon in HORIZONS:
            path = os.path.join(base_dir, "models", f"{symbol}_{horizon}")
            models[symbol][horizon] = mlflow.sklearn.load_model(path)
    return models

def get_latest_features(symbol: str):
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1h", "limit": 200}
    raw    = requests.get(url, params=params).json()

    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

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
    df.reset_index(drop=True, inplace=True)
    return df

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Crypto Price Predictor")
st.caption("Prediksi harga crypto menggunakan Machine Learning + MLflow")

st.sidebar.header("Pengaturan")
selected = st.sidebar.selectbox("Pilih Coin", list(SYMBOLS.keys()))
symbol   = SYMBOLS[selected]

if st.sidebar.button("Prediksi", use_container_width=True):
    with st.spinner("Mengambil data dan menghitung prediksi..."):
        try:
            MODELS = load_models()
            df     = get_latest_features(symbol)
            latest = df.iloc[-1]

            features = {
                "open": latest["open"], "high": latest["high"],
                "low": latest["low"],   "close": latest["close"],
                "volume": latest["volume"], "ma_7": latest["ma_7"],
                "ma_25": latest["ma_25"],   "ma_99": latest["ma_99"],
                "rsi": latest["rsi"],       "bb_mid": latest["bb_mid"],
                "bb_upper": latest["bb_upper"], "bb_lower": latest["bb_lower"],
                "bb_width": latest["bb_width"]
            }

            X = pd.DataFrame([features])
            predictions = {}
            for horizon in HORIZONS:
                pred = MODELS[symbol][horizon].predict(X)[0]
                predictions[horizon] = round(float(pred), 2)

            current = round(float(latest["close"]), 2)
            rsi     = round(float(latest["rsi"]), 2)
            trend   = "Bullish" if predictions["1h"] > current else "Bearish"

            wib      = pytz.timezone("Asia/Jakarta")
            now      = datetime.now(wib)
            time_1h  = (now + timedelta(hours=1)).strftime("%d %b, %H:%M WIB")
            time_6h  = (now + timedelta(hours=6)).strftime("%d %b, %H:%M WIB")
            time_24h = (now + timedelta(hours=24)).strftime("%d %b, %H:%M WIB")

            # Metric cards
            st.subheader(f"{selected} — {trend}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Harga Sekarang", f"${current:,.2f}",
                        now.strftime("%d %b, %H:%M WIB"))
            col2.metric(f"Prediksi 1 Jam\n{time_1h}", f"${predictions['1h']:,.2f}",
                        f"{predictions['1h'] - current:+,.2f}")
            col3.metric(f"Prediksi 6 Jam\n{time_6h}", f"${predictions['6h']:,.2f}",
                        f"{predictions['6h'] - current:+,.2f}")
            col4.metric(f"Prediksi 24 Jam\n{time_24h}", f"${predictions['24h']:,.2f}",
                        f"{predictions['24h'] - current:+,.2f}")

            # RSI gauge
            st.subheader("Indikator RSI")
            col_rsi, col_info = st.columns([1, 2])
            with col_rsi:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rsi,
                    title={"text": "RSI (14)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "royalblue"},
                        "steps": [
                            {"range": [0, 30],   "color": "lightgreen"},
                            {"range": [30, 70],  "color": "lightyellow"},
                            {"range": [70, 100], "color": "salmon"},
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

            with col_info:
                if rsi < 30:
                    st.success("**Oversold** — Harga kemungkinan akan naik")
                elif rsi > 70:
                    st.error("**Overbought** — Harga kemungkinan akan turun")
                else:
                    st.info("**Neutral** — Tidak ada sinyal kuat")

            # Chart historis + prediksi
            st.subheader("Harga Historis + Prediksi")
            last_time    = df["open_time"].iloc[-1]
            future_times = [
                last_time + pd.Timedelta(hours=1),
                last_time + pd.Timedelta(hours=6),
                last_time + pd.Timedelta(hours=24),
            ]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df["open_time"], y=df["close"],
                name="Harga Historis", line=dict(color="royalblue")
            ))
            fig2.add_trace(go.Scatter(
                x=[last_time] + future_times,
                y=[current, predictions["1h"], predictions["6h"], predictions["24h"]],
                name="Prediksi", line=dict(color="orange", dash="dash"),
                mode="lines+markers"
            ))
            fig2.update_layout(
                height=400,
                xaxis_title="Waktu",
                yaxis_title="Harga (USDT)"
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Pilih coin dan klik **Prediksi** untuk memulai")
