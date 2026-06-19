import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="",
    layout="wide"
)

API_URL = "http://localhost:8000"
SYMBOLS = {
    "Bitcoin (BTC)":  "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "BNB (BNB)":      "BNBUSDT",
    "Solana (SOL)":   "SOLUSDT",
}

st.title("Crypto Price Predictor")
st.caption("Prediksi harga crypto menggunakan Machine Learning + MLflow")

# Sidebar
st.sidebar.header("Pengaturan")
selected = st.sidebar.selectbox("Pilih Coin", list(SYMBOLS.keys()))
symbol   = SYMBOLS[selected]

if st.sidebar.button("Prediksi", use_container_width=True):
    with st.spinner("Mengambil data dan menghitung prediksi..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json={"symbol": symbol})
            data = resp.json()

            current = data["current_price"]
            preds   = data["predictions"]
            rsi     = data["rsi"]
            trend   = data["trend"]

            # Metric cards
            st.subheader(f"{selected} — {trend}")
            col1, col2, col3, col4 = st.columns(4)
            wib = pytz.timezone("Asia/Jakarta")
            now = datetime.now(wib)

            time_1h  = (now + timedelta(hours=1)).strftime("%d %b, %H:%M WIB")
            time_6h  = (now + timedelta(hours=6)).strftime("%d %b, %H:%M WIB")
            time_24h = (now + timedelta(hours=24)).strftime("%d %b, %H:%M WIB")

            col1.metric("Harga Sekarang", f"${current:,.2f}", now.strftime("%d %b, %H:%M WIB"))
            col2.metric(f"Prediksi 1 Jam\n{time_1h}",  f"${preds['1h']:,.2f}",
                        f"{preds['1h'] - current:+,.2f}")
            col3.metric(f"Prediksi 6 Jam\n{time_6h}",  f"${preds['6h']:,.2f}",
                        f"{preds['6h'] - current:+,.2f}")
            col4.metric(f"Prediksi 24 Jam\n{time_24h}", f"${preds['24h']:,.2f}",
                        f"{preds['24h'] - current:+,.2f}")

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
                            {"range": [0, 30],  "color": "lightgreen"},
                            {"range": [30, 70], "color": "lightyellow"},
                            {"range": [70, 100],"color": "salmon"},
                        ],
                        "threshold": {
                            "line":  {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": rsi
                        }
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

            # Chart harga historis + prediksi
            st.subheader("Harga Historis + Prediksi")
            raw = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": symbol, "interval": "1h", "limit": 48}
            ).json()
            df = pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_volume","trades",
                "taker_buy_base","taker_buy_quote","ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close"]     = df["close"].astype(float)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df["open_time"], y=df["close"],
                name="Harga Historis", line=dict(color="royalblue")
            ))

            last_time = df["open_time"].iloc[-1]
            future_times = [
                last_time + pd.Timedelta(hours=1),
                last_time + pd.Timedelta(hours=6),
                last_time + pd.Timedelta(hours=24),
            ]
            future_prices = [preds["1h"], preds["6h"], preds["24h"]]

            fig2.add_trace(go.Scatter(
                x=[last_time] + future_times,
                y=[current] + future_prices,
                name="Prediksi", line=dict(color="orange", dash="dash"),
                mode="lines+markers"
            ))
            fig2.update_layout(height=400, xaxis_title="Waktu", yaxis_title="Harga (USDT)")
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Pilih coin dan klik **Prediksi** untuk memulai")
