# Crypto Price Predictor — MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.11.3-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.137.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red)

End-to-end MLOps pipeline untuk prediksi harga cryptocurrency menggunakan Machine Learning, MLflow, FastAPI, dan Streamlit.

## Live Demo
**[crypto-price-predictor-yuwanda.streamlit.app](https://crypto-price-predictor-yuwanda.streamlit.app/)**

## Fitur
- Prediksi harga crypto untuk **1 jam, 6 jam, dan 24 jam** ke depan
- Support 4 coin: **BTC, ETH, BNB, SOL**
- Data real-time dari Binance API
- Indikator teknikal: RSI, Moving Average, Bollinger Bands
- Experiment tracking dengan MLflow (36 eksperimen)
- REST API dengan FastAPI
- Dashboard interaktif dengan Streamlit

## Arsitektur
Binance API (Real-time)
↓
Feature Engineering (RSI, MA, Bollinger Bands)
↓
MLflow Experiment Tracking
(3 model × 3 horizon × 4 coin = 36 eksperimen)
↓
MLflow Model Registry (best model per coin per horizon)
↓
FastAPI /predict endpoint
↓
Streamlit Dashboard

## Tech Stack
| Komponen | Teknologi |
|----------|-----------|
| Data Source | Binance REST API |
| Feature Engineering | Pandas, NumPy, TA |
| ML Models | Linear Regression, XGBoost, Random Forest |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI |
| Dashboard | Streamlit + Plotly |

## Model Performance (BTCUSDT)
| Horizon | Model | R² | MAE |
|---------|-------|----|-----|
| 1 jam | Linear Regression | 0.9981 | $208 |
| 6 jam | Linear Regression | 0.9906 | $488 |
| 24 jam | Linear Regression | 0.9531 | $1,114 |

## Cara Menjalankan Lokal

### 1. Clone repo
```bash
git clone https://github.com/YuwandaAji/mlops-crypto.git
cd mlops-crypto
```

### 2. Setup virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Fetch data & training
```bash
python data/fetch_data.py
python src/features.py
python src/train.py
python src/register_model.py
```

### 4. Jalankan FastAPI
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Jalankan Streamlit
```bash
# Versi lokal (butuh FastAPI running)
streamlit run streamlit_app.py

# Versi standalone (tanpa FastAPI)
streamlit run streamlit_cloud.py
```

### 6. Akses
- Streamlit : http://localhost:8501
- FastAPI docs : http://localhost:8000/docs
- MLflow UI : http://localhost:5001

## Future Improvements
- Containerization dengan Docker Compose
- Deploy FastAPI ke cloud (Railway/Render)
- Tambah model LSTM untuk prediksi jangka panjang
- Alert notifikasi ketika RSI oversold/overbought
