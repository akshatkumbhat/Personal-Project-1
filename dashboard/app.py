import streamlit as st

st.set_page_config(
    page_title="Anomaly Detection Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Real-Time Anomaly Detection Engine")
st.markdown(
    """
    **Detect anomalies in financial time-series data** using multiple ML methods:
    Statistical (Z-score/EWMA), Isolation Forest, LSTM Autoencoder, and a weighted Ensemble.

    ---

    ### Navigation
    Use the sidebar to select a page:
    - **Real-Time Detection** — Stream data and watch anomalies get flagged live
    - **Model Comparison** — Benchmark all models against known market events
    - **Historical Explorer** — Explore any ticker's historical anomalies

    ---

    ### Architecture
    ```
    Data (Yahoo Finance)
      → Feature Engineering (RSI, Bollinger, MACD, Volume Z-score, ...)
        → Detection Models
          ├── Statistical (Z-score + EWMA)
          ├── Isolation Forest
          └── LSTM Autoencoder
        → Weighted Ensemble → Anomaly Alerts
    ```

    ### Research Papers
    - Liu et al. (2008) — *Isolation Forest*
    - Malhotra et al. (2016) — *LSTM-based Encoder-Decoder for Multi-Sensor Anomaly Detection*
    - [Deep Learning for Time Series Anomaly Detection: A Survey (ACM 2024)](https://dl.acm.org/doi/10.1145/3691338)
    """
)
