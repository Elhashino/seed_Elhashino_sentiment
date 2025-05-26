# dashboard.py

import os
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ─── load our .env keys (even if we don't use them here) ──────────────────────
load_dotenv()

# ─── PAGE LAYOUT ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sentiment Dashboard",
    layout="wide"
)

st.title("📊 AI Sentiment Dashboard")
st.markdown("Hourly sentiment scores for each symbol")

# ─── LOAD LATEST SCORES ───────────────────────────────────────────────────────
SYMBOLS = [
  "EURUSD","USDJPY","GBPUSD","AUDUSD","USDCAD",
  "XAUUSD","CL","BTCUSD","SPY","AAPL"
]

data = {}
for sym in SYMBOLS:
    fn = f"{sym}.csv"
    if os.path.exists(fn):
        df = pd.read_csv(fn)
        # take the last row's 'close' as the current sentiment
        data[sym] = df["close"].iloc[-1]
    else:
        data[sym] = None

scores = pd.Series(data, name="Sentiment").sort_index()

# ─── PLOT BAR CHART ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
scores.plot(kind="bar", ax=ax)
ax.set_ylim(0, 1)                   # since our stub scores are between 0–1
ax.set_ylabel("Sentiment")
ax.set_xlabel("Symbol")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)

# ─── RAW DATA EXPANDER ────────────────────────────────────────────────────────
with st.expander("Show raw data"):
    st.dataframe(scores.to_frame())
