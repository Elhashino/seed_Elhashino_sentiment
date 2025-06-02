# dashboard.py
# ───────────────────────────────────────────────────────────────────
# A Streamlit app that reads “sentiment_scores.csv” and shows
# the most recent sentiment for each symbol as a bar chart.
# ───────────────────────────────────────────────────────────────────

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ─── 1) PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")
st.title("AI Sentiment Dashboard")
st.write("This shows the latest FinBERT sentiment for each symbol.")

# ─── 2) LOAD THE CSV ───────────────────────────────────────────────
if not os.path.exists("sentiment_scores.csv"):
    st.error("No sentiment_scores.csv found. Run update_sentiment first.")
    st.stop()

df = pd.read_csv("sentiment_scores.csv", parse_dates=["timestamp"])

# ─── 3) GET THE LATEST SCORE PER SYMBOL ────────────────────────────
latest_rows = (
    df.sort_values("timestamp", ascending=False)
      .groupby("symbol", as_index=False)
      .first()
)
# latest_rows now has columns: [symbol, timestamp, sentiment_score, num_texts]

# ─── 4) PLOT A BAR CHART ───────────────────────────────────────────
symbols = latest_rows["symbol"]
scores  = latest_rows["sentiment_score"].astype(float)
colors  = ["green" if s >= 0 else "red" for s in scores]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(symbols, scores, color=colors)
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_title("Latest FinBERT Sentiment by Symbol")
max_abs = max(abs(scores.min()), abs(scores.max()))
margin  = max_abs * 0.1
ax.set_ylim(-max_abs - margin, max_abs + margin)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)

# ─── 5) SHOW RAW DATA IF REQUESTED ─────────────────────────────────
with st.expander("Show raw sentiment data"):
    st.dataframe(df, use_container_width=True)
