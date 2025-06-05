# dashboard.py
# ───────────────────────────────────────────────────────────────────
# A Streamlit app that reads “sentiment_scores.csv” and shows
# the most recent FinBERT sentiment for each symbol as a bar chart.
# ───────────────────────────────────────────────────────────────────

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ─── 1) PAGE CONFIG ────────────────────────────────────────────────
# Must be the first Streamlit command
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ─── 2) LOAD & PREPROCESS ─────────────────────────────────────────
@st.cache_data
def load_data(path):
    """
    Loads the CSV at `path`, coerces sentiment_score → numeric,
    and parses timestamp as datetime. If the file doesn’t exist yet,
    returns an empty DataFrame with the correct four columns.
    """
    if not os.path.exists(path):
        # No file yet → create empty DataFrame with expected columns
        return pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score", "num_texts"])

    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Make sure sentiment_score is numeric
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return df

CSV_FILE = "sentiment_scores.csv"
df = load_data(CSV_FILE)

# ─── 3) PAGE LAYOUT ─────────────────────────────────────────────────
st.title("📊 AI Sentiment Dashboard")
st.write("This shows the latest FinBERT sentiment score for each symbol.")
st.write("Latest sentiment score per symbol (from your `update_sentiment.py`).")

if not os.path.exists(CSV_FILE):
    st.error("No sentiment_scores.csv found. Run `update_sentiment.py` first.")
    st.stop()
if df.empty:
    st.error("No data found in sentiment_scores.csv yet. Run `update_sentiment.py` first.")
    st.stop()

# ─── 4) GET THE MOST RECENT SCORE PER SYMBOL ───────────────────────
# Sort by timestamp descending, then pick the first row for each symbol
latest_rows = (
    df.sort_values("timestamp", ascending=False)
      .groupby("symbol", as_index=False)
      .first()
)
# latest_rows has columns: [symbol, timestamp, sentiment_score, num_texts]

symbols = latest_rows["symbol"].tolist()
scores  = latest_rows["sentiment_score"].astype(float).tolist()

# ─── 5) BUILD & DISPLAY THE BAR CHART ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

# Choose bar color: green if score >= 0, red if negative
colors = ["green" if s >= 0 else "red" for s in scores]

ax.bar(symbols, scores, color=colors)
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_title("Latest FinBERT Sentiment by Symbol")

# Make y-axis symmetric around zero so the 0‐line is centered
max_abs = max(abs(min(scores)), abs(max(scores)))
margin  = max_abs * 0.1  # 10% headroom
ax.set_ylim(-max_abs - margin, max_abs + margin)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)

# ─── 6) SHOW RAW DATA IF REQUESTED ─────────────────────────────────
with st.expander("Show raw sentiment data"):
    # Show the full DataFrame (sorted by timestamp descending, symbol ascending)
    st.dataframe(df.sort_values(["timestamp", "symbol"], ascending=[False, True]), use_container_width=True)
