# dashboard.py
# ───────────────────────────────────────────────────────────────────
# A Streamlit app that reads “sentiment_scores.csv” (with columns
#   timestamp,symbol,sentiment_score,num_texts)
# and plots the latest FinBERT sentiment for each symbol as a bar chart.
# ───────────────────────────────────────────────────────────────────

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ─── 1) PAGE CONFIG ────────────────────────────────────────────────
# Must be the first Streamlit command in your script
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ─── 2) LOAD & PREPROCESS ─────────────────────────────────────────
@st.cache_data
def load_data(path):
    """
    Read 'path' into a DataFrame. If missing, create an empty one
    with the four expected columns.
    """
    if not os.path.exists(path):
        # Create an empty DataFrame with exactly these four columns
        return pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score", "num_texts"])

    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Ensure 'sentiment_score' is numeric (coerce any stray strings to NaN)
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return df

CSV_FILE = "sentiment_scores.csv"
df = load_data(CSV_FILE)

# ─── 3) PAGE LAYOUT ─────────────────────────────────────────────────
st.title("📊 AI Sentiment Dashboard")
st.write("Latest FinBERT sentiment score per symbol (from your `update_sentiment.py`).")

# If no CSV exists or it's empty, show an error and stop
if not os.path.exists(CSV_FILE):
    st.error("No sentiment_scores.csv found. Please run `update_sentiment.py` first.")
    st.stop()

if df.empty:
    st.error("`sentiment_scores.csv` is empty. Run `update_sentiment.py` again to populate it.")
    st.stop()

# ─── 4) EXTRACT THE MOST RECENT ROW PER SYMBOL ───────────────────────
# Sort all rows by timestamp descending, then pick the first (latest) row for each symbol
latest_rows = (
    df.sort_values("timestamp", ascending=False)
      .groupby("symbol", as_index=False)
      .first()
)

# Now latest_rows has exactly one row per symbol, with columns:
#   ["symbol", "timestamp", "sentiment_score", "num_texts"]

symbols = latest_rows["symbol"].tolist()
scores  = latest_rows["sentiment_score"].astype(float).tolist()

# ─── 5) BUILD & DISPLAY THE BAR CHART ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

# Color bars green if sentiment >= 0, else red
colors = ["green" if s >= 0 else "red" for s in scores]

ax.bar(symbols, scores, color=colors)
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_title("Latest FinBERT Sentiment by Symbol")

# Make y‐axis symmetric around zero, so the 0‐line is centered
max_abs = max(abs(min(scores)), abs(max(scores)))
margin  = max_abs * 0.1  # add 10% headroom
ax.set_ylim(-max_abs - margin, max_abs + margin)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)

# ─── 6) SHOW RAW DATA IF REQUESTED ─────────────────────────────────
with st.expander("Show raw sentiment data"):
    # Display the full DataFrame, sorted by timestamp descending then symbol ascending
    st.dataframe(df.sort_values(["timestamp", "symbol"], ascending=[False, True]),
                 use_container_width=True)
