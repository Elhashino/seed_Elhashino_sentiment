# dashboard.py
# ───────────────────────────────────────────────────────────────────
# A Streamlit app that reads “sentiment_scores.csv” (with columns
#   timestamp,symbol,sentiment_score,num_texts)
# and plots the latest FinBERT sentiment for each symbol as a bar chart,
# with a variance band (±1 σ) on each bar.
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
def load_data(path, last_modified):
    """
    Read 'path' into a DataFrame. If missing, create an empty one
    with the four expected columns. Cache keyed on path + file mtime.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score", "num_texts"])

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return df

CSV_FILE = "sentiment_scores.csv"
df = load_data(
    CSV_FILE,
    os.path.getmtime(CSV_FILE) if os.path.exists(CSV_FILE) else None
)

# ─── 3) PAGE LAYOUT ─────────────────────────────────────────────────
st.title("📊 AI Sentiment Dashboard")
st.write("Latest FinBERT sentiment score per symbol (from your `update_sentiment.py`).")

if not os.path.exists(CSV_FILE):
    st.error("No sentiment_scores.csv found. Please run `update_sentiment.py` first.")
    st.stop()

if df.empty:
    st.error("`sentiment_scores.csv` is empty. Run `update_sentiment.py` again to populate it.")
    st.stop()

# ─── 4) AGGREGATE MEAN & STDDEV ────────────────────────────────────
# Compute mean and standard deviation of sentiment_score per symbol
agg = (
    df.groupby("symbol")
      .agg(
          mean_score=("sentiment_score", "mean"),
          std_score =("sentiment_score", "std")
      )
      .reset_index()
)

symbols = agg["symbol"].tolist()
means   = agg["mean_score"].tolist()
stds    = agg["std_score"].fillna(0).tolist()

# ─── 5) BUILD & DISPLAY THE BAR CHART WITH ERROR BARS ─────────────
fig, ax = plt.subplots(figsize=(12, 5))

# Color bars green if mean ≥ 0, else red
colors = ["green" if m >= 0 else "red" for m in means]

# Plot bars at 'means' with ±1σ error bars from 'stds'
ax.bar(symbols, means, yerr=stds, capsize=5, color=colors)

ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_title("Latest FinBERT Sentiment by Symbol (± 1σ)")

# Symmetric y-axis around zero
max_abs = max(abs(min(means)), abs(max(means)))
margin  = max_abs * 0.1
ax.set_ylim(-max_abs - margin, max_abs + margin)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

# ─── 6) SHOW RAW DATA IF REQUESTED ─────────────────────────────────
with st.expander("Show raw sentiment data"):
    st.dataframe(
        df.sort_values(["timestamp", "symbol"], ascending=[False, True]),
        use_container_width=True
    )
