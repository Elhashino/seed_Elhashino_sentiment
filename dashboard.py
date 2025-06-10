# dashboard.py
# ───────────────────────────────────────────────────────────────────
# A Streamlit app that reads “sentiment_scores.csv” (with columns
#   timestamp,symbol,sentiment_score,num_texts)
# and plots the latest FinBERT sentiment for each symbol as a bar chart,
# with a variance band (±1 σ) on each bar, plus explanatory key and
# a reliability table underneath.
# ───────────────────────────────────────────────────────────────────

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ─── 1) PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ─── 2) LOAD & PREPROCESS ─────────────────────────────────────────
def load_data(path):
    """
    Read 'path' into a DataFrame. If missing, create an empty one
    with the four expected columns.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score", "num_texts"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return df

CSV_FILE = "sentiment_scores.csv"
df = load_data(CSV_FILE)

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
colors = ["green" if m >= 0 else "red" for m in means]
ax.bar(symbols, means, yerr=stds, capsize=5, color=colors)

ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_title("Latest FinBERT Sentiment by Symbol (± 1σ)")

max_abs = max(abs(min(means)), abs(max(means)))
margin  = max_abs * 0.1
ax.set_ylim(-max_abs - margin, max_abs + margin)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

# ─── 6) EXPLANATORY KEY ─────────────────────────────────────────────
st.markdown(
    "**🔎 How to read this chart:**  \n"
    "- **Bars** = average sentiment (P₊ – P₋) for each symbol.  \n"
    "- **Whiskers** = ±1 standard deviation (σ) around the mean, showing sentiment variability."
)

# ─── 7) SENTIMENT RELIABILITY TABLE ─────────────────────────────────
rel = agg.copy()
# compute a simple reliability metric = |mean| / (σ + small ε)
rel["Reliability"] = rel["mean_score"].abs() / (rel["std_score"] + 1e-6)
rel = rel[["symbol", "mean_score", "std_score", "Reliability"]].rename(
    columns={
        "symbol":      "Symbol",
        "mean_score":  "Mean",
        "std_score":   "σ (Std Dev)",
    }
)

st.write("**Sentiment reliability by symbol:**")
st.dataframe(rel, use_container_width=True, width=500)
