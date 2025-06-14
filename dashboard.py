# dashboard.py
# ───────────────────────────────────────────────────────────────────
# A Streamlit app that fetches “sentiment_scores.csv” directly from
# GitHub on every load, then plots mean ±1 σ sentiment per symbol,
# with an explanatory key, “last updated” stamp, and reliability table.
# ───────────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ─── 1) PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ─── 2) LIVE CSV FETCH (no long cache) ──────────────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/"
    "Elhashino/seed_Elhashino_sentiment/"
    "main/sentiment_scores.csv"
)

def load_data():
    """
    Fetch the remote CSV from GitHub, parse dates,
    and coerce sentiment_score to numeric.
    Always re-runs on each Streamlit reload.
    """
    df = pd.read_csv(CSV_URL, parse_dates=["timestamp"])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return df

df = load_data()

# ─── 3) SHOW LAST-UPDATED TIMESTAMP ────────────────────────────────
latest_time = df["timestamp"].max()
st.write(f"**Last updated:** {latest_time:%Y-%m-%d %H:%M:%S} UTC")

# ─── 4) DATA CHECK ──────────────────────────────────────────────────
if df.empty:
    st.error("No data found at the GitHub CSV URL. Please check your updater.")
    st.stop()

# ─── 5) AGGREGATE MEAN & STDDEV ────────────────────────────────────
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

# ─── 6) BUILD & DISPLAY THE BAR CHART WITH ERROR BARS ─────────────
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

# ─── 7) EXPLANATORY KEY ─────────────────────────────────────────────
st.markdown(
    "**🔎 How to read this chart:**  \n"
    "- **Bars** = average sentiment (P₊ – P₋) for each symbol.  \n"
    "- **Whiskers** = ±1 standard deviation (σ) around the mean, showing sentiment variability."
)

# ─── 8) SENTIMENT RELIABILITY TABLE ─────────────────────────────────
rel = agg.copy()
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
