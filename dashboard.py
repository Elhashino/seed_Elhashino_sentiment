# dashboard.py
# ───────────────────────────────────────────────────────────────────
# Fetches “sentiment_scores.csv” live from GitHub, plots mean ±1σ
# sentiment per symbol, then shows an optional “Last updated” stamp,
# explanatory key, and a reliability table—all without error.
# ───────────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ─── 1) PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ─── 2) LIVE CSV FETCH ─────────────────────────────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/"
    "Elhashino/seed_Elhashino_sentiment/"
    "main/sentiment_scores.csv"
)

def load_data():
    """
    Read the remote CSV, coerce timestamp to datetime,
    and sentiment_score to numeric. Returns a DataFrame.
    """
    df = pd.read_csv(CSV_URL)
    # Safely parse timestamp column (if present)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Ensure sentiment_score numeric
    df["sentiment_score"] = pd.to_numeric(df.get("sentiment_score", []), errors="coerce")
    return df

df = load_data()

# ─── 3) OPTIONAL LAST-UPDATED STAMP ────────────────────────────────
if "timestamp" in df.columns and df["timestamp"].notna().any():
    latest_time = df["timestamp"].max()
    st.write(f"**Last updated:** {latest_time:%Y-%m-%d %H:%M:%S} UTC")

# ─── 4) DATA CHECK ──────────────────────────────────────────────────
if df.empty or "sentiment_score" not in df.columns:
    st.error("No valid data—please check that your CSV exists and has the right columns.")
    st.stop()

# ─── 5) AGGREGATE MEAN & STDDEV ────────────────────────────────────
agg = (
    df.groupby("symbol", as_index=False)
      .agg(
          mean_score=("sentiment_score", "mean"),
          std_score =("sentiment_score", "std")
      )
)

symbols = agg["symbol"].tolist()
means   = agg["mean_score"].tolist()
stds    = agg["std_score"].fillna(0).tolist()

# ─── 6) PLOT BAR CHART + ERROR BARS ────────────────────────────────
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
    "- **Whiskers** = ±1 standard deviation (σ) around the mean (sentiment variability)."
)

# ─── 8) RELIABILITY TABLE ──────────────────────────────────────────
rel = agg.copy()
rel["Reliability"] = rel["mean_score"].abs() / (rel["std_score"] + 1e-6)
rel = rel.rename(columns={
    "symbol":      "Symbol",
    "mean_score":  "Mean",
    "std_score":   "σ (Std Dev)",
})

st.write("**Sentiment reliability by symbol:**")
st.dataframe(rel, use_container_width=True, width=500)
