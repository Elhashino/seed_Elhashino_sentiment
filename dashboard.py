# dashboard.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# ─── 1) PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(page_title="Hourly AI Sentiment Dashboard", layout="wide")

# ─── 2) AUTO-REFRESH EVERY MINUTE ──────────────────────────────────
# (interval in milliseconds)
st_autorefresh(interval=60 * 1000, key="auto_refresh")

# ─── 3) DATA LOADING (no cache!) ───────────────────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/"
    "Elhashino/seed_Elhashino_sentiment/"
    "main/sentiment_scores.csv"
)

def load_data():
    df = pd.read_csv(CSV_URL)
    # parse timestamps & scores
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["sentiment_score"] = pd.to_numeric(df.get("sentiment_score", []), errors="coerce")
    return df.dropna(subset=["timestamp", "sentiment_score"])

df = load_data()

# ─── 4) DATA CHECK ──────────────────────────────────────────────────
if df.empty or "symbol" not in df.columns:
    st.error("🚨 No data available or missing `symbol`/`timestamp` columns.")
    st.stop()

# ─── 5) AGGREGATE HOURLY ────────────────────────────────────────────
df["hour"] = df["timestamp"].dt.floor("H")
hourly = (
    df
    .groupby(["symbol", "hour"], as_index=False)
    .agg(
        mean_score=("sentiment_score", "mean"),
        std_score =("sentiment_score", "std")
    )
)

# ─── 6) LATEST HOUR VIEW ───────────────────────────────────────────
latest_hour = hourly["hour"].max()
latest = hourly[hourly["hour"] == latest_hour].copy()
latest = latest.sort_values("mean_score", ascending=False)

st.markdown(f"## Sentiment for **{latest_hour:%Y-%m-%d %H:00} UTC** (± 1 σ)")

# Bar chart for the latest hour
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["green" if m >= 0 else "red" for m in latest["mean_score"]]
ax.bar(latest["symbol"], latest["mean_score"], yerr=latest["std_score"].fillna(0),
       capsize=4, color=colors)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_ylim(
    -max(abs(latest["mean_score"].max()), abs(latest["mean_score"].min())) * 1.1,
     max(abs(latest["mean_score"].max()), abs(latest["mean_score"].min())) * 1.1
)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

st.markdown(
    "**Legend:**  \n"
    "- **Bar** = mean hourly sentiment (P₊ – P₋)  \n"
    "- **Whiskers** = ±1 standard deviation (σ)  \n"
)

# ─── 7) OPTIONAL 24 H TREND ────────────────────────────────────────
if st.checkbox("Show last 24 hours trend by symbol"):
    last_24 = hourly[hourly["hour"] >= (latest_hour - pd.Timedelta(hours=23))]
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for sym in last_24["symbol"].unique():
        series = last_24[last_24["symbol"] == sym]
        ax2.plot(series["hour"], series["mean_score"], label=sym)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Hour (UTC)")
    ax2.set_ylabel("Mean Sentiment")
    ax2.set_title("24 Hour Sentiment Trend")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)
    st.legend(loc="upper right")

# ─── 8) RELIABILITY TABLE ──────────────────────────────────────────
rel = latest.copy()
rel["Reliability"] = rel["mean_score"].abs() / (rel["std_score"] + 1e-6)
rel = rel.rename(columns={
    "symbol":      "Symbol",
    "mean_score":  "Mean Score",
    "std_score":   "Std Dev",
})
st.markdown("### Reliability of Latest Hourly Estimates")
st.dataframe(rel[["Symbol", "Mean Score", "Std Dev", "Reliability"]], use_container_width=True)
