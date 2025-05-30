# dashboard.py

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ─── 1) Must be the first Streamlit command ──────────────────────────────────
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ─── 2) Load & preprocess ────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        # create an empty DataFrame with the right columns
        return pd.DataFrame(columns=["timestamp","symbol","sentiment_score","num_texts"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return df

CSV_FILE = "sentiment_scores.csv"
df = load_data(CSV_FILE)

st.title("AI Sentiment Dashboard")
st.write("Latest sentiment score per symbol (from your `update_sentiment.py`)")

if df.empty:
    st.warning("No data found in sentiment_scores.csv yet.")
    st.stop()

# ─── 3) Extract latest per symbol ─────────────────────────────────────────────
latest = (
    df
    .sort_values("timestamp")
    .groupby("symbol", as_index=False)
    .last()[["symbol","sentiment_score","num_texts","timestamp"]]
    .sort_values("sentiment_score", ascending=False)
)

# ─── 4) Bar chart ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,4))
colors = ["green" if v>=0 else "red" for v in latest["sentiment_score"]]
ax.bar(latest["symbol"], latest["sentiment_score"], color=colors)
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_title("Most Recent Sentiment per Symbol")
mx = max(abs(latest["sentiment_score"].min()), abs(latest["sentiment_score"].max()))
ax.set_ylim(-mx*1.1, mx*1.1)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

# ─── 5) Show raw table ────────────────────────────────────────────────────────
with st.expander("Show raw latest scores"):
    st.dataframe(
        latest.rename(columns={
            "sentiment_score":"score",
            "num_texts":"count",
            "timestamp":"last_updated"
        }),
        use_container_width=True
    )

# ─── 6) Optional history ──────────────────────────────────────────────────────
if st.checkbox("Show full time-series history"):
    symbol = st.selectbox("Pick a symbol", latest["symbol"].tolist())
    sub = df[df["symbol"] == symbol].sort_values("timestamp")
    fig2, ax2 = plt.subplots(figsize=(10,3))
    ax2.plot(sub["timestamp"], sub["sentiment_score"], marker="o")
    ax2.set_ylabel("Sentiment")
    ax2.set_xlabel("Time")
    ax2.set_title(f"{symbol} Sentiment Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
