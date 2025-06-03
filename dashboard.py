# dashboard.py

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ─── 1) Must be the first Streamlit command ────────────────────────────────────
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ─── 2) Load & preprocess ─────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        # create an empty DataFrame with the right columns
        return pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score", "num_texts"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return df

CSV_FILE = "sentiment_scores.csv"
df = load_data(CSV_FILE)

# ─── 3) Page layout ─────────────────────────────────────────────────────────────
st.title("AI Sentiment Dashboard")
st.write("Latest sentiment score per symbol (from your `update_sentiment.py`)")

if df.empty:
    st.error("No data found in sentiment_scores.csv yet. Run `update_sentiment.py` first.")
    st.stop()

# Get the most recent sentiment for each symbol
latest = df.groupby("symbol").last().reset_index()

# ─── 4) Build the bar chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Choose bar color: green if score ≥ 0, red if negative
colors = ["green" if s >= 0 else "red" for s in latest["sentiment_score"]]

ax.bar(latest["symbol"], latest["sentiment_score"], color=colors)
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment")

# Make y-axis symmetric around zero so 0-line is centered
max_abs = max(abs(latest["sentiment_score"].min()), abs(latest["sentiment_score"].max()))
margin = max_abs * 0.1  # 10% headroom
ax.set_ylim(-max_abs - margin, max_abs + margin)

ax.axhline(0, color="gray", linewidth=1, linestyle="--")  # zero reference line

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)

with st.expander("Show raw data"):
    st.dataframe(latest.rename(columns={"sentiment_score": "score"}), use_container_width=True)
