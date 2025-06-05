# dashboard.py

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1) Must be the first Streamlit command
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    """
    Load the CSV, normalize any "BTC-USD" → "BTCUSD", coerce sentiment_score to numeric,
    and ensure timestamp is in datetime format.
    """
    if not os.path.exists(path):
        # If the CSV doesn’t exist yet, return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score", "num_texts"])
    
    # 2A) Read the CSV
    df = pd.read_csv(path, parse_dates=["timestamp"])
    
    # 2B) Normalize ticker names: rename "BTC-USD" to "BTCUSD"
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].replace({"BTC-USD": "BTCUSD"})
    
    # 2C) Ensure sentiment_score is numeric
    if "sentiment_score" in df.columns:
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    
    return df

CSV_FILE = "sentiment_scores.csv"
df = load_data(CSV_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# 3) PAGE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.title("AI Sentiment Dashboard")
st.write("Latest sentiment score per symbol (from your `update_sentiment.py`)")

if df.empty or df["timestamp"].isna().all():
    st.error("No data found in sentiment_scores.csv yet. Run `update_sentiment.py` first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 4) Pick just the most recent row for each symbol
# ─────────────────────────────────────────────────────────────────────────────
latest = (
    df.sort_values("timestamp")
      .groupby("symbol", as_index=False)
      .last()
      .sort_values("symbol")
)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Build & display the bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["green" if s >= 0 else "red" for s in latest["sentiment_score"]]
ax.bar(latest["symbol"], latest["sentiment_score"], color=colors)
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_title("Current AI-computed Sentiment (per Symbol)")

# Make y-axis symmetric around zero
max_abs = max(abs(latest["sentiment_score"].min()), abs(latest["sentiment_score"].max()))
margin = max_abs * 0.1
ax.set_ylim(-max_abs - margin, max_abs + margin)

ax.axhline(0, color="gray", linewidth=1, linestyle="--")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)

with st.expander("Show raw sentiment data"):
    st.dataframe(
        df.rename(columns={"sentiment_score": "score"}).sort_values(
            ["timestamp", "symbol"], ascending=[False, True]
        ),
        use_container_width=True
    )
