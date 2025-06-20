#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

# refresh every 60 000 ms = 1 min
st_autorefresh(interval=60_000, limit=None, key="datarefresh")

st.set_page_config(page_title="Hourly AI Sentiment Dashboard", layout="wide")

# load and floor to the hour
df = pd.read_csv("sentiment.csv", parse_dates=["timestamp"])
df["timestamp"] = df["timestamp"].dt.floor("H")

latest = df["timestamp"].max()
df_l = df[df["timestamp"] == latest]

# group per symbol
grouped = df_l.groupby("symbol")["score"].agg(["mean", "std"]).reset_index()

st.title(f"Sentiment for {latest.strftime('%Y-%m-%d %H:%M')} UTC (± 1 σ)")

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["green" if m >= 0 else "red" for m in grouped["mean"]]
ax.bar(grouped["symbol"], grouped["mean"], yerr=grouped["std"], color=colors, capsize=5)
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
st.pyplot(fig)

st.markdown("""
**Legend:**
- **Bar** = mean hourly sentiment (P₊ − P₋)  
- **Whiskers** = ± 1 σ
""")
