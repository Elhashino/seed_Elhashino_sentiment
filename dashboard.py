import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import matplotlib.pyplot as plt

# —– AUTO REFRESH —–
# rerun every 60 seconds
st_autorefresh(interval=60_000, limit=999_999, key="ticker")

st.set_page_config(page_title="Hourly AI Sentiment Dashboard", layout="wide")
st.title("📊 Hourly AI Sentiment Dashboard")

@st.experimental_memo(ttl=60)
def load_data():
    df = pd.read_csv("sentiment_scores.csv", parse_dates=["timestamp"])
    # floor to the hour
    df["hour"] = df["timestamp"].dt.floor("H")
    return df

df = load_data()

# pick the latest hour
latest_hour = df["hour"].max()
st.markdown(f"**Sentiment for {latest_hour.strftime('%Y-%m-%d %H:%M UTC')} (±1 σ)**")

# subset & aggregate
subset  = df[df["hour"] == latest_hour]
grouped = subset.groupby("symbol")["score"].agg(["mean", "std"]).reset_index()

# draw
fig, ax = plt.subplots(figsize=(10, 5))
cols = ["green" if m >= 0 else "red" for m in grouped["mean"]]
ax.bar(grouped["symbol"], grouped["mean"], yerr=grouped["std"], color=cols)
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_xticklabels(grouped["symbol"], rotation=45, ha="right")
st.pyplot(fig, clear_figure=True)

# reliability table
st.markdown("### Reliability of Latest Hourly Estimates")
display = grouped.rename(columns={"mean": "Mean Score", "std": "Std Dev"})
st.dataframe(display.set_index("symbol"))
