import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import matplotlib.pyplot as plt

# auto‐refresh every hour on the hour (3600 sec × 1000 ms)
st_autorefresh(interval=3600 * 1000, key="auto_refresh")

st.set_page_config(page_title="Hourly AI Sentiment Dashboard", layout="wide")
st.title("Hourly AI Sentiment Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("sentiment_hourly.csv", parse_dates=["timestamp"])
    return df

df = load_data()

# pick the latest full‐hour timestamp
latest = df["timestamp"].max()
df_latest = df[df["timestamp"] == latest]

# compute per‐symbol mean & std
stats = df_latest.groupby("symbol")["sentiment_score"].agg(["mean","std","count"]).reset_index()
stats = stats.sort_values("symbol")  # alphabetical

# bar chart
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(
    stats["symbol"],
    stats["mean"],
    yerr=stats["std"],
    capsize=5,
    color="green",
)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Symbol")
ax.set_ylabel("Sentiment Score")
ax.set_title(f"Sentiment for {latest.strftime('%Y-%m-%d %H:%M UTC')} (±1 σ)")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# legend & optional 24-hour trend
st.markdown("""
**Legend**

- **Bar** = mean hourly sentiment (P₊ − P₋)  
- **Whiskers** = ±1 standard deviation (σ)
""")

if st.checkbox("Show last 24 hours trend by symbol"):
    trend = df[df["timestamp"] >= (latest - pd.Timedelta(hours=23))]
    pivot = trend.pivot(index="timestamp", columns="symbol", values="sentiment_score")
    st.line_chart(pivot)

# reliability table
st.subheader("Reliability of Latest Hourly Estimates")
table = stats.rename(columns={
    "mean": "Mean Score",
    "std": "Std Dev",
    "count": "Samples"
})
st.dataframe(table, use_container_width=True)
