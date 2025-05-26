import glob
import pandas as pd
import streamlit as st
import altair as alt

# 1) Find all your CSVs in this folder
csv_paths = glob.glob("*.csv")

# 2) Read the latest sentiment from each
rows = []
for path in csv_paths:
    sym = path.replace(".csv", "")
    df = pd.read_csv(path)
    latest = df["close"].iloc[-1] if not df.empty else None
    rows.append({"symbol": sym, "sentiment": latest})

sent_df = pd.DataFrame(rows).dropna()

# 3) Page header
st.title("📊 AI Sentiment Dashboard")
st.markdown("Hourly sentiment scores for each symbol")

# 4) Bar chart of sentiments
chart = (
    alt.Chart(sent_df)
    .mark_bar()
    .encode(
        x=alt.X("symbol:N", sort=None, title="Symbol"),
        y=alt.Y("sentiment:Q", title="Sentiment"),
        color=alt.condition(
            alt.datum.sentiment >= 0,
            alt.value("green"),
            alt.value("red")
        )
    )
    .properties(width=700, height=400)
)
st.altair_chart(chart, use_container_width=True)

# 5) Show the raw table if needed
with st.expander("Show raw data"):
    st.dataframe(sent_df.set_index("symbol"))
