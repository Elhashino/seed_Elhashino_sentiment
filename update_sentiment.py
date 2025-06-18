import pandas as pd
from datetime import datetime, timedelta
from sentiment_utils import get_finbert_sentiment  # your existing helper
import os

# 10 symbols we track
SYMBOLS = ["AAPL","AUDUSD","BTCUSD","CL","EURUSD","GBPUSD","SPY","USDCAD","USDJPY","XAUUSD"]
OUT_CSV = "sentiment_hourly.csv"

def load_existing():
    if os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV, parse_dates=["timestamp"])
    else:
        df = pd.DataFrame(columns=["timestamp","symbol","sentiment_score"])
    return df

def main():
    df = load_existing()
    # find the last‐written hour (or 2 hrs ago if new)
    if not df.empty:
        last_ts = df["timestamp"].max()
    else:
        last_ts = datetime.utcnow() - timedelta(hours=2)
    # round up to the next hour boundary
    next_hour = (pd.to_datetime(last_ts) + timedelta(hours=1)).replace(
        minute=0, second=0, microsecond=0
    )
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    # how many hours to catch up?
    n_hours = int((now - next_hour) / timedelta(hours=1)) + 1
    for i in range(max(n_hours, 0)):
        ts = next_hour + timedelta(hours=i)
        for sym in SYMBOLS:
            score = get_finbert_sentiment(sym, ts)
            df = pd.concat(
                [df, pd.DataFrame([{"timestamp": ts, "symbol": sym, "sentiment_score": score}])],
                ignore_index=True,
            )
    df.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
