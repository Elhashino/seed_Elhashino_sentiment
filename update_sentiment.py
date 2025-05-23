# Stub fetch functions so the script runs even without real APIs:
def fetch_twitter_posts(token, query):
    return []

def fetch_reddit_comments(client_id, client_secret, subreddit):
    return []

def fetch_stocktwits_messages(symbol):
    return []

import time
import pandas as pd

# 1) List your 10 symbols here
SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "XAUUSD", "CL",     "BTCUSD", "SPY",    "AAPL"
]

def fetch_all_texts_for(sym):
    # Call stub fetchers directly; no env vars needed
    texts = []
    texts += fetch_twitter_posts(None, f"{sym} OR ${sym}")
    texts += fetch_reddit_comments(None, None, sym)
    texts += fetch_stocktwits_messages(sym)
    return texts

def compute_sentiment(texts):
    # 1) (Optional) filter out very short texts
    cleaned = [t for t in texts if len(t) > 20]

    # 2) Stub scoring: give every cleaned text a score of 0.1
    scores = [0.1 for _ in cleaned]  # temporary non-zero placeholder

    # 3) Return the average (or 0.1 if there were no texts)
    return sum(scores) / len(scores) if scores else 0.1

def main():
    # 1) get current UNIX time in seconds
    now_s = int(time.time())
    # 2) floor down to the start of this hour
    hour_start_s = (now_s // 3600) * 3600
    # 3) convert to milliseconds
    now_ms = hour_start_s * 1000

    for sym in SYMBOLS:
        texts = fetch_all_texts_for(sym)
        score = compute_sentiment(texts)

        # Build a one-row DataFrame with the floored timestamp
        df = pd.DataFrame([{
            "time":  now_ms,
            "open":  score,
            "high":  score,
            "low":   score,
            "close": score
        }])

        # Write out the CSV for TradingView to pick up
        df.to_csv(f"{sym}.csv", index=False)
        print(f"Wrote {sym}.csv → {score:.3f}")

if __name__ == "__main__":
    main()
