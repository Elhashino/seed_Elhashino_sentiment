# update_sentiment.py

import os
import time
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline
import tweepy
from tweepy.errors import TooManyRequests, BadRequest

# ─── 1) Load API keys ─────────────────────────────────────────────────────────
load_dotenv()
TW_BEARER      = os.getenv("TWITTER_BEARER_TOKEN", "")
REDDIT_ID      = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET  = os.getenv("REDDIT_CLIENT_SECRET", "")
STOCKTWITS_KEY = os.getenv("STOCKTWITS_API_KEY", "")

# ─── 2) Initialize clients ────────────────────────────────────────────────────
client = tweepy.Client(bearer_token=TW_BEARER)
# (We'll stub Reddit + StockTwits until you add credentials)
nlp    = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

# ─── 3) Your symbols ──────────────────────────────────────────────────────────
SYMBOLS = [
    "EURUSD","USDJPY","GBPUSD","AUDUSD","USDCAD",
    "XAUUSD","CL","BTCUSD","SPY","AAPL"
]

def get_last_score(symbol):
    """Read last 'close' from SYMBOL.csv or return neutral 0.5."""
    fn = f"{symbol}.csv"
    if not os.path.exists(fn):
        return 0.5
    df = pd.read_csv(fn)
    return df["close"].iloc[-1]

def fetch_all_tweets(symbols, max_results=100):
    """
    Single call: search for ANY of the symbols in one query,
    then bucket the returned Tweets by symbol.
    """
    query = "(" + " OR ".join(symbols) + ") lang:en -is:retweet -is:reply"
    try:
        resp = client.search_recent_tweets(query=query, max_results=max_results)
        texts = [t.text for t in resp.data or []]
    except TooManyRequests:
        print("Twitter rate‐limit hit. Skipping new Tweets.")
        texts = []
    except BadRequest as e:
        print(f"Twitter bad request: {e}")
        texts = []

    buckets = {sym: [] for sym in symbols}
    for txt in texts:
        for sym in symbols:
            if sym in txt or f"#{sym}" in txt:
                buckets[sym].append(txt)
    return buckets

def compute_sentiment(texts):
    """
    Run HF sentiment on a list of texts, return average signed score.
    POSITIVE → +score, NEGATIVE → -score.
    """
    cleaned = [t for t in texts if len(t) > 20]
    if not cleaned:
        return None
    results = nlp(cleaned, truncation=True, batch_size=16)
    scores = [
        (r["score"] if r["label"]=="POSITIVE" else -r["score"])
        for r in results
    ]
    return sum(scores)/len(scores)

def main():
    now_ms      = int(time.time() * 1000)
    tweets_buckets = fetch_all_tweets(SYMBOLS, max_results=100)

    for sym in SYMBOLS:
        texts = tweets_buckets.get(sym, [])

        # ─── Optional: add Reddit & StockTwits when ready ─────────
        # texts += fetch_reddit_comments(sym)
        # texts += fetch_stocktwits_messages(sym)

        new_score = compute_sentiment(texts)
        if new_score is None:
            score = get_last_score(sym)
            print(f"No new Tweets for {sym}; using last score {score:.3f}")
        else:
            score = new_score

        # ─── Write out one‐row CSV ────────────────────────────────
        df = pd.DataFrame([{
            "time":  now_ms,
            "open":  score,
            "high":  score,
            "low":   score,
            "close": score
        }])
        filename = f"{sym}.csv"
        df.to_csv(filename, index=False)
        print(f"Wrote {filename} → {score:.3f}")

if __name__ == "__main__":
    main()
