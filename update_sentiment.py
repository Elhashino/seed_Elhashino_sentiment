# update_sentiment.py

import time
import pandas as pd
from transformers import pipeline

# 1) List your 10 symbols here
SYMBOLS = [
    "EURUSD","USDJPY","GBPUSD","AUDUSD","USDCAD",
    "XAUUSD","CL","BTCUSD","SPY","AAPL"
]

# 2) Initialize HF sentiment pipeline (CPU)
nlp = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1               # -1 = CPU
)

# 3) Dummy fetch functions for testing
def fetch_twitter_posts(token, query):
    # Temporary test data: mix of positive & negative
    return [
        "I love EURUSD, it's going to the moon!",
        "EURUSD is terrible right now, I'm so bearish."
    ]

def fetch_reddit_comments(client_id, client_secret, subreddit):
    return []

def fetch_stocktwits_messages(symbol):
    return []

def fetch_all_texts_for(sym):
    texts = []
    texts += fetch_twitter_posts(None, f"{sym} OR ${sym}")
    texts += fetch_reddit_comments(None, None, sym)
    texts += fetch_stocktwits_messages(sym)
    return texts

# 4) Compute sentiment: batch through the HF pipeline, convert to signed floats
def compute_sentiment(texts):
    cleaned = [t for t in texts if len(t) > 20]
    if not cleaned:
        return 0.0
    results = nlp(cleaned, truncation=True, batch_size=16)
    signed = [
        (r["score"] if r["label"] == "POSITIVE" else -r["score"])
        for r in results
    ]
    return sum(signed) / len(signed)

# 5) Main: write one CSV per symbol
def main():
    now_ms = int(time.time() * 1000)
    for sym in SYMBOLS:
        texts = fetch_all_texts_for(sym)
        score = compute_sentiment(texts)
        df = pd.DataFrame([{
            "time":  now_ms,
            "open":  score,
            "high":  score,
            "low":   score,
            "close": score
        }])
        fname = f"{sym}.csv"
        df.to_csv(fname, index=False)
        print(f"Wrote {fname} → {score:.3f}")

if __name__ == "__main__":
    main()
