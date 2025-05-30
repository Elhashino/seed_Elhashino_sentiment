# test_sentiment.py

from update_sentiment import fetch_all_tweets, compute_sentiment, fetch_reddit_comments

SYMBOLS = ["EURUSD","USDJPY","GBPUSD","AUDUSD","USDCAD",
           "XAUUSD","CL","BTCUSD","SPY","AAPL"]

# 1) Test Twitter fetch + sentiment
print("\n=== Twitter Test ===")
buckets = fetch_all_tweets(SYMBOLS, max_results=50)
for sym, texts in buckets.items():
    print(f"\n--- {sym} ({len(texts)} tweets) ---")
    if not texts:
        print(" • No tweets fetched")
        continue
    for t in texts[:3]:  # show up to first 3 tweets
        print(" •", t.replace("\n", " ")[:100], "…")
    score = compute_sentiment(texts)
    print(f"→ Twitter-only avg for {sym}: {score:.3f}" if score is not None else "→ Sentiment unavailable")

# 2) Test Reddit fetch + sentiment on AAPL
sample = "AAPL"
print("\n=== Reddit Test for AAPL ===")
rd = fetch_reddit_comments(sample, limit=10)
print(f"--- {sample} Reddit ({len(rd)} posts) ---")
if not rd:
    print(" • No Reddit posts fetched")
else:
    for p in rd[:3]:
        print(" •", p.replace("\n", " ")[:100], "…")
    score_rd = compute_sentiment(rd)
    print(f"→ Reddit-only avg for {sample}: {score_rd:.3f}" if score_rd is not None else "→ Sentiment unavailable")
