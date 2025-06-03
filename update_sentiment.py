#!/usr/bin/env python3
# update_sentiment.py

import os
import csv
import time
import datetime
import feedparser
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
    "USDCAD", "XAUUSD", "CL", "BTCUSD", "SPY", "AAPL"
]
OUTPUT_CSV = "sentiment_scores.csv"
MAX_ITEMS = 30

# -------------------------------------------------------------------
# DATA FETCHERS
# -------------------------------------------------------------------

def fetch_google_news(symbol, max_items=MAX_ITEMS):
    url = f"https://news.google.com/rss/search?q={symbol}+forex&hl=en-US&gl=US&ceid=US:en"
    d = feedparser.parse(url)
    texts = []
    for entry in d.entries[:max_items]:
        summary = getattr(entry, "summary", "")
        texts.append(entry.title + (" — " + summary if summary else ""))
    return texts

def fetch_reuters_fx(max_items=MAX_ITEMS):
    url = "https://www.reutersagency.com/feed/?best-topics=forex"
    d = feedparser.parse(url)
    texts = []
    for entry in d.entries[:max_items]:
        summary = getattr(entry, "summary", "")
        texts.append(entry.title + (" — " + summary if summary else ""))
    return texts

def fetch_reddit(symbol, max_items=MAX_ITEMS):
    one_day_ago = int(
        (datetime.datetime.utcnow() - datetime.timedelta(days=1)).timestamp()
    )
    url = "https://api.pushshift.io/reddit/search/submission/"
    params = {
        "q": symbol,
        "subreddit": "forex,investing,stocks",
        "after": one_day_ago,
        "size": max_items
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json().get("data", [])
    texts = []
    for post in data:
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        if title:
            texts.append(title + (" — " + selftext if selftext else ""))
    return texts

def fetch_investing(symbol, max_items=MAX_ITEMS):
    url = f"https://www.investing.com/search/?q={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    items = soup.select(".searchSection .js-article-item")[:max_items]
    return [it.get_text(separator=" — ", strip=True) for it in items]

# -------------------------------------------------------------------
# SENTIMENT ANALYSIS
# -------------------------------------------------------------------

# Load FinBERT tone model
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

nlp = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

def compute_sentiment(texts):
    cleaned = [t.replace("\n", " ").strip() for t in texts if t.strip()]
    if not cleaned:
        return 0.0, 0

    # batch inferences with truncation and padding
    results = nlp(
        cleaned,
        truncation=True,
        padding=True,
        max_length=512,
        batch_size=16
    )

    # FinBERT labels are 'negative' / 'neutral' / 'positive'
    label2score = {
        "negative": -1.0,
        "neutral":   0.0,
        "positive": +1.0
    }

    scores = []
    for r in results:
        lbl = r["label"].lower()
        sc  = r["score"] * label2score.get(lbl, 0.0)
        scores.append(sc)

    return sum(scores) / len(scores), len(scores)

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    first = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if first:
            writer.writerow(["timestamp", "symbol", "sentiment_score", "num_texts"])

        for sym in SYMBOLS:
            print(f"→ Scoring {sym} …")
            texts = []
            texts += fetch_google_news(sym)
            texts += fetch_reuters_fx()
            texts += fetch_reddit(sym)
            texts += fetch_investing(sym)

            score, count = compute_sentiment(texts)
            ts = datetime.datetime.utcnow().isoformat()
            writer.writerow([ts, sym, f"{score:.3f}", count])
            print(f"   {sym}: {score:.3f} from {count} texts")
            time.sleep(1.0)

if __name__ == "__main__":
    main()
