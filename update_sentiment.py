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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
    "USDCAD", "XAUUSD", "CL", "BTCUSD", "SPY", "AAPL"
]
OUTPUT_CSV = "sentiment_scores.csv"

# Bump this up to 60 or even 80 if you want more “chances” to catch a non-neutral headline.
MAX_ITEMS = 60

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_google_news(symbol, max_items=MAX_ITEMS):
    url = f"https://news.google.com/rss/search?q={symbol}+forex&hl=en-US&gl=US&ceid=US:en"
    d = feedparser.parse(url)
    texts = []
    for entry in d.entries[:max_items]:
        summary = getattr(entry, "summary", "")
        texts.append(entry.title + (" — " + summary if summary else ""))
    return texts

def fetch_reuters_fx(max_items=MAX_ITEMS):
    # Reuters FX feed
    url = "https://www.reuters.com/tools/rss?feedName=fxNews"
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
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get("data", [])
    except Exception:
        data = []
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
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select(".searchSection .js-article-item")[:max_items]
        return [it.get_text(separator=" — ", strip=True) for it in items]
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ANALYSIS (FinBERT)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

nlp = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

def compute_sentiment(texts):
    # Clean up and dedupe
    cleaned = [t.replace("\n", " ").strip() for t in texts if t.strip()]
    if not cleaned:
        return 0.0, 0

    results = nlp(
        cleaned,
        truncation=True,
        padding=True,
        max_length=512,
        batch_size=16
    )

    # FinBERT’s labels are 'negative' / 'neutral' / 'positive'
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

    avg = sum(scores) / len(scores)

    # Optional: If you want to treat tiny averages as exactly zero, uncomment:
    # if abs(avg) < 0.005:
    #     avg = 0.0

    return avg, len(scores)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    first_run = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if first_run:
            writer.writerow(["timestamp", "symbol", "sentiment_score", "num_texts"])

        for sym in SYMBOLS:
            print(f"→ Scoring {sym} …", flush=True)

            texts = []
            texts += fetch_google_news(sym)
            texts += fetch_reuters_fx()
            texts += fetch_reddit(sym)
            texts += fetch_investing(sym)

            score, count = compute_sentiment(texts)
            ts = datetime.datetime.utcnow().isoformat()
            writer.writerow([ts, sym, score, count])
            print(f"   {sym}: {score:.4f} from {count} texts", flush=True)

            time.sleep(1.0)  # be polite to the servers

if __name__ == "__main__":
    main()
