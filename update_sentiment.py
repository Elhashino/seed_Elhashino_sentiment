#!/usr/bin/env python3
# update_sentiment.py

import os
import csv
import time
import datetime
import logging
import feedparser
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
    "USDCAD", "XAUUSD", "CL", "BTC-USD", "SPY", "AAPL"
]
OUTPUT_CSV = "sentiment_scores.csv"
MAX_ITEMS   = 30  # how many headlines/posts per source

# -------------------------------------------------------------------
# DATA FETCHERS
# -------------------------------------------------------------------

def fetch_google_news(symbol, max_items=MAX_ITEMS):
    """Google News RSS"""
    url = f"https://news.google.com/rss/search?q={symbol}+forex&hl=en-US&gl=US&ceid=US:en"
    d = feedparser.parse(url)
    texts = []
    for e in d.entries[:max_items]:
        summary = getattr(e, "summary", "")
        texts.append(e.title + (" — " + summary if summary else ""))
    return texts

def fetch_reuters_fx(max_items=MAX_ITEMS):
    """Reuters Forex RSS (no symbol filter)"""
    url = "https://www.reutersagency.com/feed/?best-topics=forex"
    d = feedparser.parse(url)
    texts = []
    for e in d.entries[:max_items]:
        summary = getattr(e, "summary", "")
        texts.append(e.title + (" — " + summary if summary else ""))
    return texts

def fetch_bing_news(symbol, max_items=MAX_ITEMS):
    """Bing News RSS"""
    url = f"https://www.bing.com/news/search?q={symbol}+forex&format=rss"
    d = feedparser.parse(url)
    texts = []
    for e in d.entries[:max_items]:
        summary = getattr(e, "description", "")
        texts.append(e.title + (" — " + summary if summary else ""))
    return texts

def fetch_yahoo_finance(symbol, max_items=MAX_ITEMS):
    """Yahoo Finance RSS for symbol"""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US&symbol={symbol}"
    d = feedparser.parse(url)
    texts = []
    for e in d.entries[:max_items]:
        summary = getattr(e, "summary", "")
        texts.append(e.title + (" — " + summary if summary else ""))
    return texts

def fetch_reddit(symbol, max_items=MAX_ITEMS):
    """Pushshift Reddit submissions from last 24h"""
    one_day_ago = int((datetime.datetime.utcnow() - datetime.timedelta(days=1)).timestamp())
    url = "https://api.pushshift.io/reddit/search/submission/"
    params = {"q": symbol, "subreddit": "forex,investing,stocks", "after": one_day_ago, "size": max_items}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get("data", [])
    except Exception as e:
        logging.warning(f"fetch_reddit failed for {symbol}: {e}")
        return []
    texts = []
    for post in data:
        title   = post.get("title", "")
        selftxt = post.get("selftext", "")
        if title:
            texts.append(title + (" — " + selftxt if selftxt else ""))
    return texts

def fetch_investing(symbol, max_items=MAX_ITEMS):
    """Investing.com search results"""
    url = f"https://www.investing.com/search/?q={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select(".searchSection .js-article-item")[:max_items]
        return [it.get_text(separator=" — ", strip=True) for it in items]
    except Exception as e:
        logging.warning(f"fetch_investing failed for {symbol}: {e}")
        return []

# -------------------------------------------------------------------
# SENTIMENT ANALYSIS (FinBERT-Tone)
# -------------------------------------------------------------------

MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

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

    results = nlp(
        cleaned,
        truncation=True,
        padding=True,
        max_length=512,
        batch_size=16
    )

    label2score = {"negative": -1.0, "neutral": 0.0, "positive": +1.0}
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
            logging.info(f"→ Scoring {sym} …")
            texts = []
            texts += fetch_google_news(sym)
            texts += fetch_reuters_fx()          # no sym argument
            texts += fetch_bing_news(sym)
            texts += fetch_yahoo_finance(sym)
            texts += fetch_reddit(sym)
            texts += fetch_investing(sym)

            score, count = compute_sentiment(texts)
            ts = datetime.datetime.utcnow().isoformat()
            writer.writerow([ts, sym, f"{score:.3f}", count])
            logging.info(f"   {sym}: {score:.3f} from {count} texts")
            time.sleep(1.0)

if __name__ == "__main__":
    main()
