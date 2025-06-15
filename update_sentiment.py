#!/usr/bin/env python3
# update_sentiment.py
#
# Pulls news (Google/Reuters/Bing/Yahoo/Investing.com), Reddit, etc.,
# runs FinBERT sentiment on those texts for each symbol, and writes/appends
# a four-column CSV: "timestamp,symbol,sentiment_score,num_texts".
#
# Usage: “python update_sentiment.py” (it will create or append to sentiment_scores.csv).

import os
import time
import datetime
import logging
import feedparser
import requests
import pandas as pd

from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ──────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
    "USDCAD", "XAUUSD", "CL", "BTCUSD", "SPY", "AAPL"
]

OUTPUT_CSV = "sentiment_scores.csv"
MAX_ITEMS  = 30

# ──────────────────────────────────────────────────────────────────────────────
# 2) SET UP LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

# ──────────────────────────────────────────────────────────────────────────────
# 3) DATA FETCHERS
# ──────────────────────────────────────────────────────────────────────────────

def fetch_google_news(symbol, max_items=MAX_ITEMS):
    url = f"https://news.google.com/rss/search?q={symbol}+forex&hl=en-US&gl=US&ceid=US:en"
    d = feedparser.parse(url)
    texts = []
    for entry in d.entries[:max_items]:
        summary = getattr(entry, "summary", "")
        texts.append(entry.title + (" — " + summary if summary else ""))
    return texts

def fetch_reuters_fx(max_items=MAX_ITEMS):
    try:
        url = "https://www.reutersagency.com/feed/?best-topics=forex"
        d = feedparser.parse(url)
        texts = []
        for entry in d.entries[:max_items]:
            summary = getattr(entry, "summary", "")
            texts.append(entry.title + (" — " + summary if summary else ""))
        return texts
    except Exception as e:
        logging.warning(f"fetch_reuters_fx failed: {e}")
        return []

def fetch_bing_news(symbol, max_items=MAX_ITEMS):
    url = f"https://www.bing.com/news/search?q={symbol}+forex&format=rss"
    d = feedparser.parse(url)
    texts = []
    for entry in d.entries[:max_items]:
        summary = getattr(entry, "description", "")
        texts.append(entry.title + (" — " + summary if summary else ""))
    return texts

def fetch_yahoo_finance(symbol, max_items=MAX_ITEMS):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US&symbol={symbol}"
    d = feedparser.parse(url)
    texts = []
    for entry in d.entries[:max_items]:
        summary = getattr(entry, "summary", "")
        texts.append(entry.title + (" — " + summary if summary else ""))
    return texts

def fetch_reddit(symbol, max_items=MAX_ITEMS):
    one_day_ago = int((datetime.datetime.utcnow() - datetime.timedelta(days=1)).timestamp())
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
    url = f"https://www.investing.com/search/?q={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r    = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select(".searchSection .js-article-item")[:max_items]
        return [it.get_text(separator=" — ", strip=True) for it in items]
    except Exception as e:
        logging.warning(f"fetch_investing failed for {symbol}: {e}")
        return []

# ──────────────────────────────────────────────────────────────────────────────
# 4) SENTIMENT ANALYSIS (FinBERT)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
nlp        = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    truncation=True,
    padding=True,
    return_all_scores=False,
    device=-1  # set to 0 if you have a GPU
)

def compute_sentiment(texts):
    cleaned = [t for t in texts if len(t) > 20]
    if not cleaned:
        return 0.0, 0
    results = nlp(cleaned, truncation=True, padding=True, batch_size=16)
    label2score = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    scores = []
    for r in results:
        lbl = r["label"].lower()
        sc  = r["score"] * label2score.get(lbl, 0.0)
        scores.append(sc)
    return sum(scores) / len(scores), len(scores)

# ──────────────────────────────────────────────────────────────────────────────
# 5) BUILD NEW SENTIMENT DATAFRAME
# ──────────────────────────────────────────────────────────────────────────────

def build_new_sentiment_dataframe() -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        logging.info(f"Fetching texts for {sym}…")
        texts = []
        texts += fetch_google_news(sym)
        texts += fetch_reuters_fx()
        texts += fetch_bing_news(sym)
        texts += fetch_yahoo_finance(sym)
        texts += fetch_reddit(sym)
        texts += fetch_investing(sym)

        score, count = compute_sentiment(texts)
        now_date     = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        rows.append({
            "ticker":   sym,
            "date":     now_date,
            "sentiment": score,
            "num_texts": count
        })
        logging.info(f"   {sym}: sentiment={score:.3f} from {count} texts")
        time.sleep(1.0)

    new_df = pd.DataFrame(rows)
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
    return new_df

# ──────────────────────────────────────────────────────────────────────────────
# 6) REFRESH THE FOUR-COLUMN CSV
# ──────────────────────────────────────────────────────────────────────────────

def refresh_sentiment_csv(csv_path: str = OUTPUT_CSV) -> None:
    raw_new = build_new_sentiment_dataframe()

    now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    new_rows = pd.DataFrame({
        "timestamp":       [now_iso]*len(raw_new),
        "symbol":          raw_new["ticker"].astype(str),
        "sentiment_score": raw_new["sentiment"].astype(float),
        "num_texts":       raw_new["num_texts"].astype(int),
    })

    if os.path.isfile(csv_path):
        old_df = pd.read_csv(csv_path, dtype={
            "timestamp": str,
            "symbol": str,
            "sentiment_score": float,
            "num_texts": float
        })
    else:
        old_df = pd.DataFrame(columns=["timestamp","symbol","sentiment_score","num_texts"])

    combined = pd.concat([old_df, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol","timestamp"], keep="last")
    combined.to_csv(csv_path, index=False)
    logging.info(f"Appended {len(new_rows)} new row(s) → '{csv_path}'")

# ──────────────────────────────────────────────────────────────────────────────
# 7) MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    refresh_sentiment_csv()
