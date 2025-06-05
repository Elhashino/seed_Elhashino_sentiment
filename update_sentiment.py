#!/usr/bin/env python3
# update_sentiment.py
#
# Pulls news (Google/Reuters/Bing/Yahoo/Investing.com), Reddit, etc., 
# runs FinBERT sentiment on those texts for each symbol, and writes/appends 
# a four-column CSV: "timestamp,symbol,sentiment_score,num_texts".
#
# Usage: “python update_sentiment.py” (it will create or append to sentiment_scores.csv).
# If run as a GitHub Action on a schedule, it will refresh the CSV hourly (or daily) automatically.

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

# The list of symbols you track
SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
    "USDCAD", "XAUUSD", "CL", "BTCUSD", "SPY", "AAPL"
]

# Output CSV path (four columns: timestamp,symbol,sentiment_score,num_texts)
OUTPUT_CSV = "sentiment_scores.csv"

# How many items to fetch per source (e.g. 30 headlines/posts)
MAX_ITEMS = 30

# ──────────────────────────────────────────────────────────────────────────────
# 2) SET UP LOGGING (optional but helpful)
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

# ──────────────────────────────────────────────────────────────────────────────
# 3) DATA FETCHERS
# ──────────────────────────────────────────────────────────────────────────────

def fetch_google_news(symbol, max_items=MAX_ITEMS):
    """Fetch latest Google News headlines for a symbol via RSS."""
    url = f"https://news.google.com/rss/search?q={symbol}+forex&hl=en-US&gl=US&ceid=US:en"
    d = feedparser.parse(url)
    texts = []
    for entry in d.entries[:max_items]:
        summary = getattr(entry, "summary", "")
        texts.append(entry.title + (" — " + summary if summary else ""))
    return texts

def fetch_reuters_fx(max_items=MAX_ITEMS):
    """Fetch top Reuters Forex headlines (no symbol filter)."""
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
    """Fetch Bing News RSS for a given symbol."""
    url = f"https://www.bing.com/news/search?q={symbol}+forex&format=rss"
    d = feedparser.parse(url)
    texts = []
    for entry in d.entries[:max_items]:
        summary = getattr(entry, "description", "")
        texts.append(entry.title + (" — " + summary if summary else ""))
    return texts

def fetch_yahoo_finance(symbol, max_items=MAX_ITEMS):
    """Fetch Yahoo Finance RSS for a given symbol."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US&symbol={symbol}"
    d = feedparser.parse(url)
    texts = []
    for entry in d.entries[:max_items]:
        summary = getattr(entry, "summary", "")
        texts.append(entry.title + (" — " + summary if summary else ""))
    return texts

def fetch_reddit(symbol, max_items=MAX_ITEMS):
    """Fetch recent Reddit submissions from r/forex, r/investing, r/stocks mentioning symbol."""
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
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        if title:
            texts.append(title + (" — " + selftext if selftext else ""))
    return texts

def fetch_investing(symbol, max_items=MAX_ITEMS):
    """Fetch Investing.com search results (article headlines) for a given symbol."""
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

# ──────────────────────────────────────────────────────────────────────────────
# 4) SENTIMENT ANALYSIS (FinBERT)
# ──────────────────────────────────────────────────────────────────────────────

# Load FinBERT tone model
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
nlp = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    truncation=True,
    padding=True,
    return_all_scores=False,
    device=-1          # -1 for CPU; change to 0 if you have a GPU
)

def compute_sentiment(texts):
    """
    Given a list of texts, run FinBERT and return (average_score, number_of_texts_used).
    FinBERT returns 'negative', 'neutral', or 'positive' with a confidence score.
    We convert labels to signed numeric: negative → -1.0 * score, neutral → 0, positive → +1.0 * score.
    """
    cleaned = [t for t in texts if len(t) > 20]  # drop very short texts
    if not cleaned:
        return 0.0, 0

    results = nlp(
        cleaned,
        truncation=True,
        padding=True,
        batch_size=16
    )

    label2score = {"negative": -1.0, "neutral": 0.0, "positive": +1.0}
    scores = []
    for r in results:
        lbl = r["label"].lower()  # 'negative' / 'neutral' / 'positive'
        sc  = r["score"] * label2score.get(lbl, 0.0)
        scores.append(sc)

    return sum(scores) / len(scores), len(scores)

# ──────────────────────────────────────────────────────────────────────────────
# 5) BUILD NEW SENTIMENT DATAFRAME
# ──────────────────────────────────────────────────────────────────────────────

def build_new_sentiment_dataframe() -> pd.DataFrame:
    """
    Fetch and compute sentiment for each symbol in SYMBOLS,
    then return a DataFrame `new_df` with exactly these columns:
        ["ticker", "date", "sentiment", "num_texts"].

    We will convert `new_df` into [timestamp, symbol, sentiment_score, num_texts] later.
    """
    rows = []
    for sym in SYMBOLS:
        logging.info(f"Fetching texts for {sym}…")
        # 1) Gather all texts from each data source
        texts = []
        texts += fetch_google_news(sym)
        texts += fetch_reuters_fx()          # no symbol filter needed
        texts += fetch_bing_news(sym)
        texts += fetch_yahoo_finance(sym)
        texts += fetch_reddit(sym)
        texts += fetch_investing(sym)

        # 2) Compute sentiment (average score, count of texts used)
        score, count = compute_sentiment(texts)
        now_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        rows.append({
            "ticker": sym,
            "date": now_date,
            "sentiment": score,
            "num_texts": count
        })

        logging.info(f"   {sym}: sentiment={score:.3f} from {count} texts")

        # Be polite with APIs—wait a second before next symbol
        time.sleep(1.0)

    # Build a DataFrame out of the collected rows
    new_df = pd.DataFrame(rows)

    # Ensure date is formatted as "YYYY-MM-DD"
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
    return new_df

# ──────────────────────────────────────────────────────────────────────────────
# 6) REFRESH (APPEND/WRITE) THE FOUR-COLUMN CSV
# ──────────────────────────────────────────────────────────────────────────────

def refresh_sentiment_csv(csv_path: str = OUTPUT_CSV) -> None:
    """
    1) Call build_new_sentiment_dataframe() to get new_df with columns:
         ["ticker", "date", "sentiment", "num_texts"].

    2) Convert new_df into a DataFrame `new_rows` with exactly these four columns:
         ["timestamp", "symbol", "sentiment_score", "num_texts"].

    3) Read the existing CSV (if it exists) to a DataFrame old_df (with those four columns).
       If the file does not exist, start with an empty DataFrame.

    4) Append new_rows to old_df, drop exact duplicates by (symbol, timestamp) if desired,
       and save back to `csv_path` (overwriting or creating it).

    The final CSV will always have exactly these four columns, in this order:
       timestamp, symbol, sentiment_score, num_texts
    """
    # 2A) Fetch/build the raw new DataFrame (ticker, date, sentiment, num_texts)
    try:
        raw_new = build_new_sentiment_dataframe()
    except Exception as e:
        logging.error(f"Failed to build new sentiment DataFrame: {e}")
        raise

    # 2B) Convert raw_new → four-column DataFrame [timestamp, symbol, sentiment_score, num_texts]
    now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    new_rows = pd.DataFrame({
        "timestamp":       [now_iso] * len(raw_new),
        "symbol":          raw_new["ticker"].astype(str),
        "sentiment_score": raw_new["sentiment"].astype(float),
        "num_texts":       raw_new["num_texts"].astype(int),
    })

    # 2C) Load the existing CSV if it exists, else create an empty 4-column DF
    if os.path.isfile(csv_path):
        try:
            old_df = pd.read_csv(csv_path, dtype={
                "timestamp": str,
                "symbol": str,
                "sentiment_score": float,
                "num_texts": float
            })
        except Exception as e:
            logging.error(f"Could not read existing CSV '{csv_path}': {e}")
            raise
    else:
        old_df = pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score", "num_texts"])

    # 2D) Concatenate old + new
    combined = pd.concat([old_df, new_rows], ignore_index=True)

    # 2E) Drop exact duplicates by (symbol, timestamp) so each symbol/time pair is unique
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last")

    # 2F) Save the four-column CSV back to disk
    try:
        combined.to_csv(csv_path, index=False)
        logging.info(f"Appended {len(new_rows)} new row(s) → '{csv_path}'")
    except Exception as e:
        logging.error(f"Failed to write updated CSV: {e}")
        raise

# ──────────────────────────────────────────────────────────────────────────────
# 7) MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    refresh_sentiment_csv(csv_path=OUTPUT_CSV)
