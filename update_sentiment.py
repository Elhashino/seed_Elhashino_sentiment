#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import feedparser
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 10 symbols → set your actual RSS/HTML feed URLs here
SYMBOL_FEEDS = {
    "AAPL":   "https://apple.news/apple-events.rss",
    "AUDUSD": "https://www.fxstreet.com/rss/news",
    "BTCUSD": "https://news.bitcoin.com/feed/",
    "CL":     "https://www.investing.com/rss/news_902.rss",
    "EURUSD": "https://www.fxstreet.com/rss/news",
    "GBPUSD": "https://www.fxstreet.com/rss/news",
    "SPY":    "https://www.marketwatch.com/rss/marketpulse",
    "USDCAD": "https://www.fxstreet.com/rss/news",
    "USDJPY": "https://www.fxstreet.com/rss/news",
    "XAUUSD": "https://goldnews.com/feed"
}

def fetch_texts(feed_url, max_items=10):
    d = feedparser.parse(feed_url)
    texts = []
    for entry in d.entries[:max_items]:
        snippet = entry.get("summary", entry.get("title", ""))
        texts.append(BeautifulSoup(snippet, "html.parser").get_text())
    return texts

def compute_score(texts, tokenizer, model):
    scores = []
    for t in texts:
        inputs = tokenizer(t, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        # FinBERT: idx0=neg, idx2=pos
        scores.append(probs[2].item() - probs[0].item())
    return np.mean(scores) if scores else np.nan

def main():
    # load FinBERT once
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")
    model     = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-pretrain")

    # current UTC‐hour timestamp
    ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()

    rows = []
    for sym, url in SYMBOL_FEEDS.items():
        texts = fetch_texts(url)
        score = compute_score(texts, tokenizer, model)
        rows.append({"timestamp": ts, "symbol": sym, "score": score})

    df_new = pd.DataFrame(rows)
    csv_path = "sentiment.csv"

    try:
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df = df_new

    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()
