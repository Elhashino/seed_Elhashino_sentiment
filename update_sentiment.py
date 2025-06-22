import os
import pandas as pd
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ——— CONFIG ———
SYMBOLS = [
    "AAPL","AUDUSD","BTCUSD","CL","EURUSD",
    "GBPUSD","SPY","USDCAD","USDJPY","XAUUSD",
]
CSV_FILE = "sentiment_scores.csv"


def load_news_for(symbol):
    """
    Replace this with your real scraping / RSS‐parsing logic.
    For example you might read from data/news_{symbol}.txt
    """
    path = f"data/news_{symbol}.txt"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_nlp_pipeline():
    # switch to a FinBERT model that actually has a classification head
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )


def compute_score(texts, nlp):
    scores = []
    for txt in texts:
        try:
            out = nlp(txt)[0]  # a list of dicts: [{'label':'NEG',...}, {'label':'NEU',...}, ...]
            # map label→score
            d = { item["label"].lower(): item["score"] for item in out }
            # subtract negative from positive (ignore neutral if present)
            pos = d.get("positive", 0.0)
            neg = d.get("negative", 0.0)
            scores.append(pos - neg)
        except Exception as e:
            print(f"Error scoring “{txt[:30]}…”: {e}")
            # skip or append zero
    return scores


def main():
    # load old CSV or make new DataFrame
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
    else:
        df = pd.DataFrame(columns=["symbol", "timestamp", "score"])

    nlp = get_nlp_pipeline()
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    new_rows = []
    for sym in SYMBOLS:
        texts = load_news_for(sym)
        if not texts:
            continue
        scs = compute_score(texts, nlp)
        for s in scs:
            new_rows.append({
                "symbol":    sym,
                "timestamp": now,
                "score":     s
            })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        print(f"Appended {len(new_rows)} rows for {now}")
    else:
        print("No new data to append.")


if __name__ == "__main__":
    main()
