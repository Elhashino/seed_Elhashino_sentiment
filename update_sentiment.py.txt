import time
import pandas as pd
from transformers import pipeline

def fetch_dummy_sentiment():
    return 0.1  # placeholder

def build_dataframe():
    now_ms = int(time.time() * 1000)
    score = fetch_dummy_sentiment()
    return pd.DataFrame([{
        "time": now_ms,
        "open": score,
        "high": score,
        "low": score,
        "close": score
    }])

def main():
    df = build_dataframe()
    df.to_csv("SPY.csv", index=False)
    print("Wrote SPY.csv:", df)

if __name__ == "__main__":
    main()
