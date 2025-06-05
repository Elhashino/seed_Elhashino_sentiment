#!/usr/bin/env python3
# update_sentiment.py
#
# This script:
#   1) Builds or loads today’s (latest) sentiment into a DataFrame `new_df`
#   2) Renames any "BTC-USD" → "BTCUSD" in both new and old data
#   3) Reads the existing CSV (if it exists), normalizes it, then appends new rows
#   4) Drops duplicates by ("ticker", "date"), keeping the most recent entries
#   5) Writes out the merged CSV at "sentiment_scores.csv"
#
# USAGE:
#   python update_sentiment.py
#
# If you already have a different path for your CSV (e.g. data/sentiment.csv),
# you can pass that on the command line:
#   python update_sentiment.py data/sentiment.csv

import os
import sys
import pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1) REPLACE THIS FUNCTION body with your own fetching/computation logic.
#
# It must return a DataFrame `new_df` with at least these columns:
#    ["ticker", "date", "sentiment", …optional extra columns…]
#
# Example stub below returns two sample rows:
#    • ticker = "BTC-USD" and "ETHUSD"
#    • date = "YYYY-MM-DD"
#    • sentiment = float score
#
# Overwrite this function so it actually fetches or computes your real sentiment data.
# ──────────────────────────────────────────────────────────────────────────────
def build_new_sentiment_dataframe() -> pd.DataFrame:
    """
    Build the latest sentiment DataFrame. 
    Must return columns including: 'ticker', 'date', 'sentiment', etc.
    REPLACE the placeholder below with your own code.
    """
    # ── BEGIN PLACEHOLDER ──
    # Example dummy data; replace with your actual logic:
    sample_data = [
        {"ticker": "BTC-USD", "date": "2025-06-05", "sentiment": 0.12},
        {"ticker": "ETHUSD",  "date": "2025-06-05", "sentiment": 0.05},
    ]
    new_df = pd.DataFrame(sample_data)
    # ──   END PLACEHOLDER  ──

    # Ensure 'date' is in "YYYY-MM-DD" format (uncomment if needed):
    # new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")

    return new_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2) Main function that:
#   • Calls build_new_sentiment_dataframe()
#   • Normalizes ticker names in new + old data (rename "BTC-USD" → "BTCUSD")
#   • Loads existing CSV (if present), merges, drops duplicates, and writes out CSV
# ──────────────────────────────────────────────────────────────────────────────
def refresh_sentiment_csv(csv_path: str = "sentiment_scores.csv") -> None:
    """
    Loads/builds new sentiment data, normalizes tickers, merges with old CSV (if any),
    drops (ticker, date) duplicates, and writes out the updated CSV.
    """
    # 1) Build the new sentiment DataFrame
    try:
        new_df = build_new_sentiment_dataframe()
    except Exception as e:
        print(f"ERROR: Failed to build new sentiment DataFrame: {e}")
        raise

    # 2) Normalize ticker names in NEW data (rename "BTC-USD" to "BTCUSD")
    new_df["ticker"] = new_df["ticker"].replace({"BTC-USD": "BTCUSD"})

    # 3) Load the old CSV if it exists; else create an empty DataFrame
    if os.path.isfile(csv_path):
        try:
            old_df = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            print(f"ERROR: Could not read existing CSV '{csv_path}': {e}")
            raise
    else:
        # No existing CSV → make an empty DataFrame with the same columns as new_df
        old_df = pd.DataFrame(columns=new_df.columns)

    # 4) Normalize ticker names in OLD data as well
    if "ticker" in old_df.columns:
        old_df["ticker"] = old_df["ticker"].replace({"BTC-USD": "BTCUSD"})
    else:
        # If there’s no 'ticker' column, warn and proceed
        print("WARNING: 'ticker' column not found in old CSV; skipping normalization of old data.")

    # 5) Ensure 'date' columns are consistently "YYYY-MM-DD" (if they exist)
    if "date" in old_df.columns:
        try:
            old_df["date"] = pd.to_datetime(old_df["date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            pass  # assume old data is already correct
    if "date" in new_df.columns:
        try:
            new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # 6) Concatenate old + new, then drop duplicates by ['ticker','date']
    combined = pd.concat([old_df, new_df], ignore_index=True)

    # Only drop duplicates if those columns exist
    if {"ticker", "date"}.issubset(set(combined.columns)):
        # keep="last" → if there's a conflict (same ticker/day), prefer the new row
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    else:
        print("WARNING: Cannot drop duplicates because 'ticker' and/or 'date' columns are missing.")

    # 7) (Optional) Sort by date (asc) then ticker (asc), for human readability
    if {"date", "ticker"}.issubset(set(combined.columns)):
        combined = combined.sort_values(by=["date", "ticker"], ascending=[True, True])

    # 8) Write the final DataFrame back to CSV (no index column)
    try:
        combined.to_csv(csv_path, index=False)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] Successfully wrote updated CSV: '{csv_path}'")
    except Exception as e:
        print(f"ERROR: Failed to write combined DataFrame to CSV: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# If run directly (python update_sentiment.py), call refresh_sentiment_csv(…)
# You can optionally pass your own path, e.g.:
#   python update_sentiment.py data/my_sentiment.csv
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = "sentiment_scores.csv"

    refresh_sentiment_csv(csv_path=target_path)
