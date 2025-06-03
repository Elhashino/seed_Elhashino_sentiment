# test_sentiment.py

import pytest
from update_sentiment import compute_sentiment
from fetcher import fetch_reddit

def test_compute_sentiment_output_types():
    score, count = compute_sentiment(["The market looks good today."])
    assert isinstance(score, float)
    assert isinstance(count, int)

def test_empty_input():
    score, count = compute_sentiment([])
    assert score == 0.0
    assert count == 0
