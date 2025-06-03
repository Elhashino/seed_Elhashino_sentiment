import pytest

from update_sentiment import fetch_reddit, compute_sentiment


def test_fetch_reddit_returns_list():
    posts = fetch_reddit("EURUSD", max_items=1)
    assert isinstance(posts, list)


def test_compute_sentiment_output_types():
    score, count = compute_sentiment(["The market looks good today."])
    assert isinstance(score, float)
    assert isinstance(count, int)
    assert count == 1


def test_compute_sentiment_empty_input():
    score, count = compute_sentiment([])
    assert score == 0.0
    assert count == 0
