# seed_Elhashino_sentiment
Hourly AI-computed sentiment data for TradingView

## Running the updater

Execute `python update_sentiment.py` to pull recent news from Google, Reuters,
Reddit and Investing.com. Sentiment scores are appended to `sentiment_scores.csv`
in the project root.

## Launching the dashboard

Start the web interface with `streamlit run dashboard.py`. The dashboard reads
`sentiment_scores.csv` and displays the latest score for each symbol. If the CSV
is missing you'll be prompted to run the updater first.

## GitHub Actions workflow

The repository contains a workflow that runs `update_sentiment.py` every hour and
commits the updated CSV back to the repository.
