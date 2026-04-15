"""Download financial data for all configured tickers."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.utils.helpers import load_config, ensure_dir


def main():
    config = load_config()
    fetcher = DataFetcher(config)
    fe = FeatureEngineer(config)

    tickers = config["data"]["tickers"]
    print(f"Downloading data for {len(tickers)} tickers: {tickers}\n")

    for ticker in tickers:
        print(f"--- {ticker} ---")
        df = fetcher.fetch(ticker=ticker, force=True)
        print(f"  Raw: {df.shape[0]} rows, {df.shape[1]} columns")

        # Engineer features and save processed version
        featured = fe.engineer(df)
        processed_dir = ensure_dir(config["data"]["processed_dir"])
        out_path = os.path.join(processed_dir, f"{ticker}_featured.csv")
        featured.to_csv(out_path)
        print(f"  Featured: {featured.shape[0]} rows, {featured.shape[1]} columns")
        print(f"  Saved to {out_path}\n")

    # Save a small sample for testing
    sample_dir = ensure_dir(config["data"]["sample_dir"])
    default = config["data"]["default_ticker"]
    df = fetcher.fetch(ticker=default)
    sample = fe.engineer(df).tail(100)
    sample_path = os.path.join(sample_dir, "sample.csv")
    sample.to_csv(sample_path)
    print(f"Sample saved to {sample_path} ({len(sample)} rows)")


if __name__ == "__main__":
    main()
