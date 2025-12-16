#!/usr/bin/env python3
"""
Download historical daily OHLCV stock data from Stooq.
"""

import requests
import pandas as pd
import os
from pathlib import Path


def download_ticker_data(ticker, output_dir):
    """
    Download CSV data for a ticker from Stooq and save to file.
    
    Args:
        ticker: Ticker symbol (e.g., "SPY.US")
        output_dir: Directory to save the CSV file
        
    Returns:
        tuple: (success: bool, rows: int, error: str or None)
    """
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    
    try:
        # Download the CSV
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Read CSV into pandas
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        # Check if we got valid data
        if df.empty:
            return False, 0, "Empty response from server"
        
        # Clean the data
        # Ensure columns are: Date, Open, High, Low, Close, Volume
        # Stooq typically uses: Date, Open, High, Low, Close, Volume
        # But column names might vary, so let's standardize them
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if we have the right columns (case-insensitive)
        df.columns = df.columns.str.strip()
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == 'date':
                col_mapping[col] = 'Date'
            elif col_lower == 'open':
                col_mapping[col] = 'Open'
            elif col_lower == 'high':
                col_mapping[col] = 'High'
            elif col_lower == 'low':
                col_mapping[col] = 'Low'
            elif col_lower == 'close':
                col_mapping[col] = 'Close'
            elif col_lower == 'volume':
                col_mapping[col] = 'Volume'
        
        if len(col_mapping) < 6:
            return False, 0, f"Missing required columns. Found: {list(df.columns)}"
        
        df = df.rename(columns=col_mapping)
        
        # Select only the columns we need
        df = df[expected_cols]
        
        # Parse dates into YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows with missing data
        df = df.dropna()
        
        # Sort by Date ascending
        df = df.sort_values('Date')
        
        # Format date as YYYY-MM-DD string
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Reset index to have Date as a regular column
        df = df.reset_index(drop=True)
        
        # Save to file
        output_path = Path(output_dir) / f"{ticker}.csv"
        df.to_csv(output_path, index=False)
        
        return True, len(df), None
        
    except requests.exceptions.RequestException as e:
        return False, 0, f"Network error: {str(e)}"
    except pd.errors.EmptyDataError:
        return False, 0, "Empty CSV data"
    except Exception as e:
        return False, 0, str(e)


def main():
    """Main function to download data for all tickers."""
    tickers = [
        "SPY.US", "QQQ.US", "AAPL.US", "MSFT.US", "TSLA.US",
        "AMZN.US", "NVDA.US", "JPM.US", "META.US", "VOO.US"
    ]
    
    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data for each ticker
    for ticker in tickers:
        success, rows, error = download_ticker_data(ticker, output_dir)
        
        if success:
            print(f"Downloaded {ticker} ({rows} rows)")
        else:
            print(f"Failed {ticker}: {error}")


if __name__ == "__main__":
    main()
