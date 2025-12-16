"""
Preprocess raw stock data: clean, add indicators, and normalize.
"""

import pandas as pd
from pathlib import Path

from indicators.technical_indicators import (
    add_sma, add_ema, add_rsi, add_macd, add_all_indicators
)


def get_tickers():
    """Get list of all tickers to process."""
    return [
        "SPY.US", "QQQ.US", "AAPL.US", "MSFT.US", "TSLA.US",
        "AMZN.US", "NVDA.US", "JPM.US", "META.US", "VOO.US"
    ]


def load_raw_data(ticker, raw_dir):
    """
    Load raw CSV data for a ticker.
    
    Args:
        ticker: Ticker symbol (e.g., "SPY.US")
        raw_dir: Path to raw data directory
        
    Returns:
        DataFrame or None if file not found
    """
    raw_path = Path(raw_dir) / f"{ticker}.csv"
    
    if not raw_path.exists():
        return None
    
    try:
        df = pd.read_csv(raw_path)
        return df
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


def clean_data(df):
    """
    Clean the raw data.
    
    - Convert Date to datetime
    - Ensure correct column names: Date, Open, High, Low, Close, Volume
    - Sort by Date ascending
    - Drop rows with missing values
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Ensure correct column names (case-insensitive)
    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
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
    
    df = df.rename(columns=col_mapping)
    
    # Select only the columns we need
    df = df[expected_cols]
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Sort by Date ascending
    df = df.sort_values('Date')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def normalize_data(df):
    """
    Normalize numeric columns using min-max normalization.
    
    Formula: (x - x.min()) / (x.max() - x.min())
    
    Args:
        df: DataFrame with numeric columns to normalize
        
    Returns:
        DataFrame with normalized numeric columns
    """
    df = df.copy()
    
    # Identify numeric columns (exclude Date)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Normalize each numeric column
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Avoid division by zero
        if col_max - col_min != 0:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            # If all values are the same, set to 0
            df[col] = 0.0
    
    return df


def process_ticker(ticker, raw_dir, processed_dir):
    """
    Process a single ticker: load, clean, add indicators, normalize, and save.
    
    Args:
        ticker: Ticker symbol
        raw_dir: Path to raw data directory
        processed_dir: Path to processed data directory
        
    Returns:
        Number of rows in processed data, or None if failed
    """
    # Load raw data
    df = load_raw_data(ticker, raw_dir)
    if df is None:
        return None
    
    # Clean the data
    df = clean_data(df)
    
    if df.empty:
        return None
    
    # Add indicators
    df = add_all_indicators(df)
    
    # Drop rows with NaNs (indicator warm-up period)
    df = df.dropna()
    
    if df.empty:
        return None
    
    # Normalize numeric columns
    df = normalize_data(df)
    
    # Save to processed directory
    processed_path = Path(processed_dir) / f"{ticker}.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Date back to string format for CSV
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    df.to_csv(processed_path, index=False)
    
    return len(df)


def process_all():
    """Process all tickers."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    tickers = get_tickers()
    
    for ticker in tickers:
        rows = process_ticker(ticker, raw_dir, processed_dir)
        
        if rows is not None:
            print(f"Processed {ticker} → {rows} rows")
        else:
            print(f"Failed to process {ticker}")


if __name__ == "__main__":
    process_all()
