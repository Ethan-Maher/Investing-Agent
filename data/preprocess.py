"""
Preprocess raw stock data: clean, add indicators, split into train/test, and normalize.
"""

import json
import pandas as pd
from pathlib import Path

from indicators.technical_indicators import (
    add_sma, add_ema, add_rsi, add_macd, add_all_indicators
)

# Global time-based split date
SPLIT_DATE = "2020-01-01"
# Epsilon for avoiding divide-by-zero in normalization
EPSILON = 1e-8


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


def compute_scaler_params(train_df):
    """
    Compute min and max values from training data only.
    
    Args:
        train_df: Training DataFrame
        
    Returns:
        Dictionary mapping column names to {'min': float, 'max': float}
    """
    # Identify numeric columns (exclude Date)
    numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    scaler_params = {}
    for col in numeric_cols:
        col_min = float(train_df[col].min())
        col_max = float(train_df[col].max())
        scaler_params[col] = {'min': col_min, 'max': col_max}
    
    return scaler_params


def normalize_with_scaler(df, scaler_params):
    """
    Normalize DataFrame using pre-computed min/max values.
    
    Formula: (x - min) / (max - min + epsilon)
    
    Args:
        df: DataFrame to normalize
        scaler_params: Dictionary mapping column names to {'min': float, 'max': float}
        
    Returns:
        Normalized DataFrame
    """
    df = df.copy()
    
    for col, params in scaler_params.items():
        if col not in df.columns:
            continue
        
        col_min = params['min']
        col_max = params['max']
        col_range = col_max - col_min
        
        # Normalize with epsilon to avoid divide-by-zero
        if col_range > 0:
            df[col] = (df[col] - col_min) / (col_range + EPSILON)
        else:
            # If all values are the same, set to 0
            df[col] = 0.0
    
    return df


def save_scaler_params(ticker, scaler_params, scalers_dir):
    """
    Save scaler parameters to JSON file.
    
    Args:
        ticker: Ticker symbol
        scaler_params: Dictionary of scaler parameters
        scalers_dir: Directory to save scaler files
    """
    scalers_dir_path = Path(scalers_dir)
    scalers_dir_path.mkdir(parents=True, exist_ok=True)
    
    scaler_path = scalers_dir_path / f"{ticker}_scaler.json"
    
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)


def process_ticker(ticker, raw_dir, processed_dir):
    """
    Process a single ticker: load, clean, add indicators, split by date, and save.
    
    Args:
        ticker: Ticker symbol
        raw_dir: Path to raw data directory
        processed_dir: Path to processed data directory
        
    Returns:
        Tuple of (train_rows, test_rows) or None if failed
    """
    # Load raw data
    df = load_raw_data(ticker, raw_dir)
    if df is None:
        return None
    
    # Clean the data (sorts by date)
    df = clean_data(df)
    
    if df.empty:
        return None
    
    # Add indicators
    df = add_all_indicators(df)
    
    # Drop rows with NaNs (indicator warm-up period)
    df = df.dropna()
    
    if df.empty:
        return None
    
    # Ensure Date is datetime for comparison
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Split into train and test based on SPLIT_DATE
    split_date = pd.to_datetime(SPLIT_DATE)
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    
    # Assertions: both splits must be non-empty
    assert not train_df.empty, f"Train split is empty for {ticker}. Need data before {SPLIT_DATE}."
    assert not test_df.empty, f"Test split is empty for {ticker}. Need data on or after {SPLIT_DATE}."
    
    # Compute normalization parameters from TRAIN data only
    scaler_params = compute_scaler_params(train_df)
    
    # Normalize both train and test using train statistics
    train_df_normalized = normalize_with_scaler(train_df, scaler_params)
    test_df_normalized = normalize_with_scaler(test_df, scaler_params)
    
    # Create processed directory
    processed_dir_path = Path(processed_dir)
    processed_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save scaler parameters
    scalers_dir = processed_dir_path / "scalers"
    save_scaler_params(ticker, scaler_params, scalers_dir)
    
    # Convert Date back to string format for CSV
    train_df_normalized['Date'] = train_df_normalized['Date'].dt.strftime('%Y-%m-%d')
    test_df_normalized['Date'] = test_df_normalized['Date'].dt.strftime('%Y-%m-%d')
    
    # Save train and test splits as separate CSVs
    train_path = processed_dir_path / f"{ticker}_train.csv"
    test_path = processed_dir_path / f"{ticker}_test.csv"
    
    train_df_normalized.to_csv(train_path, index=False)
    test_df_normalized.to_csv(test_path, index=False)
    
    return (len(train_df_normalized), len(test_df_normalized))


def process_all():
    """Process all tickers."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    tickers = get_tickers()
    
    print(f"Splitting data at {SPLIT_DATE}")
    print("=" * 60)
    
    for ticker in tickers:
        result = process_ticker(ticker, raw_dir, processed_dir)
        
        if result is not None:
            train_rows, test_rows = result
            print(f"Processed {ticker} - Train: {train_rows} rows, Test: {test_rows} rows")
        else:
            print(f"Failed to process {ticker}")


if __name__ == "__main__":
    process_all()
