"""
Technical indicators for stock data analysis.
"""

import pandas as pd


def add_sma(df, window=14):
    """
    Add Simple Moving Average indicator.
    
    Args:
        df: DataFrame with 'Close' column
        window: Window size for SMA (default: 14)
        
    Returns:
        DataFrame with added SMA_{window} column
    """
    df = df.copy()
    df[f"SMA_{window}"] = df["Close"].rolling(window).mean()
    return df


def add_ema(df, window=14):
    """
    Add Exponential Moving Average indicator.
    
    Args:
        df: DataFrame with 'Close' column
        window: Window size for EMA (default: 14)
        
    Returns:
        DataFrame with added EMA_{window} column
    """
    df = df.copy()
    df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    return df


def add_rsi(df, window=14):
    """
    Add Relative Strength Index indicator.
    
    Args:
        df: DataFrame with 'Close' column
        window: Window size for RSI (default: 14)
        
    Returns:
        DataFrame with added RSI_{window} column
    """
    df = df.copy()
    
    # Compute price changes (delta)
    delta = df["Close"].diff()
    
    # Compute gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Compute average gain/loss with .ewm()
    avg_gain = gains.ewm(span=window, adjust=False).mean()
    avg_loss = losses.ewm(span=window, adjust=False).mean()
    
    # Compute RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Compute RSI = 100 - (100 / (1 + RS))
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    
    return df


def add_macd(df):
    """
    Add MACD (Moving Average Convergence Divergence) indicator.
    
    Creates three columns:
    - MACD: EMA_12 - EMA_26
    - MACD_signal: EMA_9 of MACD
    - MACD_hist: MACD - MACD_signal
    
    Args:
        df: DataFrame with 'Close' column
        
    Returns:
        DataFrame with added MACD, MACD_signal, MACD_hist columns
    """
    df = df.copy()
    
    # Calculate EMA_12 and EMA_26
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    
    # MACD = EMA_12 - EMA_26
    df["MACD"] = ema_12 - ema_26
    
    # Signal = EMA_9 of MACD
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # Histogram = MACD - Signal
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    
    return df


def add_all_indicators(df):
    """
    Add all technical indicators to the DataFrame.
    
    Args:
        df: DataFrame with 'Close' column
        
    Returns:
        DataFrame with all indicators added
    """
    df = add_sma(df, 14)
    df = add_ema(df, 14)
    df = add_rsi(df, 14)
    df = add_macd(df)
    return df
