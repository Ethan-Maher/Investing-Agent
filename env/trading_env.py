"""
Trading environment for reinforcement learning agent.
Implements a Gym-like API for trading on processed stock data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random


class TradingEnv:
    """
    Trading environment for single-asset, long-only trading.
    Uses position-based rebalancing to prevent leverage and compounding exploits.
    
    Actions:
        0 = Hold (no change to position)
        1 = Increase position by 25%
        2 = Increase position by 50%
        3 = Increase position by 100% (go fully long)
        4 = Decrease position by 25%
        5 = Decrease position by 50%
        6 = Decrease position by 100% (go fully to cash)
    
    Position: 0.0 = 100% cash, 1.0 = 100% invested (fully long)
    """
    
    def __init__(self, data_dir="data/processed", window_size=30, 
                 initial_cash=10_000, tickers=None, random_ticker=True):
        """
        Initialize the trading environment.
        
        Args:
            data_dir: Folder with processed CSV files
            window_size: Number of past days in the observation window
            initial_cash: Starting cash amount
            tickers: Optional list of ticker filenames (e.g., ["SPY.US.csv", ...]).
                    If None, automatically detect all CSV files in data_dir.
            random_ticker: If True, each episode picks a random ticker; 
                          otherwise always use the first.
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.random_ticker = random_ticker
        
        # position ∈ [0.0, 1.0], where:
        # 0.0 = 100% cash
        # 1.0 = 100% invested (fully long)
        self.position = 0.0
        
        # Define action space: (action_type, fraction)
        # Actions now change position rather than buy/sell
        self.actions = {
            0: ("hold", 0.0),
            1: ("increase", 0.25),
            2: ("increase", 0.50),
            3: ("increase", 1.00),
            4: ("decrease", 0.25),
            5: ("decrease", 0.50),
            6: ("decrease", 1.00),
        }
        
        # Load available tickers
        if tickers is None:
            self._load_available_tickers()
        else:
            self.ticker_files = tickers
            self.ticker_symbols = [f.replace('.csv', '') for f in tickers]
        
        # Episode state
        self.df = None
        self.features = None
        self.current_step = None
        self.cash = None
        self.shares = None
        self.current_ticker = None
        
    def _load_available_tickers(self):
        """Scan data_dir for all CSV files and store ticker information."""
        csv_files = sorted(list(self.data_dir.glob("*.csv")))
        self.ticker_files = [f.name for f in csv_files]
        self.ticker_symbols = [f.stem for f in csv_files]
        
        if not self.ticker_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
    
    def reset(self, start_index=None):
        """
        Reset the environment for a new episode.
        
        Args:
            start_index: Optional starting index. If None, uses window_size.
        
        Returns:
            observation: numpy array of shape (window_size, num_features + 1)
        """
        # Select ticker for this episode
        if self.random_ticker:
            idx = random.randint(0, len(self.ticker_files) - 1)
        else:
            idx = 0
        
        ticker_file = self.ticker_files[idx]
        self.current_ticker = self.ticker_symbols[idx]
        
        # Load DataFrame
        df_path = self.data_dir / ticker_file
        self.df = pd.read_csv(df_path)
        
        # Ensure sorted by Date ascending
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Identify numeric feature columns (all except "Date")
        self.features = [col for col in self.df.columns if col != 'Date']
        
        # Initialize episode state
        if start_index is None:
            start_index = self.window_size
        self.current_step = start_index
        self.cash = self.initial_cash
        self.shares = 0.0
        self.position = 0.0  # Start with 100% cash
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def _get_observation(self):
        """
        Get the current observation window.
        
        Returns:
            numpy array of shape (window_size, num_features + 1)
            The +1 is for the position value (0.0 to 1.0, where 0.0 = 100% cash, 1.0 = 100% invested)
        """
        # Get the window of features
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # Extract feature values
        feature_window = self.df[self.features].iloc[start_idx:end_idx].values
        
        # If we don't have enough history, pad with the first row
        if len(feature_window) < self.window_size:
            padding = np.tile(feature_window[0:1], (self.window_size - len(feature_window), 1))
            feature_window = np.vstack([padding, feature_window])
        
        # Add position (0.0 to 1.0) as a feature column
        position_column = np.full((self.window_size, 1), self.position)
        
        # Concatenate features with position
        observation = np.hstack([feature_window, position_column])
        
        return observation
    
    def step(self, action):
        """
        Clean, safe, leverage-free trading step.
        Position is always in [0, 1].
        Portfolio is rebalanced at each step based on target position.
        """
        
        # Unpack action
        action_type, fraction = self.actions[action]
        
        # Compute next target position
        if action_type == "increase":
            new_position = min(1.0, self.position + fraction)
        elif action_type == "decrease":
            new_position = max(0.0, self.position - fraction)
        else:
            new_position = self.position
        
        # Current price
        price_t = self.df["Close"].iloc[self.current_step]
        
        # Portfolio BEFORE price change
        portfolio_before = self.cash + self.shares * price_t
        
        # Rebalance portfolio to new position (NO leverage)
        target_shares = new_position * (portfolio_before / price_t)
        target_cash   = portfolio_before - (target_shares * price_t)
        
        # Update state
        self.position = new_position
        self.shares   = target_shares
        self.cash     = target_cash
        
        # Advance time
        self.current_step += 1
        
        # Terminal check
        done = self.current_step >= len(self.df) - 1
        
        # Price after move
        price_next = self.df["Close"].iloc[self.current_step]
        portfolio_after = self.cash + self.shares * price_next
        
        # Log-return reward
        reward = np.log(portfolio_after + 1e-8) - np.log(portfolio_before + 1e-8)
        
        # Build observation (window_size days + position)
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        window = self.df[self.features].iloc[start:end].values
        
        # Pad if needed (shouldn't happen after reset, but safety check)
        if len(window) < self.window_size:
            padding = np.tile(window[0:1], (self.window_size - len(window), 1))
            window = np.vstack([padding, window])
        
        # Add position as extra feature channel
        pos_column = np.full((self.window_size, 1), self.position)
        obs = np.concatenate([window, pos_column], axis=1)
        
        info = {
            "cash": self.cash,
            "shares": self.shares,
            "position": self.position,
            "portfolio_value": portfolio_after,
            "ticker": self.current_ticker,
        }
        
        return obs, reward, done, info
    
    def render(self):
        """Print current state of the environment."""
        if self.df is None:
            print("Environment not initialized. Call reset() first.")
            return
        
        current_date = self.df.loc[self.current_step, 'Date']
        current_close = self.df.loc[self.current_step, 'Close']
        portfolio_value = self.cash + self.shares * current_close
        
        print(f"Ticker: {self.current_ticker}")
        print(f"Step: {self.current_step}/{len(self.df) - 1}")
        print(f"Date: {current_date}")
        print(f"Close Price: {current_close:.6f}")
        print(f"Position: {self.position:.2%}")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Shares: {self.shares:.6f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    env = TradingEnv()
    obs = env.reset()
    done = False
    total_reward = 0.0
    
    step_count = 0
    while not done and step_count < 200:
        action = 0  # always Hold for this quick smoke test
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
    
    print(f"Test run finished. Steps: {step_count}, Total reward: {total_reward:.2f}, Final portfolio value: {info['portfolio_value']:.2f}")
