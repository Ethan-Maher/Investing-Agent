"""
Evaluation script for trained DQN trading agent.
Compares DQN greedy policy against buy-and-hold and random baselines.
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.trading_env import TradingEnv
from models.dqn import create_cnn_dqn

# Tickers to evaluate
TICKERS_TO_EVAL = [
    "SPY.US",
    "QQQ.US",
    "AAPL.US",
    "MSFT.US",
    "TSLA.US",
    "AMZN.US",
    "NVDA.US"
]

# These must match your processed CSV names (case-insensitive).


def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple MPS GPU")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


device = get_device()


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed for reproducibility
set_seed(42)


def evaluate_ticker(model, ticker, max_steps=300, window_size=30):
    """
    Runs DQN, Buy & Hold, and Random on a single ticker.
    Returns a dict with three time-series.
    
    Args:
        model: Trained CNN DQN model
        ticker: Ticker symbol (e.g., "SPY.US")
        max_steps: Maximum number of steps
        window_size: Window size for the environment
        
    Returns:
        Dictionary with "dqn", "buy_hold", and "random" portfolio value arrays
    """
    # Force env to load only this ticker
    ticker_file = f"{ticker}.csv"
    
    # ---- DQN ----
    env = TradingEnv(
        window_size=window_size,
        random_ticker=False,
        tickers=[ticker_file]
    )
    dqn_result = run_greedy_policy(env, model, max_steps=max_steps)
    
    # ---- Buy & Hold ----
    env_bh = TradingEnv(
        window_size=window_size,
        random_ticker=False,
        tickers=[ticker_file]
    )
    bh_result = run_buy_and_hold(env_bh, max_steps=max_steps)
    
    # ---- Random ----
    env_rand = TradingEnv(
        window_size=window_size,
        random_ticker=False,
        tickers=[ticker_file]
    )
    rand_result = run_random_policy(env_rand, max_steps=max_steps)
    
    return {
        "dqn": dqn_result["portfolio_values"],
        "buy_hold": bh_result["portfolio_values"],
        "random": rand_result["portfolio_values"]
    }


def load_trained_model(env, model_path="models/trained_dqn.pth", hidden_dim=64):
    """
    Load a trained CNN DQN model from file.
    
    Args:
        env: Trading environment instance
        model_path: Path to the saved model
        hidden_dim: Hidden dimension of the model
        
    Returns:
        Tuple of (model, window, num_features)
    """
    # Create a dummy obs to infer dimensions
    obs = env.reset(start_index=30)
    window, num_features = obs.shape
    output_dim = 7
    
    model = create_cnn_dqn(window, num_features, output_dim=output_dim, hidden_dim=hidden_dim)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, window, num_features


def run_greedy_policy(env, model, max_steps=1000):
    """
    Run the DQN model with greedy policy (epsilon=0, always argmax Q).
    
    Args:
        env: Trading environment instance
        model: Trained CNN DQN model
        max_steps: Maximum number of steps
        
    Returns:
        Dictionary with portfolio_values, rewards, and steps
    """
    obs = env.reset(start_index=30)
    state = obs  # Keep as 2D (window, features)
    done = False
    portfolio_values = []
    rewards = []
    step_count = 0
    
    while not done and step_count < max_steps:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = model(state_tensor)
            action = q.argmax().item()
        
        next_obs, reward, done, info = env.step(action)
        next_state = next_obs  # Keep as 2D (window, features)
        
        portfolio_values.append(info["portfolio_value"])
        rewards.append(reward)
        
        state = next_state
        step_count += 1
    
    return {
        "portfolio_values": np.array(portfolio_values),
        "rewards": np.array(rewards),
        "steps": step_count
    }


def run_buy_and_hold(env, max_steps=1000):
    """
    Run buy-and-hold baseline strategy.
    Buys 100% on first step, then holds to the end.
    
    Args:
        env: Trading environment instance
        max_steps: Maximum number of steps
        
    Returns:
        Dictionary with portfolio_values, rewards, and steps
    """
    obs = env.reset(start_index=30)
    state = obs  # Keep as 2D (window, features)
    done = False
    portfolio_values = []
    rewards = []
    step_count = 0
    bought = False
    
    while not done and step_count < max_steps:
        if not bought:
            action = 3  # Buy 100%
            bought = True
        else:
            action = 0  # Hold
        
        next_obs, reward, done, info = env.step(action)
        next_state = next_obs  # Keep as 2D (window, features)
        
        portfolio_values.append(info["portfolio_value"])
        rewards.append(reward)
        
        state = next_state
        step_count += 1
    
    return {
        "portfolio_values": np.array(portfolio_values),
        "rewards": np.array(rewards),
        "steps": step_count
    }


def run_random_policy(env, max_steps=1000):
    """
    Run random policy baseline strategy.
    Randomly selects actions (0-6: Hold, Buy 25%, Buy 50%, Buy 100%, Sell 25%, Sell 50%, Sell 100%).
    
    Args:
        env: Trading environment instance
        max_steps: Maximum number of steps
        
    Returns:
        Dictionary with portfolio_values, rewards, and steps
    """
    obs = env.reset(start_index=30)
    state = obs  # Keep as 2D (window, features)
    done = False
    portfolio_values = []
    rewards = []
    step_count = 0
    
    while not done and step_count < max_steps:
        action = random.randint(0, 6)
        
        next_obs, reward, done, info = env.step(action)
        next_state = next_obs  # Keep as 2D (window, features)
        
        portfolio_values.append(info["portfolio_value"])
        rewards.append(reward)
        
        state = next_state
        step_count += 1
    
    return {
        "portfolio_values": np.array(portfolio_values),
        "rewards": np.array(rewards),
        "steps": step_count
    }


def plot_portfolio_comparison(greedy, buy_hold, random_res, save_path):
    """
    Plot portfolio value comparison across all three strategies.
    
    Args:
        greedy: Results from greedy policy
        buy_hold: Results from buy-and-hold policy
        random_res: Results from random policy
        save_path: Path to save the plot
    """
    min_len = min(
        len(greedy["portfolio_values"]),
        len(buy_hold["portfolio_values"]),
        len(random_res["portfolio_values"])
    )
    
    plt.figure(figsize=(12, 6))
    plt.plot(greedy["portfolio_values"][:min_len], label="DQN Greedy", linewidth=2)
    plt.plot(buy_hold["portfolio_values"][:min_len], label="Buy & Hold", linewidth=2)
    plt.plot(random_res["portfolio_values"][:min_len], label="Random", linewidth=2)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Portfolio Value", fontsize=12)
    plt.legend(fontsize=11)
    plt.title("Portfolio Value Comparison", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Load trained model
    model_path = Path("models/trained_dqn.pth")
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using: make train")
        sys.exit(1)
    
    print("Loading trained model...")
    # Initialize environment with deterministic settings for model loading
    env = TradingEnv(window_size=30, random_ticker=False)
    model, window, num_features = load_trained_model(env, model_path=str(model_path))
    print(f"Model loaded successfully (window={window}, num_features={num_features})")
    
    # Create plots directory
    plots_dir = Path("evaluation") / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on multiple tickers
    results = {}
    
    for ticker in TICKERS_TO_EVAL:
        print(f"Evaluating {ticker}...")
        res = evaluate_ticker(model, ticker)
        results[ticker] = res
        
        # Plot per-ticker comparison
        plt.figure(figsize=(12, 6))
        
        dqn_vals = res["dqn"]
        bh_vals = res["buy_hold"]
        rand_vals = res["random"]
        
        min_len = min(len(dqn_vals), len(bh_vals), len(rand_vals))
        
        plt.plot(dqn_vals[:min_len], label="DQN", linewidth=2)
        plt.plot(bh_vals[:min_len], label="Buy & Hold", linewidth=2)
        plt.plot(rand_vals[:min_len], label="Random", linewidth=2)
        
        plt.title(f"Performance Comparison — {ticker}", fontsize=14, fontweight='bold')
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Portfolio Value", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(plots_dir / f"{ticker}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot for {ticker}")
    
    # Create summary CSV
    import csv
    summary_path = Path("evaluation") / "multi_ticker_summary.csv"
    
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "dqn_final", "buy_hold_final", "random_final"])
        
        for ticker, res in results.items():
            dqn_f = res["dqn"][-1]
            bh_f = res["buy_hold"][-1]
            rand_f = res["random"][-1]
            writer.writerow([ticker, dqn_f, bh_f, rand_f])
    
    print(f"\nSaved summary CSV to {summary_path}")
    
    # Combined multi-ticker plot (average of each strategy across all tickers)
    plt.figure(figsize=(12, 6))
    
    # Collect all portfolio values for each strategy
    all_dqn = []
    all_buy_hold = []
    all_random = []
    
    for ticker, res in results.items():
        all_dqn.append(res["dqn"])
        all_buy_hold.append(res["buy_hold"])
        all_random.append(res["random"])
    
    # Find minimum length to align all arrays
    min_len = min(
        min(len(vals) for vals in all_dqn),
        min(len(vals) for vals in all_buy_hold),
        min(len(vals) for vals in all_random)
    )
    
    # Truncate all arrays to min_len and compute averages
    dqn_avg = np.mean([vals[:min_len] for vals in all_dqn], axis=0)
    bh_avg = np.mean([vals[:min_len] for vals in all_buy_hold], axis=0)
    rand_avg = np.mean([vals[:min_len] for vals in all_random], axis=0)
    
    # Plot averages
    plt.plot(dqn_avg, label="DQN (Average)", linewidth=2)
    plt.plot(bh_avg, label="Buy & Hold (Average)", linewidth=2)
    plt.plot(rand_avg, label="Random (Average)", linewidth=2)
    
    plt.title("Average Performance Across All Tickers", fontsize=14, fontweight='bold')
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Portfolio Value", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "all_tickers_dqn_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined average comparison plot to {plots_dir / 'all_tickers_dqn_comparison.png'}")
    
    # Print summary
    print("\n" + "="*60)
    print("Final portfolio values by ticker and strategy:")
    print("="*60)
    print(f"{'Ticker':<12} {'DQN':>15} {'Buy & Hold':>15} {'Random':>15}")
    print("-" * 60)
    for ticker, res in results.items():
        dqn_f = res["dqn"][-1]
        bh_f = res["buy_hold"][-1]
        rand_f = res["random"][-1]
        print(f"{ticker:<12} ${dqn_f:>14,.2f} ${bh_f:>14,.2f} ${rand_f:>14,.2f}")
    print("="*60)
