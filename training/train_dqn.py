"""
Training script for DQN trading agent.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import csv
import torch
import torch.nn as nn
import numpy as np
import random
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.trading_env import TradingEnv
from models.dqn import create_cnn_dqn
from models.replay_buffer import ReplayBuffer


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
    # Enable cuDNN benchmark for better GPU performance (disable for reproducibility)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Set seed for reproducibility
set_seed(42)


def to_tensor(obs):
    """
    Convert observation to tensor, keeping 2D shape (window, features).
    
    Args:
        obs: Observation array of shape (window_size, num_features)
        
    Returns:
        Tensor of shape (window_size, num_features)
    """
    return torch.tensor(obs, dtype=torch.float32, device=device)


def select_action(state, policy_net, epsilon):
    """
    Select an action using epsilon-greedy policy.
    
    Args:
        state: State array of shape (window_size, num_features)
        policy_net: Policy network
        epsilon: Exploration rate
        
    Returns:
        Selected action (0-6)
    """
    if random.random() < epsilon:
        return random.randrange(7)
    else:
        with torch.no_grad():
            state_tensor = to_tensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()


def init_metrics_file(path):
    """
    Initialize metrics CSV file with header if it doesn't exist.
    
    Args:
        path: Path to the metrics CSV file
    """
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["episode", "total_reward", "epsilon", "steps"])


def train_dqn():
    """Main training function."""
    # Training parameters
    TRAIN_EVERY = 1  # update network every step
    GRAD_UPDATES_PER_STEP = 8  # do 8 minibatch updates each time we train
    WARMUP = 5000  # minimum transitions before training starts
    episodes = 1000
    max_steps_per_episode = 200
    batch_size = 1024  # GPU-heavy batch size (try 2048 if memory allows)
    gamma = 0.995
    lr = 1e-4  # start with 1e-4, reduce to 5e-5 if unstable
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 10000  # decay measured in training steps, not episodes
    target_update_interval = 500
    replay_capacity = 50_000
    
    # Initialize environment in train mode
    env = TradingEnv(window_size=30, mode="train")
    
    # Determine state shape (keep as 2D)
    sample_obs = env.reset()
    window, num_features = sample_obs.shape
    state_shape = (window, num_features)
    
    print(f"State shape: {state_shape} (window={window}, features={num_features})")
    
    # Create networks
    policy_net = create_cnn_dqn(window, num_features, output_dim=7, hidden_dim=128)
    target_net = create_cnn_dqn(window, num_features, output_dim=7, hidden_dim=128)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    policy_net.to(device)
    target_net.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    
    # Replay buffer
    replay = ReplayBuffer(replay_capacity)
    
    # Loss function: Huber loss (SmoothL1Loss) for better stability
    criterion = nn.SmoothL1Loss()
    
    # Mixed precision training (AMP) for CUDA
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Setup metrics logging
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    metrics_path = results_dir / "training_metrics.csv"
    init_metrics_file(metrics_path)
    
    # Track recent rewards for running average
    recent_rewards = []
    
    # Training loop
    global_step = 0
    epsilon = epsilon_start
    
    for episode in range(episodes):
        state_obs = env.reset()
        state = state_obs  # Keep as 2D (window, features)
        total_reward = 0.0
        step_count = 0
        
        for t in range(max_steps_per_episode):
            # Select action
            action = select_action(state, policy_net, epsilon)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            next_state = next_obs  # Keep as 2D (window, features)
            
            # Store transition in replay buffer (states are 2D arrays)
            replay.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            global_step += 1
            
            # Train if we have enough samples and on schedule (after warmup)
            if (global_step % TRAIN_EVERY == 0 and 
                len(replay) >= WARMUP and 
                len(replay) > batch_size):
                
                # Multiple gradient updates per step for better learning
                for _ in range(GRAD_UPDATES_PER_STEP):
                    # Sample batch
                    states, actions, rewards, next_states, dones = replay.sample(batch_size)
                    
                    # Convert to tensors - keep all on CUDA device
                    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                    next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
                    
                    # Use mixed precision if available
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            # Compute Q-values for current states
                            q_values = policy_net(states_tensor)
                            q_action = q_values.gather(1, actions_tensor)
                            
                            # Double DQN: use policy net to select actions, target net to evaluate
                            with torch.no_grad():
                                next_actions = policy_net(next_states_tensor).argmax(dim=1)
                                next_q = target_net(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                                expected_q = rewards_tensor + gamma * next_q * (1 - dones_tensor)
                            
                            # Compute loss
                            loss = criterion(q_action.squeeze(), expected_q)
                        
                        # Optimize with gradient scaling
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard precision training
                        # Compute Q-values for current states
                        q_values = policy_net(states_tensor)
                        q_action = q_values.gather(1, actions_tensor)
                        
                        # Double DQN: use policy net to select actions, target net to evaluate
                        with torch.no_grad():
                            next_actions = policy_net(next_states_tensor).argmax(dim=1)
                            next_q = target_net(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                            expected_q = rewards_tensor + gamma * next_q * (1 - dones_tensor)
                        
                        # Compute loss
                        loss = criterion(q_action.squeeze(), expected_q)
                        
                        # Optimize with gradient clipping
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                        optimizer.step()
            
            # Decay epsilon based on global step
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-global_step / epsilon_decay)
            
            if done:
                break
        
        # Update target network
        if episode % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Log metrics to CSV
        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, epsilon, step_count])
        
        # Track recent rewards for running average
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
        avg_reward = np.mean(recent_rewards) if recent_rewards else total_reward
        
        # Print progress
        warmup_status = f" (Warming up: {len(replay)}/{WARMUP})" if len(replay) < WARMUP else ""
        print(f'Episode {episode} - Reward: {total_reward:.2f}, Avg (last 10): {avg_reward:.2f}, Epsilon: {epsilon:.3f}, Steps: {step_count}, Portfolio: {info["portfolio_value"]:.2f}, Replay: {len(replay)}{warmup_status}')
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "trained_dqn_noleak.pth"
    torch.save(policy_net.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train_dqn()
