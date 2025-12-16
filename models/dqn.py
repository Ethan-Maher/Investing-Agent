"""
Deep Q-Network (DQN) model for reinforcement learning.
CNN-based architecture for sequence processing.
"""

import torch
import torch.nn as nn


class CNN_DQN(nn.Module):
    """
    CNN-based Deep Q-Network model for trading agent.
    Processes sequences using 1D convolutions.
    
    Args:
        window_size: Length of the sequence window
        num_features: Number of features per time step
        output_dim: Number of actions (7: Hold, Buy 25%, Buy 50%, Buy 100%, Sell 25%, Sell 50%, Sell 100%)
        hidden_dim: Dimension of hidden layers in FC network (default: 128)
    """
    
    def __init__(self, window_size, num_features, output_dim, hidden_dim=64):
        super().__init__()
        
        # Input shape: (batch, channels=features, seq_len=window)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(),
        )
        
        # Compute size after conv layers:
        # Each conv with kernel_size=3 reduces sequence length by 2
        conv_output_size = window_size - 2 - 2 - 2  # kernel_size=3 reduces 2 each conv
        linear_input_dim = 32 * conv_output_size
        
        self.fc = nn.Sequential(
            nn.Linear(linear_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, window, features)
            
        Returns:
            Q-values for each action, shape (batch, output_dim)
        """
        # x shape: (batch, window, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, window)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


def create_cnn_dqn(window_size, num_features, output_dim=7, hidden_dim=64):
    """
    Helper function to create a CNN-based DQN model.
    
    Args:
        window_size: Length of the sequence window
        num_features: Number of features per time step
        output_dim: Number of actions (default: 7)
        hidden_dim: Dimension of hidden layers in FC network (default: 128)
        
    Returns:
        CNN_DQN model instance
    """
    return CNN_DQN(window_size, num_features, output_dim, hidden_dim)


# Keep old DQN for backward compatibility (if needed)
class DQN(nn.Module):
    """
    Legacy MLP-based DQN (kept for backward compatibility).
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def create_dqn(input_dim, output_dim, hidden_dim=256):
    """
    Legacy helper function (kept for backward compatibility).
    """
    return DQN(input_dim, hidden_dim, output_dim)
