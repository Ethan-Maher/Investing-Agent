"""
Replay buffer for experience replay in DQN training.
"""

import numpy as np
import random


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state (numpy array, shape: (window, features) or flattened)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state (numpy array, shape: (window, features) or flattened)
            done: Whether episode is done (bool)
        """
        # state and next_state are numpy arrays (2D or 1D)
        data = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
            states and next_states will have shape (batch_size, window, features) for 2D states
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
