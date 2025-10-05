"""
Neural Network Architectures for FSRPPO

Implementation of Actor-Critic networks according to Wang & Wang (2024) paper.
The paper specifies:
- Separate policy and value networks (not shared parameters)
- Two hidden layers with 256 neurons each
- Tanh activation function
- Adam optimizer with learning rate 0.00001

References:
- Wang & Wang (2024): FSRPPO paper Section 2.4.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ActorNetwork(nn.Module):
    """
    Policy Network (Actor) for FSRPPO
    
    Architecture according to paper:
    - Input: State features (after FSR processing)
    - Hidden: 2 layers × 256 neurons with Tanh activation
    - Output: Action probabilities for continuous action space [0,1]×[0,1]
    
    Parameters:
    -----------
    state_dim : int
        Dimension of state space (50 for price history)
    action_dim : int
        Dimension of action space (2: direction + amount)
    hidden_dim : int, default=256
        Number of neurons in hidden layers
    """
    
    def __init__(self, state_dim: int, action_dim: int = 2, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Network layers according to paper specifications
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network
        
        Parameters:
        -----------
        state : torch.Tensor
            State tensor of shape (batch_size, state_dim)
            
        Returns:
        --------
        torch.Tensor
            Action probabilities of shape (batch_size, action_dim)
            Values are in [0, 1] range using sigmoid activation
        """
        # Hidden layers with Tanh activation (as specified in paper)
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        
        # Output layer with sigmoid to ensure [0,1] range
        actions = torch.sigmoid(self.fc_out(x))
        
        return actions
    
    def get_action_and_log_prob(self, state: torch.Tensor, 
                               deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action and its log probability for PPO training
        
        Parameters:
        -----------
        state : torch.Tensor
            State tensor
        deterministic : bool, default=False
            Whether to use deterministic policy (for evaluation)
            
        Returns:
        --------
        tuple
            (action, log_probability)
        """
        # Get action probabilities
        action_probs = self.forward(state)
        
        if deterministic:
            # Use mean action for evaluation
            action = action_probs
            # For deterministic actions, log prob is not meaningful
            log_prob = torch.zeros(action_probs.shape[0], device=action_probs.device)
        else:
            # Sample from Beta distribution for continuous actions in [0,1]
            # Beta distribution is appropriate for bounded continuous actions
            alpha = action_probs * 2 + 1  # Ensure alpha > 1
            beta = (1 - action_probs) * 2 + 1  # Ensure beta > 1
            
            dist = torch.distributions.Beta(alpha, beta)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, 
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO loss calculation
        
        Parameters:
        -----------
        state : torch.Tensor
            State tensor
        action : torch.Tensor
            Action tensor
            
        Returns:
        --------
        tuple
            (log_probability, entropy)
        """
        action_probs = self.forward(state)
        
        # Create Beta distribution
        alpha = action_probs * 2 + 1
        beta = (1 - action_probs) * 2 + 1
        dist = torch.distributions.Beta(alpha, beta)
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Value Network (Critic) for FSRPPO
    
    Architecture according to paper:
    - Input: State features (after FSR processing)
    - Hidden: 2 layers × 256 neurons with Tanh activation
    - Output: State value estimate
    
    Parameters:
    -----------
    state_dim : int
        Dimension of state space (50 for price history)
    hidden_dim : int, default=256
        Number of neurons in hidden layers
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Network layers according to paper specifications
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in [self.fc1, self.fc2, self.fc_value]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network
        
        Parameters:
        -----------
        state : torch.Tensor
            State tensor of shape (batch_size, state_dim)
            
        Returns:
        --------
        torch.Tensor
            State value estimates of shape (batch_size, 1)
        """
        # Hidden layers with Tanh activation (as specified in paper)
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        
        # Output layer for value estimation
        value = self.fc_value(x)
        
        return value


class FSRFeatureExtractor(nn.Module):
    """
    Feature extractor that applies FSR processing to raw price data
    
    This module integrates the Financial Signal Representation (FSR) technique
    into the neural network pipeline for end-to-end training.
    
    Parameters:
    -----------
    lookback_window : int, default=50
        Number of historical price points to use
    fsr_processor : FinancialSignalRepresentation
        FSR processor for signal cleaning
    """
    
    def __init__(self, lookback_window: int = 50, fsr_processor=None):
        super(FSRFeatureExtractor, self).__init__()
        
        self.lookback_window = lookback_window
        self.fsr_processor = fsr_processor
        
        # Feature normalization layer
        self.feature_norm = nn.LayerNorm(lookback_window)
    
    def forward(self, raw_prices: torch.Tensor) -> torch.Tensor:
        """
        Extract FSR features from raw price data
        
        Parameters:
        -----------
        raw_prices : torch.Tensor
            Raw price data of shape (batch_size, lookback_window)
            
        Returns:
        --------
        torch.Tensor
            FSR-processed features of shape (batch_size, lookback_window)
        """
        batch_size = raw_prices.shape[0]
        processed_features = []
        
        for i in range(batch_size):
            # Convert to numpy for FSR processing
            price_series = raw_prices[i].detach().cpu().numpy()
            
            if self.fsr_processor is not None and len(price_series) >= 50:
                try:
                    # Apply FSR processing
                    clean_signal = self.fsr_processor.extract_representation(price_series)
                    processed_features.append(torch.tensor(clean_signal, dtype=torch.float32))
                except Exception:
                    # Fallback to original signal if FSR fails
                    processed_features.append(raw_prices[i])
            else:
                # Use original signal if FSR not available
                processed_features.append(raw_prices[i])
        
        # Stack processed features
        processed_tensor = torch.stack(processed_features).to(raw_prices.device)
        
        # Apply normalization
        normalized_features = self.feature_norm(processed_tensor)
        
        return normalized_features


# Example usage and testing
if __name__ == "__main__":
    import torch
    
    # Test network architectures
    print("Testing FSRPPO Neural Networks")
    
    # Parameters from paper
    state_dim = 50  # 50-day price history
    action_dim = 2  # [direction, amount]
    batch_size = 32
    
    # Create networks
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim)
    
    print(f"Actor Network Parameters: {sum(p.numel() for p in actor.parameters())}")
    print(f"Critic Network Parameters: {sum(p.numel() for p in critic.parameters())}")
    
    # Test forward pass
    dummy_state = torch.randn(batch_size, state_dim)
    
    # Actor forward pass
    actions = actor(dummy_state)
    print(f"Actor output shape: {actions.shape}")
    print(f"Action range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    
    # Critic forward pass
    values = critic(dummy_state)
    print(f"Critic output shape: {values.shape}")
    
    # Test action sampling
    sampled_actions, log_probs = actor.get_action_and_log_prob(dummy_state)
    print(f"Sampled actions shape: {sampled_actions.shape}")
    print(f"Log probabilities shape: {log_probs.shape}")
    
    # Test action evaluation
    eval_log_probs, entropy = actor.evaluate_actions(dummy_state, sampled_actions)
    print(f"Evaluated log probs shape: {eval_log_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    print("\nNetwork architecture tests completed successfully!")