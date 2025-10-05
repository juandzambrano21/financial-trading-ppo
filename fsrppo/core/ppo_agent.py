"""
Robust PPO Agent for Financial Trading

Complete implementation of Proximal Policy Optimization (PPO) agent
specifically designed for financial trading with FSRPPO methodology.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import pickle
from pathlib import Path

from .networks import ActorNetwork, CriticNetwork

logger = logging.getLogger(__name__)


class PPOAgent:
    """
    Robust PPO agent for financial trading
    
    Implements the PPO algorithm from the FSRPPO paper with:
    - Actor-Critic architecture
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Entropy regularization
    - Experience replay buffer
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 1.0,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 device: Optional[str] = None):
        """
        Initialize PPO agent
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state space
        action_dim : int
            Dimension of action space
        lr : float
            Learning rate
        gamma : float
            Discount factor
        gae_lambda : float
            GAE lambda parameter
        clip_epsilon : float
            PPO clipping parameter
        entropy_coef : float
            Entropy regularization coefficient
        value_coef : float
            Value function loss coefficient
        max_grad_norm : float
            Maximum gradient norm for clipping
        n_epochs : int
            Number of optimization epochs per update
        batch_size : int
            Batch size for training
        buffer_size : int
            Size of experience replay buffer
        device : Optional[str]
            Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = ExperienceBuffer(buffer_size)
        
        # Training statistics
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100),
            'explained_variance': deque(maxlen=100)
        }
        
        # Training step counter
        self.training_step = 0
        
        logger.info(f"PPOAgent initialized on device: {self.device}")
        logger.info(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        logger.info(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Get action from current policy
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        deterministic : bool
            Whether to use deterministic policy (for evaluation)
            
        Returns:
        --------
        tuple
            (action, log_probability)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                # Use mean action for deterministic policy
                action_probs = self.actor(state_tensor)
                action = action_probs.cpu().numpy()[0]
                log_prob = 0.0  # Not used in deterministic mode
            else:
                # Sample from policy distribution
                action, log_prob = self.actor.get_action_and_log_prob(state_tensor)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]
        
        return action, log_prob
    
    def store_experience(self, 
                        state: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool,
                        log_prob: float):
        """Store experience in buffer"""
        self.buffer.add(state, action, reward, next_state, done, log_prob)
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO algorithm
        
        Returns:
        --------
        Dict[str, float]
            Training statistics
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Get batch from buffer
        batch = self.buffer.get_batch(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.BoolTensor(batch['dones']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        
        # Calculate advantages using GAE
        advantages, returns = self._calculate_gae(states, rewards, next_states, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store old policy values for KL divergence
        with torch.no_grad():
            old_values = self.critic(states).squeeze()
        
        # PPO update loop
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl_div = 0
        
        for epoch in range(self.n_epochs):
            # Get current policy values
            current_log_probs, entropy = self._get_log_probs_and_entropy(states, actions)
            current_values = self.critic(states).squeeze()
            
            # Calculate ratios
            ratios = torch.exp(current_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            value_loss = nn.MSELoss()(current_values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total losses
            total_loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_total_loss = actor_loss + self.entropy_coef * entropy_loss
            actor_total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # Accumulate statistics
            total_actor_loss += actor_loss.item()
            total_critic_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            
            # Calculate KL divergence for early stopping
            with torch.no_grad():
                kl_div = (old_log_probs - current_log_probs).mean().item()
                total_kl_div += abs(kl_div)
                
                # Early stopping if KL divergence is too high
                if abs(kl_div) > 0.01:  # KL threshold
                    logger.debug(f"Early stopping at epoch {epoch} due to high KL divergence: {kl_div}")
                    break
        
        # Calculate explained variance
        explained_var = self._explained_variance(old_values, returns)
        
        # Update training statistics
        avg_epochs = epoch + 1
        stats = {
            'actor_loss': total_actor_loss / avg_epochs,
            'critic_loss': total_critic_loss / avg_epochs,
            'entropy': total_entropy / avg_epochs,
            'kl_divergence': total_kl_div / avg_epochs,
            'explained_variance': explained_var,
            'epochs_used': avg_epochs
        }
        
        # Store statistics
        for key, value in stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        self.training_step += 1
        
        return stats
    
    def _calculate_gae(self, 
                      states: torch.Tensor,
                      rewards: torch.Tensor,
                      next_states: torch.Tensor,
                      dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate Generalized Advantage Estimation (GAE)
        
        Parameters:
        -----------
        states : torch.Tensor
            Current states
        rewards : torch.Tensor
            Rewards
        next_states : torch.Tensor
            Next states
        dones : torch.Tensor
            Done flags
            
        Returns:
        --------
        tuple
            (advantages, returns)
        """
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            
            # Calculate TD errors
            td_targets = rewards + self.gamma * next_values * (~dones)
            td_errors = td_targets - values
            
            # Calculate GAE
            advantages = torch.zeros_like(rewards)
            gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = ~dones[t]
                    next_value = next_values[t]
                else:
                    next_non_terminal = ~dones[t]
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[t] = gae
            
            returns = advantages + values
        
        return advantages, returns
    
    def _get_log_probs_and_entropy(self, 
                                  states: torch.Tensor,
                                  actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and entropy for given states and actions
        
        Parameters:
        -----------
        states : torch.Tensor
            States
        actions : torch.Tensor
            Actions
            
        Returns:
        --------
        tuple
            (log_probabilities, entropy)
        """
        action_probs = self.actor(states)
        
        # Create Beta distribution for continuous actions in [0, 1]
        # Transform actions from [-1, 1] to [0, 1]
        actions_01 = (actions + 1) / 2
        action_probs_01 = (action_probs + 1) / 2
        
        # Use Beta distribution parameters
        alpha = action_probs_01 * 2 + 1  # Ensure alpha > 1
        beta = (1 - action_probs_01) * 2 + 1  # Ensure beta > 1
        
        dist = torch.distributions.Beta(alpha, beta)
        
        # Calculate log probabilities
        log_probs = dist.log_prob(actions_01).sum(dim=-1)
        
        # Calculate entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy
    
    def _explained_variance(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Calculate explained variance"""
        var_y = torch.var(y_true)
        return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
    
    def save(self, filepath: str):
        """Save agent state"""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'lr': self.lr,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'max_grad_norm': self.max_grad_norm,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'buffer_size': self.buffer_size
            }
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        save_dict = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(save_dict['actor_state_dict'])
        self.critic.load_state_dict(save_dict['critic_state_dict'])
        self.actor_optimizer.load_state_dict(save_dict['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(save_dict['critic_optimizer_state_dict'])
        self.training_step = save_dict['training_step']
        
        logger.info(f"Agent loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get recent training statistics"""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
        
        return stats
    
    def set_eval_mode(self):
        """Set networks to evaluation mode"""
        self.actor.eval()
        self.critic.eval()
    
    def set_train_mode(self):
        """Set networks to training mode"""
        self.actor.train()
        self.critic.train()


class ExperienceBuffer:
    """Experience replay buffer for PPO"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, 
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            log_prob: float):
        """Add experience to buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        }
        self.buffer.append(experience)
    
    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Get random batch from buffer"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': []
        }
        
        for idx in indices:
            experience = self.buffer[idx]
            batch['states'].append(experience['state'])
            batch['actions'].append(experience['action'])
            batch['rewards'].append(experience['reward'])
            batch['next_states'].append(experience['next_state'])
            batch['dones'].append(experience['done'])
            batch['log_probs'].append(experience['log_prob'])
        
        # Convert to numpy arrays
        for key in batch:
            batch[key] = np.array(batch[key])
        
        return batch
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)