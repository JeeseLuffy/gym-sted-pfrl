"""
Gym Wrapper that applies uncertainty-based reward penalty.

This wrapper intercepts env.step() and adjusts the reward based on
the current policy's MC Dropout uncertainty estimate.

Design decisions:
- Uncertainty is computed on s_t (current obs BEFORE stepping), not s_{t+1}
- Policy train/eval state is saved and restored
- P10-P90 quantile normalization for robust penalty scaling
- penalty/reward ratio is tracked for λ calibration

Usage in main.py:
    from src.uncertainty_wrapper import UncertaintyRewardWrapper
    ...
    env = UncertaintyRewardWrapper(env, policy, lambda_unc=0.1)
"""

import gym
import numpy as np
import torch

from src.uncertainty import estimate_uncertainty, UncertaintyTracker


class UncertaintyRewardWrapper(gym.Wrapper):
    """
    Wraps a gym environment to penalize high-uncertainty aggressive actions.
    
    r_adjusted = r_original - λ * normalize(uncertainty) * normalize(action)
    
    The policy model is passed by reference so the wrapper always uses
    the current (evolving) policy weights during training.
    
    Args:
        env: The gym environment to wrap
        policy: The Policy neural network (must have Dropout layers)
        lambda_unc: Penalty weight (calibrate so penalty/reward ratio = 5-20%)
        n_samples: Number of MC Dropout forward passes
        tracker_window: Size of the uncertainty tracker sliding window
    """
    
    def __init__(self, env, policy, lambda_unc=0.0, n_samples=20,
                 tracker_window=1000):
        super().__init__(env)
        self.policy = policy
        self.lambda_unc = lambda_unc
        self.n_samples = n_samples
        self.tracker = UncertaintyTracker(window=tracker_window)
        
        # For logging penalty/reward ratios
        self.last_penalty = 0.0
        self.last_raw_reward = 0.0
        self.last_ratio = 0.0
        self.last_uncertainty = 0.0
        
        # Store current obs for uncertainty computation on s_t
        self._current_obs = None
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._current_obs = obs
        return obs
    
    def step(self, action):
        # Compute uncertainty on s_t (BEFORE env.step)
        uncertainty = 0.0
        if self.lambda_unc > 0 and self._current_obs is not None:
            uncertainty = self._compute_uncertainty(self._current_obs)
        
        # Original step
        obs, reward, done, info = self.env.step(action)
        self._current_obs = obs
        
        # Apply penalty
        self.last_raw_reward = float(reward)
        if self.lambda_unc > 0 and uncertainty > 0:
            self.tracker.update(uncertainty)
            unc_norm = self.tracker.normalize(uncertainty)
            
            # Normalize action to [0, 1]
            act_space = self.env.action_space
            if hasattr(act_space, 'low') and hasattr(act_space, 'high'):
                act_range = act_space.high - act_space.low
                act_norm = np.where(
                    act_range > 1e-10,
                    (np.array(action) - act_space.low) / act_range,
                    0.5
                ).mean()
            else:
                act_norm = 0.5
            
            penalty = self.lambda_unc * unc_norm * act_norm
            self.last_penalty = float(penalty)
            self.last_ratio = float(penalty / (abs(reward) + 1e-8))
            self.last_uncertainty = float(uncertainty)
            
            reward = reward - penalty
        else:
            self.last_penalty = 0.0
            self.last_ratio = 0.0
            self.last_uncertainty = float(uncertainty)
        
        # Add uncertainty info to info dict
        info['unc_penalty'] = self.last_penalty
        info['unc_raw_reward'] = self.last_raw_reward
        info['unc_ratio'] = self.last_ratio
        info['unc_value'] = self.last_uncertainty
        
        return obs, reward, done, info
    
    def _compute_uncertainty(self, obs):
        """Compute MC Dropout uncertainty on current observation."""
        try:
            # Convert observation to tensor
            if isinstance(obs, tuple):
                # Tuple obs: (image, signal)
                image, signal = obs
                image_t = torch.from_numpy(
                    np.array(image, dtype=np.float32)
                ).unsqueeze(0)
                signal_t = torch.from_numpy(
                    np.array(signal, dtype=np.float32)
                ).unsqueeze(0)
                obs_tensor = (image_t, signal_t)
            else:
                obs_tensor = torch.from_numpy(
                    np.array(obs, dtype=np.float32)
                ).unsqueeze(0)
            
            _, std = estimate_uncertainty(
                self.policy, obs_tensor, n_samples=self.n_samples
            )
            return float(std.mean())
        except Exception:
            # If uncertainty computation fails, return 0 (no penalty)
            return 0.0
