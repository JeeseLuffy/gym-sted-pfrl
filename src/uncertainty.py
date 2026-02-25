"""
MC Dropout uncertainty estimation + P10-P90 quantile normalization.

Usage in training loop:
    from src.uncertainty import estimate_uncertainty, UncertaintyTracker
    
    tracker = UncertaintyTracker(window=1000)
    ...
    _, unc_vec = estimate_uncertainty(policy, obs_tensor, n_samples=20)
    unc = unc_vec.mean()
    tracker.update(unc)
    unc_norm = tracker.normalize(unc)
"""

import numpy as np
import torch
from collections import deque


def estimate_uncertainty(model, obs, n_samples=20):
    """
    Run N forward passes with Dropout active on the same observation.
    Returns (mean_action, std_action).
    
    The model must have Dropout layers. This function saves and restores
    the model's training state to avoid side effects.
    
    Args:
        model: Policy network with Dropout layers
        obs: Observation tensor (same format as model.forward expects)
        n_samples: Number of MC samples (20 balances precision vs speed)
        
    Returns:
        mean_pred: Mean of N predictions (numpy array)
        std_pred: Std of N predictions (numpy array) — the uncertainty
    """
    was_training = model.training
    model.train()  # Activate Dropout
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(obs)
            # PFRL GaussianHead outputs a Distribution object with .mean
            if hasattr(out, 'mean'):
                predictions.append(out.mean.cpu().numpy())
            else:
                predictions.append(out.cpu().numpy())
    
    # Restore original state
    if not was_training:
        model.eval()
    
    predictions = np.stack(predictions)  # (N, batch, action_dim)
    return predictions.mean(axis=0), predictions.std(axis=0)


def extract_feature_map_uncertainty(model, obs, n_samples=20):
    """
    Extract spatial uncertainty from CNN encoder's last conv layer.
    Used for generating Fig 5 spatial heatmaps (NOT for training).
    
    Returns:
        heatmap: (H', W') variance map from CNN feature maps
    """
    import torch.nn.functional as F
    from skimage.transform import resize
    
    was_training = model.training
    model.train()
    
    feature_maps = []
    with torch.no_grad():
        for _ in range(n_samples):
            x = obs
            # Pass through recording queue encoder
            if hasattr(model, 'recording_queue_encoder_layers'):
                for layer in model.recording_queue_encoder_layers:
                    x = F.leaky_relu(layer(x))
            # Pass through image encoder
            for layer in model.image_encoder_layers:
                x = F.leaky_relu(layer(x))
            feature_maps.append(x.cpu().numpy())
    
    if not was_training:
        model.eval()
    
    # (N, batch, C, H', W') -> variance across N, mean across C
    fmaps = np.stack(feature_maps)
    spatial_var = fmaps.var(axis=0).mean(axis=1)  # (batch, H', W')
    
    # Resize to original image size (64x64) for overlay
    if spatial_var.ndim == 3:
        spatial_var = spatial_var[0]  # Take first batch item
    heatmap = resize(spatial_var, (64, 64), preserve_range=True)
    return heatmap


class UncertaintyTracker:
    """
    P10-P90 quantile normalization for uncertainty values.
    
    More robust than min-max normalization because it is resistant
    to outlier uncertainty values that would otherwise compress
    the normal range to near-zero.
    
    Args:
        window: Size of the sliding window buffer
    """
    
    def __init__(self, window=1000):
        self.buffer = deque(maxlen=window)
    
    def update(self, unc_value):
        """Add a new uncertainty value to the buffer."""
        self.buffer.append(float(unc_value))
    
    def normalize(self, unc_value):
        """
        Normalize uncertainty to [0, 1] using P10-P90 quantiles.
        
        During warmup (< 50 samples), returns 0.5 (neutral).
        Values below P10 clip to 0, above P90 clip to 1.
        """
        if len(self.buffer) < 50:
            return 0.5  # Warmup period
        p10 = np.percentile(self.buffer, 10)
        p90 = np.percentile(self.buffer, 90)
        return float(np.clip(
            (unc_value - p10) / (p90 - p10 + 1e-8), 0.0, 1.0
        ))
    
    def stats(self):
        """Return current buffer statistics for logging."""
        if len(self.buffer) < 2:
            return {"mean": 0, "std": 0, "p10": 0, "p90": 0, "n": 0}
        arr = np.array(self.buffer)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "n": len(self.buffer),
        }
