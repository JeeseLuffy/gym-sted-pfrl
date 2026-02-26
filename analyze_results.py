"""
Analysis script for Route A experiments.

Usage:
    python analyze_results.py --data-dir ./data

Features:
1. Plot learning curves (Eval Return, Eval Bleach, Eval F1) across 4 groups × 5 seeds
2. Run statistical tests (KS, Mann-Whitney U) on action distributions partitioned by uncertainty
3. Generate spatial CNN feature map uncertainty heatmaps
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, ks_2samp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import pfrl
import gym
import gym_sted

from src import models, WrapPyTorch, GymnasiumWrapper

def load_tb_data(logdir, tag):
    """Load scalar data from a TensorBoard log directory."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    return None, None

def plot_learning_curves(data_dir):
    """Plot learning curves for Return, F1, and Bleach across the 4 groups."""
    groups = {
        'A_random': 'Random',
        'B_baseline': 'PPO Baseline',
        'C_dropout': 'PPO+Dropout ($\lambda=0$)',
        'D_unc_penalty': 'PPO+Unc Penalty'
    }
    metrics = {
        'eval/mean_reward': 'Evaluation Return',
        'eval/mean_bleach': 'Evaluation Bleach Ratio',
        'eval/mean_f1': 'Evaluation F1 Score'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (tag, ylabel) in enumerate(metrics.items()):
        ax = axes[i]
        for prefix, label in groups.items():
            # Find all seeds for this group
            seed_dirs = glob.glob(os.path.join(data_dir, f"{prefix}_s*"))
            if not seed_dirs:
                continue
                
            all_values = []
            steps = None
            for s_dir in seed_dirs:
                s, v = load_tb_data(s_dir, tag)
                if v is not None:
                    steps = s
                    # Interpolate to common steps if necessary, but PFRL aligns eval steps
                    all_values.append(v)
            
            if all_values and len(all_values) > 0:
                # Pad sequences if they differ in length (happens if some runs stopped early)
                min_len = min(len(v) for v in all_values)
                all_values = np.array([v[:min_len] for v in all_values])
                steps = steps[:min_len]
                
                mean_val = all_values.mean(axis=0)
                std_val = all_values.std(axis=0)
                
                ax.plot(steps, mean_val, label=label)
                ax.fill_between(steps, mean_val - std_val, mean_val + std_val, alpha=0.2)
        
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    print("Saved learning_curves.png")

def verify_core_hypothesis(model_path, env_id="ContextualMOSTED-easy-hslb-v0"):
    """
    Test 1000 steps to collect (action, uncertainty) pairs.
    Run non-parametric tests to see if agent's action magnitude 
    differs significantly in high vs low uncertainty states.
    """
    from src.uncertainty import estimate_uncertainty
    
    print(f"\n--- Verifying Core Hypothesis ---")
    print(f"Loading model from {model_path}...")
    
    env = gym.make(env_id, disable_env_checker=True)
    env = pfrl.wrappers.NormalizeActionSpace(env)
    env = WrapPyTorch(env)
    env = GymnasiumWrapper(env)
    
    obs_space = env.observation_space
    action_space = env.action_space
    
    policy = models.Policy(obs_space=obs_space, action_size=action_space.shape[0])
    vf = models.ValueFunction(obs_space=obs_space)
    model = pfrl.nn.Branched(policy, vf)
    
    try:
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu'))
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Train the D group first.")
        return
        
    actions = []
    uncertainties = []
    
    obs, _ = env.reset()
    print("Collecting 1000 steps of data...")
    for _ in range(1000):
        # Convert obs to tensor
        obs_tensor = pfrl.utils.batch_states([obs], torch.device("cpu"), lambda x: x)
        _, std = estimate_uncertainty(policy, obs_tensor, n_samples=20)
        unc = float(std.mean())
        
        # Take deterministic action for testing
        with torch.no_grad():
            action_mean = policy(obs_tensor).mean.cpu().numpy()[0]
        
        # Log action magnitude (using p_sted specifically as it's the primary bleaching factor)
        # Action space is usually (p_sted, p_ex, pdt)
        p_sted = action_mean[0] 
        
        actions.append(p_sted)
        uncertainties.append(unc)
        
        obs, reward, done, trunc, info = env.step(action_mean)
        if done or trunc:
            obs, _ = env.reset()
            
    actions = np.array(actions)
    uncertainties = np.array(uncertainties)
    median_unc = np.median(uncertainties)
    
    high_unc_actions = actions[uncertainties > median_unc]
    low_unc_actions  = actions[uncertainties <= median_unc]
    
    print(f"\nResults for P_STED actions:")
    print(f"  High uncertainty (N={len(high_unc_actions)}): mean={high_unc_actions.mean():.4f}, std={high_unc_actions.std():.4f}")
    print(f"  Low uncertainty  (N={len(low_unc_actions)}): mean={low_unc_actions.mean():.4f}, std={low_unc_actions.std():.4f}")
    
    # Statistical tests
    try:
        ks_stat, p_ks = ks_2samp(high_unc_actions, low_unc_actions)
        mw_stat, p_mw = mannwhitneyu(high_unc_actions, low_unc_actions, alternative='two-sided')
        
        print("\nStatistical Significance Tests:")
        print(f"  Kolmogorov-Smirnov: statistic={ks_stat:.4f}, p-value={p_ks:.4e}")
        print(f"  Mann-Whitney U:     statistic={mw_stat:.4f}, p-value={p_mw:.4e}")
        
        if p_ks < 0.05 and p_mw < 0.05:
            print("  ✅ SIGNIFICANT DIFFERENCE DETECTED (p < 0.05)")
            print("  Agent learned to differentiate behavior based on uncertainty!")
        else:
            print("  ❌ NO SIGNIFICANT DIFFERENCE (p > 0.05)")
            print("  Agent treats high/low uncertainty states similarly.")
            
    except Exception as e:
        print(f"Stats test failed: {e}")
        
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(high_unc_actions, bins=30, alpha=0.5, density=True, label='High Uncertainty')
    plt.hist(low_unc_actions, bins=30, alpha=0.5, density=True, label='Low Uncertainty')
    plt.xlabel('P_STED Action Magnitude')
    plt.ylabel('Density')
    plt.title('Action Distribution Partitioned by Uncertainty')
    plt.legend()
    plt.savefig('action_distribution.png', dpi=300)
    print("Saved action_distribution.png")

def generate_spatial_heatmap(model_path, env_id="ContextualMOSTED-easy-hslb-v0"):
    """Generate spatial uncertainty heatmap from CNN feature maps (Fig 5)."""
    from src.uncertainty import extract_feature_map_uncertainty
    import matplotlib.cm as cm
    
    print(f"\n--- Generating Spatial Uncertainty Heatmap ---")
    
    env = gym.make(env_id, disable_env_checker=True)
    env = pfrl.wrappers.NormalizeActionSpace(env)
    env = WrapPyTorch(env)
    env = GymnasiumWrapper(env)
    
    obs_space = env.observation_space
    action_space = env.action_space
    
    policy = models.Policy(obs_space=obs_space, action_size=action_space.shape[0])
    vf = models.ValueFunction(obs_space=obs_space)
    model = pfrl.nn.Branched(policy, vf)
    
    try:
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu'))
    except FileNotFoundError:
        print(f"Model not found at {model_path}.")
        return
        
    obs, _ = env.reset()
    
    obs_tensor = pfrl.utils.batch_states([obs], torch.device("cpu"), lambda x: x)
    
    # Get heatmap (64, 64)
    heatmap = extract_feature_map_uncertainty(policy, obs_tensor, n_samples=20)
    
    # Plot original image and heatmap
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image (taking channel 0 representing confocal/fluorophores)
    base_img = image[..., 0] if image.ndim == 3 else image
    ax1.imshow(base_img, cmap='gray')
    ax1.set_title('Observation Image')
    ax1.axis('off')
    
    # Heatmap
    im2 = ax2.imshow(heatmap, cmap='inferno')
    ax2.set_title('CNN Feature Variance (N=20)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    ax3.imshow(base_img, cmap='gray')
    im3 = ax3.imshow(heatmap, cmap='inferno', alpha=0.5)
    ax3.set_title('Overlay (Uncertainty on Image)')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('spatial_heatmap.png', dpi=300)
    print("Saved spatial_heatmap.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--best-model-path", type=str, default="", help="Path to best D group model dir (e.g., ./data/D_unc_penalty_s0/best)")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        plot_learning_curves(args.data_dir)
        
    if args.best_model_path:
        verify_core_hypothesis(args.best_model_path)
        generate_spatial_heatmap(args.best_model_path)
    else:
        print("\nSkipping hypothesis testing and heatmap generation.")
        print("To run these, provide --best-model-path <path_to_agent_folder>")
