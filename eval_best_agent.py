import argparse
import numpy as np
import gym
import pfrl
import torch

from src import models, WrapPyTorch, GymnasiumWrapper

def evaluate(model_dir, env_id="ContextualMOSTED-easy-hslb-v0", n_episodes=5):
    env = gym.make(env_id, disable_env_checker=True)
    env = pfrl.wrappers.NormalizeActionSpace(env)
    env = WrapPyTorch(env)
    env = GymnasiumWrapper(env)
    
    policy = models.Policy(obs_space=env.observation_space, action_size=env.action_space.shape[0])
    vf = models.ValueFunction(obs_space=env.observation_space)
    
    agent = pfrl.agents.PPO(
        pfrl.nn.Branched(policy, vf),
        torch.optim.Adam(policy.parameters(), lr=1e-3),
        gpu=-1,
    )
    
    agent.load(model_dir)
    print(f"Loaded agent from {model_dir}")
    
    all_returns, all_f1, all_bleach = [], [], []
    
    for ep in range(n_episodes):
        obs = env.reset()
        ep_return = 0
        ep_steps = 0
        
        while True:
            with agent.eval_mode():
                action = agent.act(obs)
            # GymnasiumWrapper returns 4 values: obs, reward, done, info
            obs, reward, done, info = env.step(action)
            ep_return += reward
            ep_steps += 1
            if done:
                f1 = info.get('f1-score', 0)
                bleach = info.get('bleach', 0)
                all_returns.append(ep_return)
                all_f1.append(f1)
                all_bleach.append(bleach)
                print(f"Episode {ep+1}/{n_episodes}: Return={ep_return:.4f}, F1={f1:.4f}, Bleach={bleach:.4f}")
                break
                
    print("\n--- Baseline Evaluation Results ---")
    print(f"Episodes evaluated: {n_episodes}")
    print(f"Mean Return: {np.mean(all_returns):.4f} +/- {np.std(all_returns):.4f}")
    print(f"Mean F1-Score: {np.mean(all_f1):.4f} +/- {np.std(all_f1):.4f}")
    print(f"Mean Bleach:   {np.mean(all_bleach):.4f} +/- {np.std(all_bleach):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    evaluate(args.model_dir)
