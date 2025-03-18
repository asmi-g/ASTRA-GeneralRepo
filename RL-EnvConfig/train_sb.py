import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from astra_rev1.envs import NoiseReductionEnv

# Create environment
env = NoiseReductionEnv()

# Wrap environment in vectorized wrapper (for compatibility)
vec_env = make_vec_env(lambda: env, n_envs=1)
#'''
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_noise_reduction")
#'''

# Load trained model (optional)
model = DQN.load("dqn_noise_reduction", env=env)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Run a test episode to visualize results
obs, _ = env.reset()
rewards = []
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    next_state, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    if done:
        break

# Plot reward progression
plt.plot(rewards)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("DQN Reward Progression in Noise Reduction")
plt.show()
