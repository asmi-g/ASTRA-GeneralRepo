# not tested yet

import gym
from stable_baselines3 import DQN  # or any other algorithm (e.g., DQN, A2C)
import numpy as np
from astra_rev1.envs import NoiseReductionEnv

# Create the environment
env = gym.make("NoiseReductionEnv-v0")

# Instantiate the model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Test the trained model
state = env.reset()
for i in range(10):
    action, _states = model.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    print(f"Step {i+1} | Action: {action} | Reward: {reward:.4f} | Done: {done}")
    if done:
        state = env.reset()

env.close()