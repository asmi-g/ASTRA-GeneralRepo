import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from astra_rev1.envs import NoiseReductionEnv

# Load dataset (first 10,000 rows)
df = pd.read_csv("Data/simulated_signal_data.csv", nrows=10000)

# Initialize environment
env = NoiseReductionEnv()

# Check if environment is compatible with Stable-Baselines3
check_env(env, warn=True)

# --- DQN Model Configuration ---
model = DQN(
    "MlpPolicy",  # Use a multi-layer perceptron policy
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    target_update_interval=500,
    train_freq=1,
    gradient_steps=1,
    verbose=1
)

# --- Training the Agent ---
TIMESTEPS = 10000  # Adjust based on compute power
print(f"Training DQN for {TIMESTEPS} timesteps...")
model.learn(total_timesteps=TIMESTEPS, log_interval=100)

# Save trained model
model.save("dqn_noise_reduction")
print("Model saved as 'dqn_noise_reduction'")

# Close environment
env.close()