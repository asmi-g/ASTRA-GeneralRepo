# train_noise_reduction.py

import os
import numpy as np
import random
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from astra_rev1.envs import NoiseReductionEnv
from callbacks import NoiseReductionLogger
from stable_baselines3.common.noise import NormalActionNoise

# 🔄 CHANGED: Added action noise for exploration
action_noise = NormalActionNoise(mean=np.array([0.0]), sigma=np.array([0.1]))

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Create directories for logs and models
os.makedirs("logs/tensorboard", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 🔄 CHANGED: Now supports varied signal types and noise levels
def generate_synthetic_signal(signal_type=None, noise_level=None, length=100):
    t = np.linspace(0, 1, length)
    signal_type = signal_type or random.choice(["sine", "square", "sawtooth", "random"])
    noise_level = noise_level if noise_level is not None else np.random.uniform(0.2, 0.5)

    if signal_type == "sine":
        clean = np.sin(2 * np.pi * 5 * t)
    elif signal_type == "square":
        clean = np.sign(np.sin(2 * np.pi * 5 * t))
    elif signal_type == "sawtooth":
        clean = 2 * (t * 5 % 1) - 1
    elif signal_type == "random":
        clean = np.random.uniform(-1, 1, size=length)
    else:
        raise ValueError("Unknown signal type!")

    noisy = clean + np.random.normal(0, noise_level, size=length)
    return clean, noisy

# Initialize environment
env = NoiseReductionEnv()
check_env(env, warn=True)

# Initialize model
model = SAC(
    "MlpPolicy",
    env,
    ent_coef="auto",  # 🔄 CHANGED: Encourage exploration via entropy bonus
    action_noise=action_noise,  # 🔄 CHANGED: Added action noise
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=128,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=4,  # 🔄 CHANGED: More frequent updates per step
    verbose=1,
    tensorboard_log="logs/tensorboard"
)
from stable_baselines3.common.logger import configure

model._logger = configure(folder="logs/tensorboard", format_strings=["stdout", "tensorboard"])
model._current_progress_remaining = 1.0  # Full training progress at start


WINDOW_SIZE = 10
SIGNAL_LENGTH = 100
TOTAL_EPISODES = 5000

# Training Loop
for episode in range(TOTAL_EPISODES):
    clean_full, noisy_full = generate_synthetic_signal(length=SIGNAL_LENGTH)

    for i in range(SIGNAL_LENGTH - WINDOW_SIZE):
        clean_window = clean_full[i:i + WINDOW_SIZE]
        noisy_window = noisy_full[i:i + WINDOW_SIZE]

        if i == 0:
            obs, _ = env.reset(clean_signal=clean_window, noisy_signal=noisy_window)
        else:
            env.set_signal_window(clean_window, noisy_window)

        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, done, _, info = env.step(action)

        # Manually store transition
        model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
        model.train(batch_size=model.batch_size, gradient_steps=1)

        obs = next_obs

    if episode % 100 == 0:
        print(f"Step {episode} | Action: {action[0]:.4f} | Threshold: {env.threshold_factor:.3f} | Reward: {reward:.3f}") 


# Callback for custom logging
# callback = NoiseReductionLogger()
# TOTAL_TIMESTEPS = 1000
# model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

model.save("models/sac_noise_reduction")
print("Model saved to 'models/sac_noise_reduction'")

# # Training parameters
# WINDOW_SIZE = 10
# SIGNAL_LENGTH = 100
# TOTAL_EPISODES = 1000

# # 🔄 CHANGED: Use sliding window over new signal every episode
# for episode in range(TOTAL_EPISODES):
#     clean_full, noisy_full = generate_synthetic_signal(length=SIGNAL_LENGTH)
#     total_reward = 0

#     for i in range(SIGNAL_LENGTH - WINDOW_SIZE):
#         clean_window = clean_full[i:i + WINDOW_SIZE]
#         noisy_window = noisy_full[i:i + WINDOW_SIZE]

#         if i == 0:
#             obs, _ = env.reset(clean_signal=clean_window, noisy_signal=noisy_window)
#         else:
#             env.set_signal_window(clean_window, noisy_window)

#         action, _ = model.predict(obs)
#         obs, reward, done, _, info = env.step(action)
#         total_reward += reward

#     # 🔄 CHANGED: Better logging for debugging
#     if episode % 100 == 0:
#         print(f"Episode {episode} | Total Reward: {total_reward:.2f} "
#               f"| Threshold: {info['threshold_factor']:.3f} | Action: {action}")

# Save the model
