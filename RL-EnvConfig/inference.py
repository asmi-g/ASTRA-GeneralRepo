import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from astra_rev1.envs import NoiseReductionEnv
import os

# --- Load signal data ---
df = pd.read_csv("Data/simulated_signal_data.csv")

# Load trained SAC model
model_path = os.path.join("models", "sac_noise_reduction")
model = SAC.load(model_path)

# Initialize environment
env = NoiseReductionEnv()

# Sliding window parameters
window_size = 10
current_window_clean = df.iloc[:window_size]["Clean Signal"].tolist()
current_window_noisy = df.iloc[:window_size]["Noisy Signal"].tolist()
time = df.iloc[:window_size]["Time"].tolist()

# Reset environment with initial state
state, _ = env.reset(clean_signal=np.array(current_window_clean),
                     noisy_signal=np.array(current_window_noisy))

# --- Tracking metrics ---
actions = []
rewards = []
snr_raw_list = []
snr_filtered_list = []
clean_signal_data = []
noisy_signal_data = []
filtered_signal_data = []
thresholds = []
mse = []

print("Running inference using trained SAC model (continuous actions)...")
for i in range(window_size, len(df)):  
    # Predict continuous action
    action, _ = model.predict(state, deterministic=True)
    actions.append(action[0])  # Extract scalar from array

    # Step the environment
    next_state, reward, done, truncated, info = env.step(action)

    # Extract useful info
    snr_raw = info["SNR_raw"]
    snr_filtered = info["SNR_filtered"]
    filtered_signal = info["filtered_signal"]
    t_factor = info["threshold_factor"]

    rewards.append(reward)
    thresholds.append(t_factor)
    snr_raw_list.append(snr_raw)
    snr_filtered_list.append(snr_filtered)
    mse.append(np.square(snr_filtered - snr_raw))

    # Store signals
    clean_signal_data.extend(info["clean_signal"])
    noisy_signal_data.extend(info["noisy_signal"])
    filtered_signal_data.extend(filtered_signal)
    mse.append(np.square(np.subtract(snr_filtered, snr_raw)).mean())

    print(f"Rows {i-window_size, i} | Action: {action} | Reward: {reward:.4f} | SNR Raw: {snr_raw:.2f} | SNR Filtered: {snr_filtered:.2f} | Done: {done} | filtered signal: {np.mean(filtered_signal):.4f} | clean signal: {np.mean(current_window_clean):.4f} | threshold factor: {t_factor:.4f}")

    # Update sliding window
    current_window_clean.pop(0)
    current_window_noisy.pop(0)
    current_window_clean.append(df.iloc[i]["Clean Signal"])
    current_window_noisy.append(df.iloc[i]["Noisy Signal"])
    time.pop(0)
    time.append(df.iloc[i]["Time"])
    env.set_signal_window(np.array(current_window_clean), np.array(current_window_noisy))

    # Update state
    state = next_state

    if done:
        print(f"Early termination at index {i}")
        state = env.reset(clean_signal=np.array(current_window_clean),
                          noisy_signal=np.array(current_window_noisy))
        break

env.close()

# --- Visualization ---
plt.figure(figsize=(12, 6))

# Signal comparison
plt.subplot(3, 1, 1)
plt.plot(clean_signal_data, label="Clean Signal", color="blue", alpha=0.8)
plt.plot(noisy_signal_data, label="Noisy Signal", color="orange", alpha=0.5)
plt.plot(filtered_signal_data, label="Filtered Signal", color="green", alpha=0.8)
plt.title("Clean vs. Noisy vs. Filtered Signal with SAC")
plt.ylabel("Signal Amplitude")
plt.legend()

# Threshold factor evolution
plt.subplot(3, 1, 2)
plt.plot(thresholds, label="Threshold Factor", color="purple")
plt.ylabel("Threshold")
plt.title("Threshold Adjustment Over Time")
plt.legend()

# MSE evolution
plt.subplot(3, 1, 3)
plt.plot(mse, label="MSE (SNR Filtered vs. Raw)", color="blue")
plt.xlabel("Time Step")
plt.ylabel("MSE")
plt.title("MSE Over Time")
plt.legend()

plt.tight_layout()
plt.show()
