# inference_noise_reduction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from astra_rev1.envs import NoiseReductionEnv
import os

# Load signal data
df = pd.read_csv("Data/signal.csv").head(5000).rename(columns={'TX Magnitude': 'Noisy Signal', 'RX Magnitude': 'Clean Signal'})

#df = pd.read_csv("C:/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/Data/simulated_signal_data.csv").head(5000).rename(columns={"TX Magnitude": "Noisy Signal", "RX Magnitude": "Clean Signal"})

custom_objects = {
    "lr_schedule": lambda x: 0.003,
    "clip_range": lambda x: 0.02
}

# Load trained SAC model
model_path = os.path.join("models", "sac_noise_reduction_071225_400")
model = SAC.load(model_path, custom_objects=custom_objects)

# Initialize environment
env = NoiseReductionEnv()

# Sliding window parameters
window_size = 10
current_window_clean = df.iloc[:window_size]["Clean Signal"].tolist()
current_window_noisy = df.iloc[:window_size]["Noisy Signal"].tolist()

# Reset environment
state = env.reset(
    clean_signal=np.array(current_window_clean, dtype=np.float32),
    noisy_signal=np.array(current_window_noisy, dtype=np.float32)
)

# Tracking metrics
actions = []
rewards = []
snr_raw_list = []
snr_filtered_list = []
snr_improvement = []
clean_signal_data = []
noisy_signal_data = []
filtered_signal_data = []
thresholds = []
mse = []

print("Running inference using trained SAC model (continuous actions)...")

for i in range(window_size, len(df)):
    # Predict continuous action
    action, _ = model.predict(state, deterministic=True)
    actions.append(action[0])

    # Step the environment
    next_state, reward, done, info = env.step(action)

    # Extract metrics
    snr_raw = info["SNR_raw"]
    snr_filtered = info["SNR_filtered"]
    filtered_signal = info["filtered_signal"]
    t_factor = info["threshold_factor"]

    rewards.append(reward)
    thresholds.append(t_factor)
    snr_raw_list.append(snr_raw)
    snr_filtered_list.append(snr_filtered)
    snr_improvement.append(snr_filtered - snr_raw)

    # MSE between filtered and clean signals
    mse_value = np.mean((filtered_signal - info["clean_signal"]) ** 2)
    mse.append(mse_value)

    # Store signals
    clean_signal_data.extend(info["clean_signal"])
    noisy_signal_data.extend(info["noisy_signal"])
    filtered_signal_data.extend(filtered_signal)

    print(
        "Rows {} | Action: {:.4f} | Reward: {:.4f} | SNR Improvement: {:.2f} | SNR Raw: {:.2f} | SNR Filtered: {:.2f} | Done: {} | Threshold Factor: {:.4f}".format(
            (i - window_size, i),
            action[0],
            reward,
            snr_improvement[-1],
            snr_raw,
            snr_filtered,
            done,
            t_factor
        )
    )

    # Update sliding window
    current_window_clean.pop(0)
    current_window_noisy.pop(0)
    current_window_clean.append(df.iloc[i]["Clean Signal"])
    current_window_noisy.append(df.iloc[i]["Noisy Signal"])
    env.set_signal_window(np.array(current_window_clean), np.array(current_window_noisy))

    # Update state
    state = next_state

    if done:
        print("Early termination at index {}".format(i))
        state = env.reset(
            clean_signal=np.array(current_window_clean, dtype=np.float32),
            noisy_signal=np.array(current_window_noisy, dtype=np.float32)
        )

        break

env.close()
print("done inference")

# Visualization
# plt.figure(figsize=(12, 6))

# plt.subplot(3, 1, 1)
# plt.plot(clean_signal_data, label="Clean Signal", color="blue", alpha=0.8)
# plt.plot(noisy_signal_data, label="Noisy Signal", color="orange", alpha=0.5)
# plt.plot(filtered_signal_data, label="Filtered Signal", color="green", alpha=0.8)
# plt.title("Clean vs. Noisy vs. Filtered Signal with SAC")
# plt.ylabel("Signal Amplitude")
# plt.legend()

# snr_improvement = np.array(snr_improvement)
# x = np.arange(len(snr_improvement))
# coeffs = np.polyfit(x, snr_improvement, deg=1)
# trendline = np.polyval(coeffs, x)

# plt.subplot(3, 1, 2)
# plt.plot(snr_improvement, label="SNR Improvement", color="red")
# plt.plot(x, trendline, label="Trendline", color="blue", linestyle="--")
# plt.axhline(y=0, color="black", linestyle="--")
# plt.ylabel("SNR Improvement")
# plt.title("SNR Improvement Over Time")
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(mse, label="MSE (Filtered vs. Clean)", color="blue")
# plt.xlabel("Time Step")
# plt.ylabel("MSE")
# plt.title("MSE Over Time")
# plt.legend()

# plt.tight_layout()
# plt.show()
