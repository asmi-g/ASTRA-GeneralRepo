import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astra_rev1.envs import NoiseReductionEnv

# Load dataset (first 10,000 rows)
df = pd.read_csv("Data/simulated_signal_data.csv")

# Initialize environment
env = NoiseReductionEnv()

# Sliding window setup
window_size = 10  # Window size
current_window_clean = df.iloc[:window_size]["Clean Signal"].tolist()
current_window_noisy = df.iloc[:window_size]["Noisy Signal"].tolist()

# Initialize the environment
state = env.reset(clean_signal=np.array(current_window_clean),
                  noisy_signal=np.array(current_window_noisy))

print("State variables: time, clean_signal, raw_signal, filtered_signal, SNR_raw, SNR_filtered, prev_reward")
print("Initial state:", state)

# Tracking metrics
rewards = []
snr_raw_list = []
snr_filtered_list = []
snr_diff = []
threshold_factors = []

action = 0  # Initial action
prev_reward = 0

# Iterate through the data using a sliding window
for i in range(window_size, len(df)):
    # Update environment with new signals
    env.set_signal(clean_signal=np.array(current_window_clean),
                   noisy_signal=np.array(current_window_noisy))

    # Step through the environment
    next_state, curr_reward, done, truncated, info = env.step(action)
    
    # Extract SNR values from next_state
    snr_raw = next_state[5]  # Assuming index 7 is SNR_raw
    snr_filtered = next_state[6]  # Assuming index 8 is SNR_filtered
    filtered_signal = next_state[3]
    t_factor = next_state[4]

    # Store results
    rewards.append(curr_reward)
    snr_raw_list.append(snr_raw)
    snr_filtered_list.append(snr_filtered)
    snr_diff.append(snr_filtered - snr_raw)
    threshold_factors.append(t_factor)

    print(f"Rows {i-window_size, i} | Action: {action} | Reward: {curr_reward:.4f} | SNR Raw: {snr_raw:.2f} | SNR Filtered: {snr_filtered:.2f} | Done: {done} | filtered signal: {filtered_signal} | clean signal: {np.mean(current_window_clean)} | threshold factor: {t_factor}")

    # Choose next action based on SNR improvement
    if (prev_reward > curr_reward) | (curr_reward == -1):
        action = 1 
    elif prev_reward <= curr_reward:
        action = 0
    prev_reward = curr_reward

    '''
    if done:
        print("Termination condition met. Resetting environment.")
        state = env.reset(clean_signal=np.array(current_window_clean),
                          noisy_signal=np.array(current_window_noisy))
        break
    '''
    
    # Slide window: Remove oldest, add new
    current_window_clean.pop(0)
    current_window_noisy.pop(0)
    
    current_window_clean.append(df.iloc[i]["Clean Signal"])
    current_window_noisy.append(df.iloc[i]["Noisy Signal"])

env.close()

# --- Visualization ---
plt.figure(figsize=(10, 5))

# Plot SNR changes
#'''
plt.subplot(1, 2, 1)
plt.plot(threshold_factors, label="Threshold Factor", color="purple")
plt.xlabel("Time Steps")
plt.ylabel("Threshold Factor")
plt.title("Threshold Factor Over Time")
plt.legend()
#'''

# Plot rewards
plt.subplot(1, 2, 2)
plt.plot(rewards, label="Reward", color="blue")
plt.xlabel("Time Steps")
plt.ylabel("Reward")
plt.title("Reward Over Time")
plt.legend()

#plt.tight_layout()
plt.show()
