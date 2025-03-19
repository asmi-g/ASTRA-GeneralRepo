import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from astra_rev1.envs import NoiseReductionEnv

# Load dataset (first 10,000 rows)
df = pd.read_csv("Data/simulated_signal_data.csv", skiprows = lambda x: 0<x<=10000, nrows=2000)

# Load trained DQN model
model = DQN.load("dqn_noise_reduction")

# Initialize environment
env = NoiseReductionEnv()

# Sliding window setup
window_size = 10
current_window_clean = df.iloc[:window_size]["Clean Signal"].tolist()
current_window_noisy = df.iloc[:window_size]["Noisy Signal"].tolist()
time = df.iloc[:window_size]["Time"].tolist()

# Reset environment with initial state
state, _ = env.reset(clean_signal=np.array(current_window_clean),
                     noisy_signal=np.array(current_window_noisy))

# Tracking metrics
actions = []
rewards = []
snr_raw_list = []
snr_filtered_list = []

# Run inference
print("Running inference using trained DQN model...")
for i in range(window_size, len(df)):
    
    # Predict action using the trained model
    action, _ = model.predict(state, deterministic=True)
    actions.append(action)
    
    # Step in environment
    next_state, reward, done, truncated, info = env.step(action)
    
    # Extract SNR values from next state
    snr_raw = next_state[5]      # Assuming SNR Raw is at index 5
    snr_filtered = next_state[6]  # Assuming SNR Filtered is at index 6
    filtered_signal = next_state[3]
    t_factor = next_state[4]

    
    snr_raw_list.append(snr_raw)
    snr_filtered_list.append(snr_filtered)
    rewards.append(reward)

    #print(f"Step {i} | Action: {action} | Reward: {reward:.4f} | SNR Raw: {snr_raw:.2f} | SNR Filtered: {snr_filtered:.2f}")
    print(f"Rows {i-window_size, i} | Action: {action} | Reward: {reward:.4f} | SNR Raw: {snr_raw:.2f} | SNR Filtered: {snr_filtered:.2f} | Done: {done} | filtered signal: {filtered_signal} | clean signal: {np.mean(current_window_clean)} | threshold factor: {t_factor}")

    # Slide window: Remove oldest, add new
    current_window_clean.pop(0)
    current_window_noisy.pop(0)
    current_window_clean.append(df.iloc[i]["Clean Signal"])
    current_window_noisy.append(df.iloc[i]["Noisy Signal"])
    time.pop(0)
    time.append(df.iloc[i]["Time"])

    # Update the state with the new window
    state = np.array([
        time[-1],                 # Time value at step i (this can change based on your data structure)
        np.mean(current_window_clean),  # Mean of the clean signal over the window
        np.mean(current_window_noisy),  # Mean of the noisy signal over the window
        np.mean(next_state[2]),         # Placeholder: Mean of the filtered signal, needs proper handling
        env.threshold_factor,          # Current threshold factor
        next_state[5],                 # SNR raw from the next state
        next_state[6],                 # SNR filtered from the next state
        reward                        # The previous reward as the state (this could be adjusted depending on your logic)
    ], dtype=np.float32)

    if done:
        print("Termination condition met. Resetting environment.")
        state = env.reset(clean_signal=np.array(current_window_clean),
                          noisy_signal=np.array(current_window_noisy))
        break

# Close environment
env.close()

# --- Visualization ---
plt.figure(figsize=(10, 5))

# Plot SNR changes
plt.subplot(1, 2, 1)
plt.plot(snr_raw_list, label="SNR Raw", linestyle="dashed", color="red")
plt.plot(snr_filtered_list, label="SNR Filtered", color="green")
plt.xlabel("Time Steps")
plt.ylabel("SNR (dB)")
plt.title("SNR Raw vs. SNR Filtered")
plt.legend()

'''
# Plot actions
plt.subplot(1, 2, 2)
plt.plot(actions, label="Actions Taken", color="blue")
plt.xlabel("Time Steps")
plt.ylabel("Action")
plt.title("Actions Over Time")
plt.legend()
'''

plt.subplot(1, 2, 2)
plt.plot(rewards, label="Reward", color="blue")
plt.xlabel("Time Steps")
plt.ylabel("Reward")
plt.title("Reward Over Time")
plt.legend()

plt.tight_layout()
plt.show()
