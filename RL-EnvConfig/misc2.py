import numpy as np
import matplotlib.pyplot as plt
from astra_rev1.envs import NoiseReductionEnv

# Initialize environment
env = NoiseReductionEnv(window_size=10)
env.reset()

# Select a stable window for testing
start = 100
clean_window = env.clean[start:start + env.window_size]
noisy_window = env.noisy[start:start + env.window_size]

threshold_factors = np.linspace(0.5, 2.5, 50)
rewards, snr_improvements, mses = [], [], []

# Evaluate over threshold range
for threshold in threshold_factors:
    action = np.interp(threshold, [0.5, 2.5], [-1.0, 1.0])
    env.set_signal_window(clean_window, noisy_window)
    _, reward, _, info = env.step(np.array([action], dtype=np.float32))

    rewards.append(info["reward"])
    snr_improvements.append(info["SNR_filtered"] - info["SNR_raw"])
    mses.append(np.mean((info["filtered_signal"] - clean_window) ** 2))

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].plot(threshold_factors, rewards, color='red')
axs[0].set_title("Reward vs Threshold")
axs[0].set_xlabel("Threshold Factor")
axs[0].set_ylabel("Reward")

axs[1].plot(threshold_factors, snr_improvements, color='blue')
axs[1].set_title("SNR Improvement vs Threshold")
axs[1].set_xlabel("Threshold Factor")
axs[1].set_ylabel("SNR Improvement (dB)")

axs[2].plot(threshold_factors, mses, color='green')
axs[2].set_title("MSE vs Threshold")
axs[2].set_xlabel("Threshold Factor")
axs[2].set_ylabel("Mean Squared Error")

plt.tight_layout()
plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt

# filename = 'C:/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/Data/rx_tx_data.csv'
# df= pd.read_csv(filename)

# print(df.head())

# plt.figure(figsize=(12, 10))

# plt.subplot(3, 1, 1)
# plt.plot(df['RX Magnitude'], label='Noisy Signal', color='blue')
# plt.plot(df['TX Magnitude'], label='Clean Signal', color='orange')
# plt.title('Noisy and Clean Signals')
# plt.legend()


# plt.subplot(3, 1, 2)
# plt.plot(df['RX Real'], label='Noisy Signal', color='blue')
# plt.plot(df['TX Real'], label='Clean Signal', color='orange')
# plt.title('Real Component of Noisy and Clean Signals')
# plt.legend()


# plt.subplot(3, 1, 3)
# plt.plot(df['RX Imag'], label='Noisy Signal', color='blue')
# plt.plot(df['TX Imag'], label='Clean Signal', color='orange')
# plt.title('Imag Component of Noisy and Clean Signals')
# plt.legend()

# plt.tight_layout()
# plt.show()