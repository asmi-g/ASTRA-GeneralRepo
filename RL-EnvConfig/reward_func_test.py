# import numpy as np
# import matplotlib.pyplot as plt

# # Define a sample clean and noisy window
# np.random.seed(0)
# window_size = 10
# clean = np.sin(np.linspace(0, np.pi, window_size))
# noise = np.random.normal(0, 0.3, window_size)
# noisy = clean + noise

# # Define wavelet denoising function
# import pywt
# def wavelet_denoise(window, threshold_factor, wavelet='db4', level=1):
#     coeffs = pywt.wavedec(window, wavelet, level=level)
#     sigma = np.median(np.abs(coeffs[-1])) / 0.6745
#     threshold = threshold_factor * sigma
#     coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
#     return pywt.waverec(coeffs, wavelet)[:len(window)]

# # Define reward function
# def compute_reward(clean, noisy, threshold_factor, alpha, beta, gamma):
#     filtered = wavelet_denoise(noisy, threshold_factor)
#     snr_raw = 10 * np.log10(np.mean(clean**2) / (np.mean((noisy - clean)**2) + 1e-10))
#     snr_filtered = 10 * np.log10(np.mean(clean**2) / (np.mean((filtered - clean)**2) + 1e-10))
#     snr_gain = snr_filtered - snr_raw
#     signal_loss = np.mean((filtered - clean)**2)
#     correlation = np.corrcoef(filtered, clean)[0, 1]
#     reward = alpha * snr_gain - beta * signal_loss + gamma * correlation
#     return reward

# # Threshold factors to test
# threshold_factors = np.linspace(0.1, 5.0, 100)

# # Sweep through different (alpha, beta, gamma) values
# param_sets = [
#     (0.5, 0.5, 0.5),
#     (0.25, 0.5, 0.5),
#     (0.75, 0.25, 0.5),
#     (0.5, 0.5, 0.75),
#     (1.0, 1.0, 1.0),
# ]

# # Plot reward vs threshold for each parameter set
# plt.figure(figsize=(12, 6))
# for alpha, beta, gamma in param_sets:
#     rewards = [compute_reward(clean, noisy, tf, alpha, beta, gamma) for tf in threshold_factors]
#     plt.plot(threshold_factors, rewards, label=f"α={alpha}, β={beta}, γ={gamma}")

# plt.xlabel("Threshold Factor")
# plt.ylabel("Reward")
# plt.title("Reward vs Threshold Factor for Different α, β, γ Combinations")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from astra_rev1.envs import NoiseReductionEnv  # Update path as needed

# Setup
env = NoiseReductionEnv(window_size=10)
clean_signal, noisy_signal = env._generate_signals()

window_starts = np.linspace(0, len(clean_signal) - env.window_size, 5, dtype=int)  # 5 sample windows
threshold_factors = np.linspace(0.5, 2.5, 50)

results = []

for t in window_starts:
    clean_win = clean_signal[t:t+env.window_size]
    noisy_win = noisy_signal[t:t+env.window_size]
    
    rewards = []
    for tf in threshold_factors:
        action = [np.interp(tf, [0.5, 2.5], [-1.0, 1.0])]  # inverse mapping
        _, reward, _, info = env.denoiser.step(noisy_win, action, clean_win)
        rewards.append(reward)
    
    results.append((t, rewards))

# --- Plot ---
plt.figure(figsize=(10, 6))
for t, rewards in results:
    plt.plot(threshold_factors, rewards, label=f"Window @ {t}")
plt.xlabel("Threshold Factor")
plt.ylabel("Reward")
plt.title("Reward vs Threshold Factor Across Signal Windows")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
