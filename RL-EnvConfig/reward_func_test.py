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
