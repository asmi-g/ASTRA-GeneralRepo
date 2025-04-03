import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from astra_rev1.envs import NoiseReductionEnv

# --- Generate Synthetic Signal ---
def generate_synthetic_signal(signal_type="sine", noise_level=0.2, length=100):
    """
    Generates a clean signal and adds noise to create a noisy version.
    
    :param signal_type: Type of signal ("sine", "square", "sawtooth", "random")
    :param noise_level: Standard deviation of Gaussian noise
    :param length: Number of samples
    :return: (clean_signal, noisy_signal)
    """
    t = np.linspace(0, 1, length)

    if signal_type == "sine":
        clean_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    elif signal_type == "square":
        clean_signal = np.sign(np.sin(2 * np.pi * 5 * t))  # Square wave
    elif signal_type == "sawtooth":
        clean_signal = 2 * (t * 5 % 1) - 1  # Sawtooth wave
    elif signal_type == "random":
        clean_signal = np.random.uniform(-1, 1, size=length)  # Random signal
    else:
        raise ValueError("Unknown signal type!")

    noise = np.random.normal(0, noise_level, size=length)
    noisy_signal = clean_signal + noise
    return clean_signal, noisy_signal

# Generate the signal
clean_signal, noisy_signal = generate_synthetic_signal("sine", noise_level=0.3)

'''
# Optional: Plot the signals to verify
plt.figure(figsize=(10, 4))
plt.plot(clean_signal, label="Clean Signal")
plt.plot(noisy_signal, label="Noisy Signal", alpha=0.7)
plt.legend()
plt.title("Generated Synthetic Signal")
plt.show()'
'''

env = NoiseReductionEnv()
env.reset(clean_signal=clean_signal, noisy_signal=noisy_signal)

# Check if environment is compatible with Stable-Baselines3
check_env(env, warn=True)

# --- DQN Model Configuration ---
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.01,
    exploration_initial_eps = 1,
    exploration_final_eps=0.01,
    target_update_interval=500,
    train_freq=1,
    gradient_steps=1,
    verbose=1
)

# --- Training the Agent ---
TIMESTEPS = 100000
print(f"Training DQN for {TIMESTEPS} timesteps...")
model.learn(total_timesteps=TIMESTEPS, log_interval=10000)

# Save trained model
model.save("dqn_noise_reduction")
print("Model saved as 'dqn_noise_reduction'")

# Close environment
env.close()