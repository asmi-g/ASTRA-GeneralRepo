import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate clean signal
np.random.seed(42)
t = np.linspace(0, 10, 1000)
x = np.sin(t)  # Example clean signal

# Generate noise
eps = np.random.normal(0, 0.2, len(x))

# Mixture signal
mixture = x + eps

# Define the filter function (simple moving average)
def filter_function(signal, wavelet='db4', level=1, threshold_factor=0.5):
    # threshold_factor is modifiable
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = threshold_factor * sigma
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

def calculate_SNR(clean_signal, noisy_signal):
    if np.mean(noisy_signal) == 0:
        return np.nan
    noise = noisy_signal - clean_signal  # Extract noise component
    signal_power = np.mean(clean_signal**2)  # Power of the clean signal
    noise_power = np.mean(noise**2)  # Power of the noise
    return 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))  # in dB

# Define reward function (example: signal-to-noise ratio improvement)
def reward_function(filtered_signal, noisy_signal, clean_signal, lambda_penalty=0.5):
    """Compute reward based on SNR improvement and distortion penalty."""
    # Compute SNR values
    SNR_noisy = calculate_SNR(clean_signal, noisy_signal)
    SNR_filtered = calculate_SNR(clean_signal, filtered_signal)
    
    # Compute SNR improvement
    delta_SNR = (SNR_filtered - SNR_noisy)   # Normalized improvement
    
    # Compute distortion penalty
    #distortion_penalty = lambda_penalty * np.mean((filtered_signal - clean_signal) ** 2)
    
    # Final reward
    reward = np.log1p(max(SNR_filtered - SNR_noisy, 0))
#delta_SNR/ (abs(SNR_noisy) + 1e-6) #- distortion_penalty
    return reward

alphas = np.linspace(0, 1, 50)
A_values = []
R_values = []

for alpha in alphas:
    y = x + alpha * eps  # Sweep y
    y_filtered = filter_function(y)  # Apply filter
    A_values.append(np.mean(np.abs(y_filtered - x)))  # Example metric A(y)
    R_values.append(reward_function(y_filtered, mixture, x))  # Compute reward

# Plot results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(alphas, A_values, label="A(y)")
plt.xlabel("Alpha")
plt.ylabel("A(y)")
plt.title("Alpha vs A(y)")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(alphas, R_values, label="R(y)", color='r')
plt.xlabel("Alpha")
plt.ylabel("Reward R(y)")
plt.title("Alpha vs R(y)")
plt.grid()

plt.tight_layout()
plt.show()














'''
import numpy as np
import matplotlib.pyplot as plt
from astra_rev1.envs import NoiseReductionEnv  # Import your custom environment

# Step 1: Create a clean signal and add noise to it
def generate_synthetic_signal(signal_type="sine", noise_level=0.3, length=100):
    t = np.linspace(0, 1, length)

    if signal_type == "sine":
        clean_signal = np.sin(2 * np.pi * 5 * t)
    elif signal_type == "square":
        clean_signal = np.sign(np.sin(2 * np.pi * 5 * t))
    elif signal_type == "sawtooth":
        clean_signal = 2 * (t * 5 % 1) - 1
    elif signal_type == "random":
        clean_signal = np.random.uniform(-1, 1, size=length)
    else:
        raise ValueError("Unknown signal type!")

    noise = np.random.normal(0, noise_level, size=length)
    noisy_signal = clean_signal + noise
    return clean_signal, noisy_signal

# Step 2: Initialize the environment
env = NoiseReductionEnv()

# Step 3: Generate the signals
clean_signal, noisy_signal = generate_synthetic_signal("sine", noise_level=0.3, length=100)

# Pass the signals to the environment
env.reset(clean_signal=clean_signal, noisy_signal=noisy_signal)

# Step 4: Update threshold_factor externally (outside of the environment)
new_threshold_factor = 2.2 # Example: change the threshold factor to 0.3
env.set_threshold_factor(new_threshold_factor)

# Run the environment for a few steps to get the filtered signal
filtered_signal = np.copy(noisy_signal)
for _ in range(10):  # Run a few steps
    action = env.action_space.sample()  # Random action (increase or decrease threshold)
    _, _, _, _, info = env.step(action)  # Take the step
    snr_raw = info["SNR_raw"]     
    snr_filtered = info["SNR_filtered"]  
    filtered_signal = info["filtered_signal"]  # Full window
    t_factor = info["threshold_factor"]

    # Retrieve the filtered signal after the step
    filtered_signal = env.filtered_signal
    print(f"SNR Raw: {snr_raw:.2f} | SNR Filtered: {snr_filtered:.2f} | filtered signal: {np.mean(filtered_signal):.4f} | clean signal: {np.mean(clean_signal):.4f} | threshold factor: {t_factor:.4f}")


# Step 5: Plot the clean, noisy, and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(clean_signal, label="Clean Signal", color="g", alpha=0.7)
plt.plot(noisy_signal, label="Noisy Signal", color="r", alpha=0.7)
plt.plot(filtered_signal, label="Filtered Signal", color="b", alpha=0.7)
plt.legend()
plt.title("Clean, Noisy, and Filtered Signals")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Clean up environment
env.close()
'''