import numpy as np
import matplotlib.pyplot as plt

# Parameters
F_S = 10e6 # Sampling or ADC frequency (10 MHz)
F_SIGNAL = 10e3 # Signal frequency (10 kHz)
DURATION = 1 # Signal duration in seconds

# Discrete time points vector
t = np.arange(0, DURATION, 1/F_S)

# Generate RF signal (sine wave)
signal = np.sin(2 * np.pi * F_SIGNAL * t)

# Generate cosmic noise (Gaussian noise for now)
# TODO: Look into whether this value actually makes sense?
# Increasing the noise_power generates more intense noise
noise_power = 0.1
noise = np.random.normal(0, np.sqrt(noise_power), len(t))

# Noisy signal
noisy_signal = signal + noise

# Calculate power levels
signal_power = np.mean(signal**2)
noise_power = np.mean(noise**2)

# Compute SNR in dB
SNR_dB = 10 * np.log10(signal_power / noise_power)
print(f"SNR: {SNR_dB:.2f} dB")

# Plot results
# TODO: Do we want to increase the viewing window?
plt.figure(figsize=(10, 5))
plt.plot(t[:10000], signal[:10000], label="Clean signal")
plt.plot(t[:10000], noisy_signal[:10000], label="Noisy Signal", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Signal with Cosmic Noise")
plt.show()

'''
TODO: Next steps: continuously streaming in a while loop
- Make it real-time
- Signal plot being updated as you go
- Live plot and then write values to a txt and then integrate with OpenAI gym
- Have multiple sets of randomized signal data
- Real-time plotting and printing to a file
'''