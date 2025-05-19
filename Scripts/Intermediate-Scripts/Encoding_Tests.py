import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000  # Sampling frequency in Hz
duration = 1  # Duration of the signal in seconds
bit_rate = 100  # Bit rate in bits per second
t = np.linspace(0, duration, int(fs * duration))  # Time vector

# Generate random binary data (0 and 1)
num_bits = int(bit_rate * duration)
data_bits = np.random.randint(0, 2, num_bits)

# BPSK Modulation: map 0 to -1 and 1 to +1
bpsk_signal = 2 * data_bits - 1  # Map 0 -> -1, 1 -> +1

# Repeat each bit to form the signal over time
bpsk_signal = np.repeat(bpsk_signal, fs // bit_rate)

# Plot BPSK signal
plt.figure(figsize=(10, 4))
plt.plot(t[:len(bpsk_signal)], bpsk_signal, label="BPSK Signal")
plt.title("BPSK Modulated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()