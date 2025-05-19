import numpy as np
import matplotlib.pyplot as plt

# Load the complex IQ data
rx_file_path = 'C:/Users/asmig/OneDrive/Documents/GNURadio/Data/rxdata'
rx_data = np.fromfile(open(rx_file_path), dtype=np.complex64)

# Limit to first 10,000 samples
rx_data = rx_data[:10000]
samples = np.arange(len(rx_data))

# Compute magnitude
magnitude = np.abs(rx_data)

# Plot real and imaginary components
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(samples, rx_data.real, label="Real")
plt.plot(samples, rx_data.imag, label="Imag", alpha=0.7)
plt.title("Received Signal (First 10,000 Samples): Complex Components")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Plot magnitude
plt.subplot(2, 1, 2)
plt.plot(samples, magnitude, color="purple")
plt.title("Received Signal (First 10,000 Samples): Magnitude (|IQ|)")
plt.xlabel("Sample Index")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()
