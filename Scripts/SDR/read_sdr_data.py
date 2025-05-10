import numpy as np
import matplotlib.pyplot as plt

rx_file_path = '' # Insert here
tx_file_path = '' # Insert here


# Load data file
rx_data = np.fromfile(open(rx_file_path), dtype=np.complex64)
tx_data = np.fromfile(open(tx_file_path), dtype=np.complex64)

plt.figure("RX Signal")
plt.plot(np.abs(rx_data[:10000]))  # plot rx
plt.title("RX Magnitude Over Time")

plt.figure ("TX Signal")
plt.plot(np.abs(tx_data[:10000]))  # plot tx
plt.title("TX Magnitude Over Time")
plt.show()


# print("RX data", rx_data, "TX data", tx_data)

# Visualize logged data
#plt.plot(np.abs(data))
#plt.title("Signal Magnitude")
#plt.xlabel("Sample Index")
#plt.ylabel("Amplitude")
#plt.grid()
#plt.show()