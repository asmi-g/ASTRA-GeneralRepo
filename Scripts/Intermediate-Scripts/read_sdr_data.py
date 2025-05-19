import numpy as np
import matplotlib.pyplot as plt
import csv

rx_file_path = 'C:/Users/asmig/OneDrive/Documents/GNURadio/Data/rxdata' # Insert here
tx_file_path = 'C:/Users/asmig/OneDrive/Documents/GNURadio/Data/txdata' # Insert here


# Load data file
rx_data = np.fromfile(open(rx_file_path), dtype=np.complex64)
tx_data = np.fromfile(open(tx_file_path), dtype=np.complex64)

plt.figure("RX Signal")
plt.plot(np.abs(rx_data[-500000:]))  # plot first chunk
plt.title("RX Magnitude Over Time")

plt.figure ("TX Signal")
plt.plot(np.abs(tx_data[-500000:]))  # plot first chunk
plt.title("TX Magnitude Over Time")
plt.show()

# Save to CSV
csv_file_path = 'C:/Users/asmig/OneDrive/Documents/GNURadio/Data/rx_tx_data.csv'



# Save last 10,000 samples of both
tx_data_last = tx_data[-500000:] if len(tx_data) >= 500000 else tx_data
rx_data_last = rx_data[-500000:] if len(rx_data) >= 500000 else rx_data

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Index", 
                     "TX Real", "TX Imag", "TX Magnitude", 
                     "RX Real", "RX Imag", "RX Magnitude"])
    
    for i in range(len(tx_data_last)):  # should be 10,000 if data is long enough
        tx = tx_data_last[i]
        rx = rx_data_last[i] if i < len(rx_data_last) else 0
        writer.writerow([
            i, 
            np.real(tx), np.imag(tx), np.abs(tx), 
            np.real(rx), np.imag(rx), np.abs(rx)
        ])


# print("RX data", rx_data, "TX data", tx_data)

# Visualize logged data
#plt.plot(np.abs(data))
#plt.title("Signal Magnitude")
#plt.xlabel("Sample Index")
#plt.ylabel("Amplitude")
#plt.grid()
#plt.show()