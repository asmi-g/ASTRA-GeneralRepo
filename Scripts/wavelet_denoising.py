import csv
import pywt
import matplotlib.pyplot as plt

# Constants
THRESHOLD = 2
WAVELET = 'db4'

# Globals
time = []
noisy_signal = []
clean_signal = []

# Read in noisy signal data from csv
# TODO: Can do this with pandas
with open('../Data/simulated_signal_data.csv') as signal_data_csv:
    signal_data_reader = csv.reader(signal_data_csv, delimiter=',')
    
    # Skipping header
    next(signal_data_reader)

    # Reading in data
    for i, row in enumerate(signal_data_reader):
        if i >= 2000:
            break
        time.append(row[0])
        noisy_signal.append(row[1])
        clean_signal.append(row[2])

# Apply wavelet transform and obtain coefficients
coeffs = pywt.dwt(noisy_signal, WAVELET)

# Apply soft thresholding
denoised_coeffs = [pywt.threshold(c, THRESHOLD, mode='soft') for c in coeffs]

# Reconstruct the signal
denoised_signal = pywt.idwt(denoised_coeffs[0], denoised_coeffs[1], WAVELET)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, noisy_signal, label='Noisy Signal')
plt.plot(time, denoised_signal, label='Denoised Signal')
plt.plot(time, clean_signal, label='Original Signal')
plt.autoscale()
plt.legend()
plt.title("wavelet Denoising Example")
plt.show()
