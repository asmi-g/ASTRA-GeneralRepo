import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

# Constants
ADC_BITS = 12  # 12-bit ADC
VREF = 5.0     # Reference voltage

def read_noisy_data(file_path):
    """Reads only the noisy ADC values from the second column of a CSV file and converts them to voltages."""
    df = pd.read_csv(file_path)  # Read CSV
    noisy_adc_values = df.iloc[:, 1].values  # Select the second column (Noisy Data)
    voltages = (noisy_adc_values / (2**ADC_BITS - 1)) * VREF  # Convert ADC values to voltage
    return voltages

def wavelet_denoise(signal, wavelet='db4', level=2):
    """Applies wavelet denoising to the signal."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimate noise level
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))  # Universal threshold
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)

# Load noisy ADC data
data_path = '../Data/simulated_signal_data.csv'
noisy_voltages = read_noisy_data(data_path)


# Apply wavelet denoising
filtered_signal = wavelet_denoise(noisy_voltages)

plt.figure(figsize=(10, 5))
plt.plot(noisy_voltages, label="Noisy Signal", linestyle='solid', alpha=0.7)
plt.plot(filtered_signal, label="Filtered Signal", linestyle='solid', linewidth=2)
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.title("Wavelet Denoising of ADC Signal")
plt.legend()
plt.grid()
plt.show()