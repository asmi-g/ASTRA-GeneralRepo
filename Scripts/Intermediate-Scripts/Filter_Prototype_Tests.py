import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

ADC_BITS = 12  # 12-bit ADC
V_REF = 5    # Reference voltage

# SCRIPT: TEST OUT HOW EFFECTIVE WAVELET DENOISING FILTERING IS, ON CURRENT GENERATED DATA
# METHOD: FILTERS AND PLOTS A RANGE OF THRESHOLDING VALUES USING THE PYWAVELETS LIBRARY

# Sample parameters 
fs = 10000
data_path = 'C:/Users/asmig/design-team-work/ASTRA-GeneralRepo/Data/simulated_signal_data.csv'

def read_noisy_data(file_path):
    """Reads the noisy voltage values from CSV"""
    df = pd.read_csv(file_path)  # Read CSV
    noisy_voltages = df.iloc[:, 1].values  
    return noisy_voltages

def convert_ADC_to_signal(adc_value):
    normalized_signal_reversed = adc_value / (2**ADC_BITS - 1)
    signal_reversed = normalized_signal_reversed * V_REF - V_REF / 2
    
    return signal_reversed 

def wavelet_denoise(signal, wavelet='db4', level=2, threshold=None):
    """Applies wavelet denoising to the signal in time domain"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimate noise level using the last level's coefficients (approximation)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimate noise level
    
    if threshold is None:
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))  # Default to universal threshold
    
    # Apply soft thresholding
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    
    return pywt.waverec(coeffs_thresh, wavelet)

# Read noisy data from file
noisy_voltages = read_noisy_data(data_path)

# Normalize the ADC values to signals
noisy_signal = convert_ADC_to_signal(noisy_voltages)

# Set range for threshold values to plot
threshold_values = np.linspace(0, np.max(noisy_signal), 10)  # Adjust range and number of values

# Loop through threshold values, plot each
for threshold in threshold_values:
    # Apply wavelet denoising for the current threshold
    filtered_signal = wavelet_denoise(noisy_signal, wavelet='db4', level=2, threshold=threshold)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the noisy signal
    plt.plot(noisy_signal, label="Noisy Signal", linestyle='solid', alpha=0.7)
    
    # Plot the filtered signal
    plt.plot(filtered_signal, label=f'Denoised Signal (Threshold = {threshold:.4f})', linestyle='solid', linewidth=2)
    
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (V) / Signal")
    plt.title(f"Wavelet Denoising with Threshold = {threshold:.4f}")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.grid()
    
    # Show plot 
    plt.show()
