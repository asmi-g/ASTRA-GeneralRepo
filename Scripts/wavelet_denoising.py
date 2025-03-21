import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

# Constants
FILE_PATH = 'Data/simulated_signal_data.csv'
ADC_BIT_RESOLUTION = 12
V_REF = 5
WAVELET = 'db4'
THRESHOLD = 2
THRESHOLD_MODE = 'soft'
LEVEL = 4
LAST_SIGNAL_INDEX_TO_PLOT = 500

def convert_ADC_to_signal(adc_values, v_ref, adc_bit_resolution):
    normalized_signal_reversed = adc_values / (2 ** adc_bit_resolution - 1)
    signal_reversed = normalized_signal_reversed * v_ref - v_ref / 2
    return signal_reversed

class WaveletDenoiser:
    def __init__(self, time_vector, noisy_signal, clean_signal, wavelet, threshold=None, threshold_mode='soft', level=0):
        self.time_vector = time_vector
        self.noisy_signal = noisy_signal
        self.clean_signal = clean_signal
        self.denoised_signal = None
        self.wavelet = wavelet
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.level = level

    def __apply_wavelet_transform(self, noisy_signal):
        coeffs = pywt.wavedec(noisy_signal, self.wavelet, level=self.level)
        return coeffs
        # You could also use single-level transform but it seems the multilevel transform accepts anything >= 0 for the level

    def denoise_signal(self):
        coeffs = self.__apply_wavelet_transform(self.noisy_signal)
        if not self.threshold:
            # Donoho and Silverman Universal Threshold
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            self.threshold = sigma * np.sqrt(2 * np.log(len(self.noisy_signal)))
        coeffs[1:] = [pywt.threshold(c, self.threshold, mode=self.threshold_mode) for c in coeffs[1:]]
        self.denoised_signal = pywt.waverec(coeffs, self.wavelet)
        return self.denoised_signal

def main():
    # Reading in ADC data
    df = pd.read_csv(FILE_PATH)
    time_vector = df["Time"].values
    noisy_signal = df["Noisy Signal"].values
    clean_signal = df["Clean Signal"].values

    # Converting ADC data to signal 
    normalized_noisy_signal = convert_ADC_to_signal(noisy_signal, V_REF, ADC_BIT_RESOLUTION)
    normalized_clean_signal = convert_ADC_to_signal(clean_signal, V_REF, ADC_BIT_RESOLUTION)

    # Denoise the signal
    waveletDenoiser = WaveletDenoiser(time_vector, normalized_noisy_signal, normalized_clean_signal, WAVELET, THRESHOLD, THRESHOLD_MODE, LEVEL)
    denoised_signal = waveletDenoiser.denoise_signal()

    # TODO: Why is this happening with the our data?
    denoised_signal = denoised_signal[:-1]

    # Plot
    plt.figure()
    plt.plot(time_vector[:LAST_SIGNAL_INDEX_TO_PLOT], normalized_noisy_signal[:LAST_SIGNAL_INDEX_TO_PLOT], label="Noisy Signal")
    plt.plot(time_vector[:LAST_SIGNAL_INDEX_TO_PLOT], denoised_signal[:LAST_SIGNAL_INDEX_TO_PLOT], label = "Filtered Signal")
    plt.plot(time_vector[:LAST_SIGNAL_INDEX_TO_PLOT], normalized_clean_signal[:LAST_SIGNAL_INDEX_TO_PLOT], label="Clean Signal")
    plt.title("Wavelet Denoising Signal Plot")
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Amplitude [V]")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()