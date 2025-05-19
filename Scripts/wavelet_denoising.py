import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt

data_path = '../Data/test_denoised_data.csv'
class WaveletDenoiser:
    def __init__(self, path_to_signal_data, wavelet, threshold_mode='soft', level=0, threshold=0):
        self.wavelet = wavelet
        self.threshold_mode = threshold_mode
        self.threshold = threshold
        self.level = level
        self.signal_data_frame = pd.read_csv(path_to_signal_data)
        self.denoised_signal = None

    def __apply_wavelet_transform(self, noisy_signal):
        if self.level:
            coeffs = pywt.wavedec(noisy_signal, self.wavelet, level=self.level)
        else:
            coeffs = pywt.dwt(noisy_signal, self.wavelet)
        return coeffs
    
    def denoise_signal(self):
        noisy_signal = self.signal_data_frame["Noisy Signal"].values
        coeffs = self.__apply_wavelet_transform(noisy_signal)
        if not self.threshold:
            # Donoho and Silverman Universal Threshold
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745 
            self.threshold = sigma * np.sqrt(2 * np.log(len(noisy_signal)))
        coeffs[1:] = [pywt.threshold(c, self.threshold, mode=self.threshold_mode) for c in coeffs[1:]]
        self.denoised_signal = pywt.waverec(coeffs, self.wavelet)
    
    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.signal_data_frame["Time"], self.signal_data_frame["Noisy Signal"].values, label="Noisy Signal", linestyle="dashed", alpha=0.4, color='red')
        plt.plot(self.signal_data_frame["Time"], self.denoised_signal[:-1], label="Denoised Signal (Multi-Level, Hard Threshold)", linewidth=2, color='blue')
        #df = pd.DataFrame({
         #           'Denoising Signal': self.denoised_signal, 
        #})
        #df.to_csv(data_path, mode='a', index=False, header=not pd.io.common.file_exists(data_path))
        plt.plot(self.signal_data_frame["Time"], self.signal_data_frame["Clean Signal"].values, label="Clean Signal (Reference)", linestyle="dotted", linewidth=2, color='green')

        plt.legend()
        plt.title("Wavelet Denoising")
        plt.xlabel("Time")
        plt.ylabel("Signal Amplitude")
        plt.grid()
        plt.show()

def main():
    waveletDenoiser = WaveletDenoiser('../Data/simulated_signal_data.csv', 'db4', 'soft', 4)
    waveletDenoiser.denoise_signal()
    waveletDenoiser.plot()

if __name__ == "__main__":
    main()
    
