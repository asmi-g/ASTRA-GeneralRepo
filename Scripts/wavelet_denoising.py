import pandas as pd
import matplotlib.pyplot as plt
import pywt

# Constants
FILE_PATH = 'Data/simulated_signal_data.csv'
ADC_BIT_RESOLUTION = 12
V_REF = 5
WAVELET = 'db8'
THRESHOLD = 2
THRESHOLD_MODE = 'soft'
LEVEL = 2
LAST_SIGNAL_INDEX_TO_PLOT = 500

LEVEL_MIN_VAL = 4
LEVEL_MAX_VAL = 6
THRESHOLD_MIN_VAL = 1
THRESHOLD_MAX_VAL = 4

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
        # TODO: As per Asmi's comments, do we want/need this with the RL model?
        # if not self.threshold:
        #     # Donoho and Silverman Universal Threshold
        #     sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        #     self.threshold = sigma * np.sqrt(2 * np.log(len(self.noisy_signal)))
        coeffs[1:] = [pywt.threshold(c, self.threshold, mode=self.threshold_mode) for c in coeffs[1:]]
        self.denoised_signal = pywt.waverec(coeffs, self.wavelet)
        return self.denoised_signal

    def get_threshold(self):
        return self.threshold

    def get_level(self):
        return self.level

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
    denoisers = [[WaveletDenoiser(time_vector, normalized_noisy_signal, normalized_clean_signal, WAVELET, threshold, THRESHOLD_MODE, level) 
                  for threshold in range(THRESHOLD_MIN_VAL, THRESHOLD_MAX_VAL + 1)] 
                  for level in range(LEVEL_MIN_VAL, LEVEL_MAX_VAL + 1)]

    # Plot thresholding coefficients and decomposition level sweep
    for denoiser_level in denoisers:
        fig, axes = plt.subplots(THRESHOLD_MAX_VAL - THRESHOLD_MIN_VAL + 1, sharex=True, sharey=True)
        for index, denoiser in enumerate(denoiser_level):
            denoised_signal = denoiser.denoise_signal()
            # TODO: Why is this happening with the our data?
            denoised_signal = denoised_signal[:-1]
            subplot = axes[index]
            subplot.plot(time_vector[:LAST_SIGNAL_INDEX_TO_PLOT], normalized_noisy_signal[:LAST_SIGNAL_INDEX_TO_PLOT], label="Noisy Signal")
            subplot.plot(time_vector[:LAST_SIGNAL_INDEX_TO_PLOT], denoised_signal[:LAST_SIGNAL_INDEX_TO_PLOT], label = "Filtered Signal")
            subplot.plot(time_vector[:LAST_SIGNAL_INDEX_TO_PLOT], normalized_clean_signal[:LAST_SIGNAL_INDEX_TO_PLOT], label="Clean Signal")
            subplot.grid()
        level = denoiser.get_level()
        fig.suptitle(f"Wavelet: {WAVELET}   Level: {level}   Threshold: {THRESHOLD_MIN_VAL} - {THRESHOLD_MAX_VAL}") 
        fig.supxlabel("Time [s]")
        fig.supylabel("Signal Amplitude [V]")
        handles, labels = subplot.get_legend_handles_labels()
        fig.legend(handles, labels)
    plt.show()

if __name__ == "__main__":
    main()