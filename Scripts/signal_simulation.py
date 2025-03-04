# TO DO:
# - Look into how encoding might change wave
# - Verify w/ both filters: Wavelet Denoising and EMD
# - Verify cosmic noise model is accurate
# - Verify burst is accurate (i.e, amplitude and duration)
# - Validate data somehow
# - Save data according to the # of data points we want

# CHANGES:
# - X plot now updates w/ time
# - Signal data now converted to ADC values
# - "Real-time" ADC stream of data added (as opposed to an array of values)
# - Integrated cosmic noise model
# - Both noisy and clean/ground truth signal ADC streams added
# - Added in the ability to save ADC streaming data to csv 
# - Added in burst noise function


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Signal Parameters

'''
F_SAMPLING = 10e6 # Sampling or ADC frequency (10 MHz)
F_SIGNAL = 10e3 # Signal frequency (10 kHz)
NOISE_POWER = 0.1 # Noise power level
WINDOW_SIZE = 500

'''

F_SAMPLING = 1e4
F_SIGNAL = 100
NOISE_POWER = 0.1
WINDOW_SIZE = 500

# ADC Conversion Parameters (to map the noisy signal value to an ADC 'ratio')
'''
ADC_BITS = 12 # 12 BIT ADC, RANGE: 0 - 4096 (2^12 Bits)
V_REF = 5V    # Reference Voltage, 5V

'''
ADC_BITS = 12 
V_REF = 5

# Save CSV to the Data folder outside of Scripts
data_path = '../Data/simulated_signal_data.csv'  # ../ goes up one directory


# Add in time counter, to track and update in plot
current_time = 0
x_axis = []
y_axis_noisy = []
y_axis_clean = []

# ADC Stream Data Buffers
adc_noise_stream = np.zeros(WINDOW_SIZE)  # Initialize ADC value buffer for noisy signal samples
adc_clean_stream = np.zeros(WINDOW_SIZE)  # Initialize ADC value buffer for clean signal samples



# Initialize figure
fig, ax = plt.subplots()
line_noisy, = ax.plot([], [], label="Noisy Signal", color='blue')
line_clean, = ax.plot([], [], label="Clean Signal", color='red')


ax.set_ylim(-2.5, 2.5) # Adjust based on signal strength
ax.set_xlim(0, WINDOW_SIZE / F_SAMPLING)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")


# Fixed plot legend
legend = ax.legend(loc='upper right')
legend.set_draggable(False)  # Prevents moving when updated

ax.set_title("Real-Time Simulated Signal")

def calculate_SNR(signal, noise):
    # Calculate power levels
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    # Compute SNR in dB
    SNR_dB = 10 * np.log10(signal_power / noise_power)
    # print(f"SNR: {SNR_dB:.2f} dB")

    return SNR_dB 
   

def convert_ADC_array(signal):
    # Normalize incoming signal to a range of 0 to reference voltage (5)
    # Assumes the input signal range is from -V_REF/2, to +V_REF/2
    normalized_signal = (signal + V_REF / 2) / V_REF
    # Scale to ADC range and clip values that are out of bounds
    adc_values = np.clip(np.round(normalized_signal * (2**ADC_BITS - 1)), 0, 2**ADC_BITS - 1)
    return adc_values.astype(int)

def convert_ADC_stream(signal):
    normalized_signal = (signal + V_REF / 2) / V_REF
    adc_value = np.clip(np.round(normalized_signal * (2**ADC_BITS - 1)), 0, 2**ADC_BITS - 1)
    return int(adc_value)  

def generate_pink_noise(size, white_noise):
    # Uses the Voss-McCartney Algorithm to generate pink noise
    # Create a filter to shape the noise into pink noise
    freq = np.fft.fftfreq(size)
    # 1/f power law filter, according to cosmic noise documentation
    pink_filter = np.sqrt(np.abs(freq))  
    
    # Apply the filter in the frequency domain
    spectrum = np.fft.fft(white_noise/np.sqrt(NOISE_POWER))
    spectrum *= pink_filter
    
    pink_noise = np.fft.ifft(spectrum)
    return np.real(pink_noise)

def add_burst(signal, burst_probability=0.01, burst_amplitude=3.0, burst_duration=10):
    # Add in random bursts of noise
    noisy_signal = signal.copy()
    for i in range(len(signal)):
        if np.random.rand() < burst_probability:
            burst_start = i
            # Add burst noise at random interval and amplitude
            burst_end = min(i + burst_duration, len(signal))
            noisy_signal[burst_start:burst_end] += np.random.uniform(-burst_amplitude, burst_amplitude)
    return noisy_signal


def update(frame):
    # Track the time as frames are updated
    global current_time, x_axis, y_axis_noisy, y_axis_clean, adc_noise_stream, adc_clean_stream

    # Generate new signal and noise, which consist of both, white and pink noise
    signal = np.sin(2 * np.pi * F_SIGNAL * (frame + np.arange(WINDOW_SIZE)) / F_SAMPLING)
    white_noise = np.random.normal(0, np.sqrt(NOISE_POWER), WINDOW_SIZE)
    pink_noise = generate_pink_noise(WINDOW_SIZE, white_noise) * np.sqrt(NOISE_POWER)
    noisy_signal = add_burst(signal) + pink_noise + white_noise

    # ADC ARRAY: PLOT
    # Convert to ADC values
    adc_values = convert_ADC_array(noisy_signal)
    # Convert ADC to voltage
    quantized_signal = (adc_values / (2**ADC_BITS - 1)) * V_REF - V_REF / 2
    
    # ADC DATA STREAM VALUE: SINGLE SAMPLE, NOISY DATA
    noisy_sample = noisy_signal[frame % WINDOW_SIZE]
    # Convert to ADC value
    adc_noisy_datastream_val = convert_ADC_stream(noisy_sample)
    print("Time", current_time/10000, "Streaming NOISY ADC Value:", adc_noisy_datastream_val)
    # Convert ADC value to voltage 
    quantized_noisy_datastream_signal = (adc_noisy_datastream_val / (2**ADC_BITS - 1)) * V_REF - V_REF / 2
    # Shift data buffer
    adc_noise_stream = np.roll(adc_noise_stream, -1)  # Shift left by one
    adc_noise_stream[-1] = quantized_noisy_datastream_signal  # Add the new value


    # ADC DATA STREAM VALUE: SINGLE SAMPLE, CLEAN DATA
    clean_sample = signal[frame % WINDOW_SIZE]
    # Convert to ADC value
    adc_clean_datastream_val = convert_ADC_stream(clean_sample)
    print("Time", current_time/10000, "Streaming CLEAN ADC Value:", adc_clean_datastream_val)
    # Convert ADC value to voltage 
    quantized_clean_datastream_signal = (adc_clean_datastream_val / (2**ADC_BITS - 1)) * V_REF - V_REF / 2
    # Shift data buffer
    adc_clean_stream = np.roll(adc_clean_stream, -1)  # Shift left by one
    adc_clean_stream[-1] = quantized_clean_datastream_signal  # Add the new value
    


    # Update the time vector to shift with the data
    new_time = (current_time + np.arange(WINDOW_SIZE)) / F_SAMPLING
    x_axis.extend(new_time)
    y_axis_noisy.extend(quantized_signal)
    y_axis_clean.extend(signal)
    current_time += WINDOW_SIZE  # Move forward in time

    # Update plot data
    line_clean.set_data(x_axis[-WINDOW_SIZE:], y_axis_clean[-WINDOW_SIZE:])
    line_noisy.set_data(x_axis[-WINDOW_SIZE:], y_axis_noisy[-WINDOW_SIZE:])


    # Save to csv
    df = pd.DataFrame({
    'Time': x_axis[-WINDOW_SIZE:], 
    'Noisy Signal':  adc_noisy_datastream_val, 
    'Clean Signal': adc_clean_datastream_val
    })

    df.to_csv(data_path, mode='a', index=False, header=not pd.io.common.file_exists(data_path))

    # Adjust the x-axis limits to show the time passed
    ax.set_xlim(x_axis[-WINDOW_SIZE], x_axis[-1])
    return line_clean, line_noisy


# Create animation
ani = animation.FuncAnimation(fig, update, interval=50, blit=False)

plt.show()