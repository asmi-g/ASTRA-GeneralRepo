# TO DO:
# - Integrate cosmic noise model
# - Add in the ability to save ADC streaming data to csv 

# CHANGES:
# - X plot now updates w/ time
# - Signal data now converted to ADC values
# - "Real-time" ADC stream of data added (as opposed to an array of values)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# Add in time counter, to track and update in plot
current_time = 0
x_axis = []
y_axis = []

# ADC Stream Data Buffer
adc_stream = np.zeros(WINDOW_SIZE)  # Initialize ADC value buffer


# Initialize figure
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Noisy Signal")
ax.set_ylim(-1.5, 1.5) # Adjust based on signal strength
ax.set_xlim(0, WINDOW_SIZE / F_SAMPLING)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")


# Fixed plot legend
legend = ax.legend(loc='upper right')
legend.set_draggable(False)  # Prevents moving when updated

ax.set_title("Real-Time Noisy Signal")

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

def update(frame):
    # Track the time as frames are updated
    global current_time, x_axis, y_axis, adc_stream

    # Generate new signal and noise
    signal = np.sin(2 * np.pi * F_SIGNAL * (frame + np.arange(WINDOW_SIZE)) / F_SAMPLING)
    noise = np.random.normal(0, np.sqrt(NOISE_POWER), WINDOW_SIZE)
    noisy_signal = signal + noise

    # ADC ARRAY: PLOT
    # Convert to ADC values
    adc_values = convert_ADC_array(noisy_signal)
    # Convert ADC to voltage
    quantized_signal = (adc_values / (2**ADC_BITS - 1)) * V_REF - V_REF / 2
    
    # ADC DATA STREAM VALUE: SINGLE SAMPLE
    sample = noisy_signal[frame % WINDOW_SIZE]
    # Convert to ADC value
    adc_datastream_val = convert_ADC_stream(sample)
    print("Time", current_time/10000, "Streaming ADC Value:", adc_datastream_val)
    # Convert ADC value to voltage 
    quantized_datastream_signal = (adc_datastream_val / (2**ADC_BITS - 1)) * V_REF - V_REF / 2
    # Shift data buffer
    adc_stream = np.roll(adc_stream, -1)  # Shift left by one
    adc_stream[-1] = quantized_datastream_signal  # Add the new value


    # Update the time vector to shift with the data
    new_time = (current_time + np.arange(WINDOW_SIZE)) / F_SAMPLING
    x_axis.extend(new_time)
    y_axis.extend(quantized_signal)
    current_time += WINDOW_SIZE  # Move forward in time

    # Update plot data
    line.set_data(x_axis[-WINDOW_SIZE:],y_axis[-WINDOW_SIZE:])

    # Adjust the x-axis limits to show the time passed
    ax.set_xlim(x_axis[-WINDOW_SIZE], x_axis[-1])
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, interval=50, blit=False)

plt.show()