import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Signal Parameters
F_SAMPLING = 1e4  # 10 kHz sampling frequency
F_SIGNAL = 100  # 100 Hz signal frequency
NOISE_POWER = 0.1
WINDOW_SIZE = 500  # Number of points visible on the plot

# ADC Parameters
ADC_BITS = 12
V_REF = 5

# Time and Data Buffers
t = np.arange(WINDOW_SIZE) / F_SAMPLING  # Initial time window
adc_stream = np.zeros(WINDOW_SIZE)  # Initialize ADC value buffer

# Initialize Figure
fig, ax = plt.subplots()
line, = ax.plot(t, adc_stream, label="Streaming ADC Signal")
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(0, WINDOW_SIZE / F_SAMPLING)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Real-Time Streaming ADC Values")
legend = ax.legend(loc='upper right')
legend.set_draggable(False)

# ADC Conversion Function
def convert_ADC(signal):
    normalized_signal = (signal + V_REF / 2) / V_REF
    adc_value = np.clip(np.round(normalized_signal * (2**ADC_BITS - 1)), 0, 2**ADC_BITS - 1)
    return int(adc_value)

# Update Function for Streaming
current_sample = 0

def update(frame):
    global current_sample, adc_stream, t

    # Generate one new noisy signal value
    time_point = current_sample / F_SAMPLING
    signal = np.sin(2 * np.pi * F_SIGNAL * time_point)
    noise = np.random.normal(0, np.sqrt(NOISE_POWER))
    noisy_signal = signal + noise

    # Convert to ADC value
    adc_value = convert_ADC(noisy_signal)
    print("Streaming ADC Value:", adc_value)

    # Convert ADC value back to voltage for plotting
    quantized_signal = (adc_value / (2**ADC_BITS - 1)) * V_REF - V_REF / 2

    # Shift data buffer and time
    adc_stream = np.roll(adc_stream, -1)  # Shift left by one
    adc_stream[-1] = quantized_signal  # Add the new value

    t = np.roll(t, -1)
    t[-1] = time_point  # Update time for the newest sample

    # Update plot
    line.set_data(t, adc_stream)
    ax.set_xlim(t[0], t[-1])  # Keep x-axis moving forward

    current_sample += 1  # Increment the sample count
    return line,

# Create Animation
ani = animation.FuncAnimation(fig, update, interval=50, blit=True)

plt.show()