import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters

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

# Discrete time points vector
t = np.arange(0, WINDOW_SIZE) / F_SAMPLING

# Initialize figure
fig, ax = plt.subplots()
line, = ax.plot(t, np.zeros_like(t), label="Noisy Signal")
ax.set_ylim(-1.5, 1.5) # Adjust based on signal strength
ax.set_xlim(0, WINDOW_SIZE / F_SAMPLING)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.set_title("Real-Time Noisy Signal")

def update(frame):
    # Generate new signal and noise
    signal = np.sin(2 * np.pi * F_SIGNAL * (frame + np.arange(WINDOW_SIZE)) / F_SAMPLING)
    noise = np.random.normal(0, np.sqrt(NOISE_POWER), WINDOW_SIZE)
    noisy_signal = signal + noise
    print(frame)

    # Update plot data
    line.set_ydata(noisy_signal)
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, interval=50, blit=True)

plt.show()