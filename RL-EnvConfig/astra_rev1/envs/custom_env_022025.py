#step 1: creating custom environment by subclassing gym.Env
import gym
import numpy as np
from gym import spaces
import scipy.signal
from scipy.signal import butter, filtfilt #for filDeiter

class NoiseReductionEnv(gym.Env):
    def __init__(self):
        super(NoiseReductionEnv, self).__init__()
        # Define state space: [time, raw_signal, filtered_signal, C, L, R, SNR_raw, SNR_filtered, prev_reward]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: Increase, 1: Decrease & Rotate

        self.time = 0
        self.C = 0.000001  # 1 uF; initial value
        self.L = 0.000001  # 1 uH; initial value
        self.R = 1000 # 1 kOhm; initial value
        self.raw_signal = self.generate_noisy_signal() # in practice, will be connect to signal simulation or real-time signal feed
        self.filtered_signal = self.apply_filter(self.raw_signal) # in practice, signal will be taken directly from output of DSP
        self.SNR_raw = self.calculate_SNR(self.raw_signal) # verify calculation
        self.SNR_filtered = self.calculate_SNR(self.filtered_signal)
        self.prev_reward = 0
        self.iteration = 0
        self.reward_history = []
        
        self.var_map = {'C': self.C, 'L': self.L, 'R': self.R}
        self.change_var = 'C'  # Default to capacitance
        self.param_index = 0  # Tracks which parameter is being modified
        self.step_size = 0.01 * self.var_map[self.change_var]  # Dynamic step size
    
    #helper functions
    def generate_noisy_signal(self):
        """Simulates a synthetic noisy signal"""
        signal_length = 100
        pure_signal = np.sin(np.linspace(0, 2 * np.pi, signal_length))  # Pure sine wave
        noise = np.random.normal(0, 0.5, signal_length)  # Gaussian noise
        return pure_signal + noise

    def apply_filter(self, signal, cutoff_freq=50, fs=1000, order=2):
        """
        Applies a Butterworth low-pass filter to the input signal.
        
<<<<<<< Updated upstream
        :param signal: The input signal array
        :param cutoff_freq: Cutoff frequency in Hz
        :param fs: Sampling frequency in Hz
        :param order: Order of the filter (higher = sharper roll-off)
        :return: Filtered signal
        """
        nyquist = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff_freq / nyquist  # Normalize frequency
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)  # Zero-phase filtering

    def calculate_SNR(self, noisy_signal, fs = 1.0): # TODO: verify
        """Estimate SNR using Power Spectral Density (PSD) method."""
        freqs, psd = scipy.signal.welch(noisy_signal, fs=fs, nperseg=len(noisy_signal)//8)  # Compute PSD
        # Assume noise power is the average power in high frequencies
        noise_power = np.mean(psd[len(psd)//2:])  # Upper half of frequencies
        signal_power = np.mean(psd)  # Total power
        return 10 * np.log10(signal_power / (noise_power + 1e-10)) # SNR in dB
    
    def reset(self):
        """Resets the environment to an initial state"""
        self.time = 0
        self.iteration = 0
        self.C = 0.000001  # 1 uF; initial value
        self.L = 0.000001  # 1 uH; initial value
        self.R = 1000 # 1 kOhm; initial value
        self.raw_signal = self.generate_noisy_signal()
=======
        if (self.clean_signal.size > 0) and (np.max(np.abs(self.clean_signal))) > 1000:
            self.clean_signal = self.clean_signal / 1000.0
            self.raw_signal = self.raw_signal / 1000.0

        self.threshold_factor = 1.0
>>>>>>> Stashed changes
        self.filtered_signal = self.apply_filter(self.raw_signal)
        self.SNR_raw = self.calculate_SNR(self.raw_signal)
        self.SNR_filtered = self.calculate_SNR(self.filtered_signal)
        self.prev_reward = 0
        self.reward_history = []
        self.var_map = {'C': self.C, 'L': self.L, 'R': self.R}  # Update after reset
        self.change_var = 'C'  # Default to capacitance
        self.param_index = 0  # Tracks which parameter is being modified
        self.step_size = 0.01 * self.var_map[self.change_var]  # Dynamic step size to account for differences between units

        state = np.array([self.time, np.mean(self.raw_signal), np.mean(self.filtered_signal), self.C, self.L, self.R, self.SNR_raw, self.SNR_filtered, self.prev_reward], dtype=np.float32)
        return state

    def step(self, action):
        self.iteration += 1
        # actions
        if action == 0:  # Increase change_var
            self.var_map[self.change_var] += self.step_size
        elif action == 1:  # Decrease change_var, then rotate
            self.var_map[self.change_var] -= self.step_size  # Reduce current param; this is still the previous step_size value
            # Rotate: C -> L -> R -> C
            self.param_index = (self.param_index + 1) % 3
            self.change_var = ['C', 'L', 'R'][self.param_index]
            self.step_size = 0.01 * self.var_map[self.change_var]
            self.var_map[self.change_var] += self.step_size

        # SNR calculation
        self.filtered_signal = self.apply_filter(self.raw_signal)
        self.SNR_filtered = self.calculate_SNR(self.filtered_signal)
        reward = self.SNR_filtered - self.SNR_raw
        self.reward_history.append(reward)
        self.prev_reward = reward
        # Termination condition: if ratio of last 10 rewards < 1% --> plateau
        done = False
        if self.iteration >= 100:
            recent_rewards = self.reward_history[-10:]
            if len(recent_rewards) >= 10 and np.abs(np.mean(np.diff(recent_rewards))) < 1.0:
                done = True

        # update individual variable (self.C, self.L, or self.R)
        if self.change_var == 'C':
            self.C = self.var_map['C']
        elif self.change_var == 'L':
            self.L = self.var_map['L']
        else:
            self.R = self.var_map['R']

        # New state representation
        state = np.array([
            self.time, np.mean(self.raw_signal), np.mean(self.filtered_signal),
            self.C, self.L, self.R,  # These are the actual state variables now
            self.SNR_raw, self.SNR_filtered, self.prev_reward
        ], dtype=np.float32)

        return state, reward, done, {}

    def render(self, mode='human'):
        """Optional: Print current state"""
        print(f"Iteration: {self.iteration}, C: {self.C:.2f}, L: {self.L:.2f}, R: {self.R:.2f}, SNR: {self.SNR_filtered:.2f}")

