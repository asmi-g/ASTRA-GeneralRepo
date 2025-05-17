#step 1: creating custom environment by subclassing gym.Env
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pywt

class NoiseReductionEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.step_size = 0.1
        self.threshold_factor = 1.0
        self.prev_reward = 0
        self.reward_history = []
        self.iteration = 0
        self.no_signal = False

        # Dummy init values
        self.clean_signal = np.zeros(10)
        self.raw_signal = np.zeros(10)
        self.filtered_signal = np.zeros(10)
    
    # helper functions
    def set_signal_window(self, clean_signal, noisy_signal):
        self.clean_signal = clean_signal
        self.raw_signal = noisy_signal
        if self.no_signal == True:
            self.clean_signal = clean_signal[self.iteration:self.iteration+10]
            self.raw_signal = noisy_signal[self.iteration:self.iteration+10]
        if np.max(np.abs(self.clean_signal)) > 1000:
            self.clean_signal = self.clean_signal / 1000.0
            self.raw_signal = self.raw_signal / 1000.0

    def apply_filter(self, signal, wavelet='db4', level=1, threshold_factor=1.0):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = threshold_factor * sigma
        coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        return pywt.waverec(coeffs, wavelet)

    def calculate_SNR(self, clean, noisy):
        noise = noisy - clean
        signal_power = np.mean(clean**2)
        noise_power = np.mean(noise**2)
        return 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))

    def reset(self, seed=None, options=None, clean_signal=None, noisy_signal=None):
        self.no_signal = False
        super().reset(seed=seed)
        if clean_signal is None or noisy_signal is None:
            self.no_signal = True
            print("signal not provided; generating random signal")
            self.clean_signal = np.sin(np.linspace(0, 2*np.pi, 5000))
            self.raw_signal = self.clean_signal + np.random.normal(0, 0.3, size=5000)
        else:
            self.clean_signal = clean_signal
            self.raw_signal = noisy_signal
        
        if np.max(np.abs(self.clean_signal)) > 1000:
            self.clean_signal = self.clean_signal / 1000.0
            self.raw_signal = self.raw_signal / 1000.0

        self.threshold_factor = 1.0
        self.filtered_signal = self.apply_filter(self.raw_signal)
        self.prev_reward = 0
        self.reward_history = []
        self.iteration = 0

        snr_raw = self.calculate_SNR(self.clean_signal, self.raw_signal)
        snr_filtered = self.calculate_SNR(self.clean_signal, self.filtered_signal)

        state = np.array([
            self.iteration,
            np.mean(self.clean_signal),
            np.mean(self.raw_signal),
            np.mean(self.filtered_signal),
            self.threshold_factor,
            snr_raw,
            snr_filtered,
            self.prev_reward,
            np.mean(self.reward_history) if self.reward_history else 0.0
        ], dtype=np.float32)
        return state, {}

    def step(self, action):
        self.iteration += 1
        delta = np.clip(float(action[0]), -1, 1)  # continuous action ∈ [-1, 1]
        self.threshold_factor += self.step_size * delta
        self.threshold_factor = np.clip(self.threshold_factor, 0.0, 5.0)

        self.filtered_signal = self.apply_filter(self.raw_signal, threshold_factor=self.threshold_factor)
        snr_raw = self.calculate_SNR(self.clean_signal, self.raw_signal)
        snr_filtered = self.calculate_SNR(self.clean_signal, self.filtered_signal)

        #reward = np.tanh(0.5 * (snr_filtered - snr_raw))
        reward = snr_filtered - snr_raw

        # Positive reward component
        if reward > 0:
            reward += 0.2 * reward  # encourage even modest improvements

        # Penalize for distortion
        bias_penalty = np.abs(np.mean(self.filtered_signal - self.clean_signal))
        reward -= 0.05 * bias_penalty


        # Discourage unnecessary filtering (but not too hard)
        if abs(self.threshold_factor) < 0.01:
            reward -= 0.02


        self.reward_history.append(reward)
        self.prev_reward = reward

        state = np.array([
            self.iteration,
            np.mean(self.clean_signal),
            np.mean(self.raw_signal),
            np.mean(self.filtered_signal),
            self.threshold_factor,
            snr_raw,
            snr_filtered,
            self.prev_reward,
            np.mean(self.reward_history) if self.reward_history else 0.0
        ], dtype=np.float32)

        info = {
            "iteration": self.iteration,
            "clean_signal": self.clean_signal,  # Full window
            "noisy_signal": self.raw_signal,    # Full window
            "filtered_signal": self.filtered_signal,  # Full window
            "threshold_factor": self.threshold_factor,
            "SNR_raw": snr_raw,
            "SNR_filtered": snr_filtered,
            "prev_reward": self.prev_reward,
            "reward_history": self.reward_history
        }

        done = False
        if self.iteration >= 100:
            if len(self.reward_history) > 10:  # Ensure we have enough data
                # Get the last window_size rewards
                recent_rewards = self.reward_history[-10:]

                # Calculate the standard deviation of the recent rewards
                std_dev = np.std(recent_rewards)
                # if self.iteration % 10 == 0:
                #     print(f"Standard deviation over last {10} steps: {std_dev:.4f}")

                # If the standard deviation is below a certain threshold, consider the reward plateaued
                if std_dev < 1e-4:  # Threshold for plateau detection (tune as needed)
                    done = True
                else:
                    done = False
            else:
                done = False  # Not enough data to determine plateau yet

        return state, reward, done, False, info

    def close(self):
        pass
