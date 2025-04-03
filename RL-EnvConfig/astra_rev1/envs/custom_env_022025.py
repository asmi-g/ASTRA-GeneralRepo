#step 1: creating custom environment by subclassing gym.Env
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pywt

class NoiseReductionEnv(gym.Env):
    def __init__(self):
        super(NoiseReductionEnv, self).__init__()

        # Define observation space: [time, clean_signal, raw_signal, filtered_signal, threshold_factor, SNR_raw, SNR_filtered, prev_reward, inuse_operation]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: repeat operation, 1: use opposite operation

        self.time = 0
        self.threshold_factor = 1
        self.clean_signal = np.zeros(10)
        self.raw_signal = np.zeros(10)
        self.filtered_signal = np.zeros(10)
        self.SNR_raw = 0
        self.SNR_filtered = 0
        self.prev_reward = 0
        self.iteration = 0
        self.reward_history = []
        
        self.inuse_operation = 1 # Default: Increase
        self.step_size = 0.1  # Dynamic step size
    
    #helper functions
    def set_threshold_factor(self, new_threshold_factor):
        """Sets the threshold factor from outside the environment."""
        self.threshold_factor = new_threshold_factor

    def apply_filter(self, signal, wavelet='db4', level=1, threshold_factor=0.5):
        # threshold_factor is modifiable
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = threshold_factor * sigma
        coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        return pywt.waverec(coeffs, wavelet)
    
    def calculate_SNR(self, clean_signal, noisy_signal):
        if np.mean(noisy_signal) == 0:
            return np.nan
        noise = noisy_signal - clean_signal  # Extract noise component
        signal_power = np.mean(clean_signal**2)  # Power of the clean signal
        noise_power = np.mean(noise**2)  # Power of the noise
        return 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))  # in dB

    
    def set_signal(self, clean_signal, noisy_signal):
        """Manually update the clean and noisy signal"""
        self.clean_signal = clean_signal
        self.raw_signal = noisy_signal  # Assuming 'raw_signal' stores the noisy input
    
    def reset(self, seed=None, options=None, clean_signal=None, noisy_signal=None):
        super().reset(seed=seed)  # Ensure the superclass method is called
        
        if clean_signal is None or noisy_signal is None:
            # Generate default synthetic signals if none are provided
            self.clean_signal = np.random.randn(100)  # Example: 100-sample random signal
            self.raw_signal = self.clean_signal + np.random.normal(0, 0.1, size=100)
        else:
            self.clean_signal = clean_signal
            self.raw_signal = noisy_signal

        self.current_index = 0  # Start at the beginning of the signal
        self.filtered_signal = self.apply_filter(self.raw_signal)

        self.SNR_raw = self.calculate_SNR(self.clean_signal, self.raw_signal)
        self.SNR_filtered = self.calculate_SNR(self.clean_signal, self.filtered_signal)

        self.iteration = 0
        self.reward_history = []
        self.inuse_operation = 1
        
        state = np.array([
            self.time, np.mean(self.clean_signal), np.mean(self.raw_signal),
            np.mean(self.filtered_signal), self.threshold_factor,
            self.SNR_raw, self.SNR_filtered, 0, self.inuse_operation
        ], dtype=np.float32)

        return state, {}

    def step(self, action):
        self.iteration += 1

        # Modify filter parameters
        if action == 0: # repeat operation
            if self.inuse_operation == 1:
                self.threshold_factor = self.threshold_factor + self.step_size
            elif self.inuse_operation == -1:
                self.threshold_factor = self.threshold_factor - self.step_size
        elif action == 1: #change operation
            if self.inuse_operation == 1:
                self.threshold_factor = self.threshold_factor - self.step_size
                self.inuse_operation = -1
            elif self.inuse_operation == -1:
                self.threshold_factor = self.threshold_factor + self.step_size
                self.inuse_operation = 1
        #elif action == 2:
            #self.threshold_factor = self.threshold_factor - self.step_size

        self.threshold_factor = min(5, self.threshold_factor)
        self.threshold_factor = max(-5, self.threshold_factor)

        # Apply filtering using new values
        self.filtered_signal = self.apply_filter(signal = self.raw_signal, threshold_factor = self.threshold_factor)

        # Compute updated SNR
        self.SNR_raw = self.calculate_SNR(self.clean_signal, self.raw_signal)
        self.SNR_filtered = self.calculate_SNR(self.clean_signal, self.filtered_signal)

        reward = np.log1p(max(self.SNR_filtered - self.SNR_raw, 0))
        #reward = np.square(np.subtract(self.SNR_filtered, self.SNR_raw)).mean()
        #reward = max(0, -(np.mean(self.filtered_signal) - np.mean(self.clean_signal)) + 1e-10)

        if np.isnan(reward):
            reward = -1
        self.reward_history.append(reward)
        self.prev_reward = reward

        done = False
        if self.iteration >= 100:
            if len(self.reward_history) > 10:  # Ensure we have enough data
                # Get the last window_size rewards
                recent_rewards = self.reward_history[-10:]

                # Calculate the standard deviation of the recent rewards
                std_dev = np.std(recent_rewards)

                #print(f"Standard deviation over last {10} steps: {std_dev:.4f}")

                # If the standard deviation is below a certain threshold, consider the reward plateaued
                if std_dev < 1e-2:  # Threshold for plateau detection (tune as needed)
                    done = True
                else:
                    done = False
            else:
                done = False  # Not enough data to determine plateau yet
        '''
        if np.mean(self.filtered_signal) == 0:
            print(self.raw_signal)
            print(self.filtered_signal)
        '''
        state = np.array([
            self.time, np.mean(self.clean_signal), np.mean(self.raw_signal),
            np.mean(self.filtered_signal), self.threshold_factor, self.SNR_raw, self.SNR_filtered, self.prev_reward, self.inuse_operation
        ], dtype=np.float32)

        info = {
            "time": self.time,
            "clean_signal": self.clean_signal,  # Full window
            "noisy_signal": self.raw_signal,    # Full window
            "filtered_signal": self.filtered_signal,  # Full window
            "threshold_factor": self.threshold_factor,
            "SNR_raw": self.SNR_raw,
            "SNR_filtered": self.SNR_filtered,
            "prev_reward": self.prev_reward,
            "inuse_operation": self.inuse_operation
        }

        return state, reward, done, False, info  # False = 'truncated', {} = 'info'


    def render(self, mode='human'):
        print(f"Iteration: {self.iteration}, Threshold: {self.threshold_factor:.6f}, "f"SNR Raw: {self.SNR_raw:.2f}, SNR Filtered: {self.SNR_filtered:.2f}")