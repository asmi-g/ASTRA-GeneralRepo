# callbacks.py

from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class NoiseReductionLogger(BaseCallback):
    """
    Custom callback for logging threshold factors and rewards during training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.thresholds = []
        self.rewards = []

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if "threshold_factor" in info:
            self.thresholds.append(info["threshold_factor"])
        self.rewards.append(self.locals["rewards"][0])
        return True

    def _on_training_end(self) -> None:
        # Plotting the threshold factors and rewards
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.thresholds)
        plt.title("Threshold Factor Over Time")
        plt.xlabel("Step")
        plt.ylabel("Threshold Factor")

        plt.subplot(1, 2, 2)
        plt.plot(self.rewards)
        plt.title("Reward Over Time")
        plt.xlabel("Step")
        plt.ylabel("Reward")

        plt.tight_layout()
        plt.savefig("logs/training_metrics.png")
        plt.close()