# step 2: register environment
from gym.envs.registration import register

register(
    id="NoiseReductionEnv-v0",
    entry_point="astra_rev1.envs.custom_env_022025:NoiseReductionEnv"
)

# Optional: Import the environment for easier access
from .custom_env_022025 import NoiseReductionEnv
