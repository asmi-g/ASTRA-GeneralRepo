import gymnasium as gym
from gymnasium import envs
from stable_baselines3.common.env_checker import check_env
from astra_rev1.envs import NoiseReductionEnv

# Register your custom environment
env = NoiseReductionEnv()
check_env(env)  # This will run several tests on your environment
