import gym
import numpy as np
from astra_rev1.envs import NoiseReductionEnv 
# need to look into including this line a bit more because from what I've read once you register the env you shouldn't need this but this script doesn't work without this

env = gym.make("NoiseReductionEnv-v0")

state = env.reset()
print("Initial state:", state)

action = 0
curr_reward = 0
prev_reward = 0
for i in range(10):  # Take 10 actions
    next_state, curr_reward, done, _ = env.step(action)
    print(f"Step {i+1} | Action: {action} | Reward: {curr_reward:.4f} | Done: {done}")

    if prev_reward > curr_reward:
        action = 1
    else:
        action = 0
    
    prev_reward = curr_reward

    if done:
        print("Termination condition met. Resetting environment.")
        state = env.reset()
        break

env.close()
