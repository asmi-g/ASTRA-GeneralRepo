# train_noise_reduction.py
import time
start_time = time.time()
import os
import numpy as np
import random
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from astra_rev1.envs import NoiseReductionEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

env = Monitor(NoiseReductionEnv())
check_env(env, warn=True)

seed = 42
np.random.seed(seed)

log_path = os.path.join('Training', 'Logs')
os.makedirs(log_path, exist_ok=True)
os.makedirs("models", exist_ok=True)

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
eval_callback = EvalCallback(
    env,
    best_model_save_path='models/best_model',
    log_path='Training/Logs',
    eval_freq=1000,
    deterministic=True,
    render=False
)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='models/', name_prefix='sac_checkpoint')

model.learn(total_timesteps=100000, callback=[eval_callback, checkpoint_callback])

model.save("models/sac_noise_reduction")
print("Model saved to 'models/sac_noise_reduction'")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)
print(f"Mean reward: {mean_reward} ± {std_reward}")

episode_rewards = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=True)
print("Evaluation rewards over episodes: ", episode_rewards)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal runtime: {elapsed_time:.2f} seconds")





# 
# def generate_synthetic_signal(signal_type=None, noise_level=None, length=100):
#     t = np.linspace(0, 1, length)
#     signal_type = signal_type or random.choice(["sine", "square", "sawtooth", "random"])
#     noise_level = noise_level if noise_level is not None else np.random.uniform(0.2, 0.5)

#     if signal_type == "sine":
#         clean = np.sin(2 * np.pi * 5 * t)
#     elif signal_type == "square":
#         clean = np.sign(np.sin(2 * np.pi * 5 * t))
#     elif signal_type == "sawtooth":
#         clean = 2 * (t * 5 % 1) - 1
#     elif signal_type == "random":
#         clean = np.random.uniform(-1, 1, size=length)
#     else:
#         raise ValueError("Unknown signal type!")

#     noisy = clean + np.random.normal(0, noise_level, size=length)
#     return clean, noisy


# # Initialize model
# model = SAC(
#     "MlpPolicy",
#     env,
#     ent_coef="auto",  # 🔄 CHANGED: Encourage exploration via entropy bonus
#     action_noise=action_noise,  # 🔄 CHANGED: Added action noise
#     learning_rate=1e-4,
#     buffer_size=100000,
#     batch_size=128,
#     tau=0.005,
#     gamma=0.99,
#     train_freq=1,
#     gradient_steps=4,  # 🔄 CHANGED: More frequent updates per step
#     verbose=1,
#     tensorboard_log="logs/tensorboard"
# )
# from stable_baselines3.common.logger import configure

# model._logger = configure(folder="logs/tensorboard", format_strings=["stdout", "tensorboard"])
# model._current_progress_remaining = 1.0  # Full training progress at start


# WINDOW_SIZE = 10
# SIGNAL_LENGTH = 100
# TOTAL_EPISODES = 5000

# # Training Loop
# for episode in range(TOTAL_EPISODES):
#     clean_full, noisy_full = generate_synthetic_signal(length=SIGNAL_LENGTH)

#     for i in range(SIGNAL_LENGTH - WINDOW_SIZE):
#         clean_window = clean_full[i:i + WINDOW_SIZE]
#         noisy_window = noisy_full[i:i + WINDOW_SIZE]

#         if i == 0:
#             obs, _ = env.reset(clean_signal=clean_window, noisy_signal=noisy_window)
#         else:
#             env.set_signal_window(clean_window, noisy_window)

#         action, _ = model.predict(obs, deterministic=False)
#         next_obs, reward, done, _, info = env.step(action)

#         # Manually store transition
#         model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
#         model.train(batch_size=model.batch_size, gradient_steps=1)

#         obs = next_obs

#     if episode % 100 == 0:
#         print(f"Step {episode} | Action: {action[0]:.4f} | Threshold: {env.threshold_factor:.3f} | Reward: {reward:.3f}") 


# # Callback for custom logging
# # callback = NoiseReductionLogger()
# # TOTAL_TIMESTEPS = 1000
# # model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# model.save("models/sac_noise_reduction")
# print("Model saved to 'models/sac_noise_reduction'")

# # # Training parameters
# # WINDOW_SIZE = 10
# # SIGNAL_LENGTH = 100
# # TOTAL_EPISODES = 1000

# # # 🔄 CHANGED: Use sliding window over new signal every episode
# # for episode in range(TOTAL_EPISODES):
# #     clean_full, noisy_full = generate_synthetic_signal(length=SIGNAL_LENGTH)
# #     total_reward = 0

# #     for i in range(SIGNAL_LENGTH - WINDOW_SIZE):
# #         clean_window = clean_full[i:i + WINDOW_SIZE]
# #         noisy_window = noisy_full[i:i + WINDOW_SIZE]

# #         if i == 0:
# #             obs, _ = env.reset(clean_signal=clean_window, noisy_signal=noisy_window)
# #         else:
# #             env.set_signal_window(clean_window, noisy_window)

# #         action, _ = model.predict(obs)
# #         obs, reward, done, _, info = env.step(action)
# #         total_reward += reward

# #     # 🔄 CHANGED: Better logging for debugging
# #     if episode % 100 == 0:
# #         print(f"Episode {episode} | Total Reward: {total_reward:.2f} "
# #               f"| Threshold: {info['threshold_factor']:.3f} | Action: {action}")

# # Save the model
