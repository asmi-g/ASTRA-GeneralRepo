
import gym
import random
import pandas
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# Create environment, define states, actions
env = gym.make("CartPole-v1")

states = env.observation_space.shape[0]
actions = env.action_space.n


# Create model, define parameters for NN model
model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))


# Create DQN Agent
agent = DQNAgent(
    model = model,
    memory = SequentialMemory (limit=50000, window_length=1),
    policy = BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

# Compile agent w/ Adam optimizer, train
agent.compile(Adam(lr=0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

# Evaluate model, test
results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

# Close environment
env.close()


'''episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0,1])
        _, reward, done,  info = env.step(action)
        score += reward
        env.render()

    print(f"Episode {episode}, Score: {score}")

env.close()'''