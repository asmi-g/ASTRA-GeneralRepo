from astra_rev1.envs import NoiseReductionEnv
from agent import DQNAgent
import numpy as np

def train_agent(episodes=1000, max_steps=100):
    env = NoiseReductionEnv()
    agent = DQNAgent()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Store transition and train
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Optional: Save model periodically
        if episode % 100 == 0:
            torch.save(agent.model.state_dict(), f'model_checkpoint_{episode}.pth')

if __name__ == "__main__":
    train_agent()
