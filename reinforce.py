import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

'''
This neural network applies a Monte Carlo Policy Gradient, aka REINFORCE.
It is policy-based, so its neural network takes input of state, and outputs policy.
It computes gradient of the expected return with respect to the policy to update network parameters.
Updates are made after each training episode, using the reward from each time step to update policy
'''
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1) # normalize to distribution
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    return (returns - returns.mean()) / (returns.std() + 1e-8) # normalize

def train(env, policy, optimizer, episodes=5000, gamma=0.99):
    rewards_history = []
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        while not done:
            state = torch.tensor(state, dtype=torch.float32)
            action_probs = policy(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
        
        rewards = np.clip(rewards, -1, 1)
        returns = compute_returns(rewards, gamma)
        entropy_bonus = -torch.sum(action_probs * torch.log(action_probs))
        loss = -sum(log_prob * G for log_prob, G in zip(log_probs, returns)) - 0.01 * entropy_bonus
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_history.append(sum(rewards))
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Total Reward: {sum(rewards):.2f}")

    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

def record_video(env, policy, output_dir="videos"):
    env = RecordVideo(env, output_dir)
    state, _ = env.reset()
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = policy(state)
        action = torch.argmax(action_probs).item()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    print(f"Video saved in {output_dir}")

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", continuous=False, gravity=-9.8, enable_wind=False, render_mode="rgb_array")
    policy = PolicyNet(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=0.002)
    train(env, policy, optimizer)
    record_video(env, policy)