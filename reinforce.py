import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
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
        
        returns = compute_returns(rewards, gamma)
        loss = -sum(log_prob * G for log_prob, G in zip(log_probs, returns))