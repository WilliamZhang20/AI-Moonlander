import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Using Advantage Actor-Critic Model (A2C)

class ActorNet(nn.Module): # Actor network updates policy params θ
    def __init__(self, input_dim, output_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_layer = nn.Linear(64, output_dim)
        self.log_std_layer = nn.Linear(64, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        std = torch.exp(log_std)  # Convert log_std to std
        return mu, std
    
class CriticNet(nn.Module): # computes Q-values
    def __init__(self, input_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_layer = nn.Linear(64, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        value = self.value_layer(x)
        return value
    
def select_action(state, actor, critic):
    mu, std = actor(state)
    action_distr = torch.distributions.Normal(mu, std)
    action_sample = action_distr.sample()
    log_prob = action_distr.log_prob(action_sample).sum(dim=-1)
    return action_sample.numpy(), log_prob

def advantage(reward, value_next, value_current, gamma):
    return (reward + gamma*value_next - value_current) # Q - V, but Q = R + γ*V_next*Prob

def train(env, actor, critic, optimizerA, optimizerC, episodes=3000, gamma=0.99):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, log_prob = select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # compute advantage estimation 
            value_current = critic(state)
            value_next = critic(next_state) if not done else 0
            advantage = advantage(reward, value_next, value_current, gamma)

            actor_loss = -(log_prob * advantage).mean()
            critic_loss = torch.nn.functional.mse_loss(value_current, value_next)

            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()
            state = next_state

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", continuous=True, gravity=-9.8, enable_wind=False, render_mode="rgb_array")
    actor = ActorNet(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)
    critic = CriticNet(input_dim=env.observation_space.shape[0])
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    train(env, actor, critic)