# AI Moonlander

In this repository, I apply reinforcement learning (RL) to make an agent controlling a spacecraft land on the moon in a simulation environment!

The agent's goal is to land the lunar module between the yellow flags, avoid crashing into the rocks around, and complete the landing as fast as possible!

To do this, it has come up with a pattern for the decisions, or *policies*, it will take at each time throughout the landing. At any time, it can do nothing, or fire the left, bottom, or right rocket engines.

Hence, the agent will simulate many rounds of landings using various action policies. Through its observations of unsuccessful ones and searches for better policies, it will settle on the optimal policy.

The RL models are designed using the PyTorch framework, based on various [policy gradient](https://arxiv.org/pdf/2401.13662) methods, such as REINFORCE, Advantage Actor-Critic (A2C), Deep Deterministic Policy Gradient (DDPG), and Proximal Policy Optimization (PPO). 

Demo of a successful landing by a model trained using the REINFORCE policy gradient method:

https://github.com/user-attachments/assets/1fe87cd9-be23-47b9-9dbb-ab5235fb352c
