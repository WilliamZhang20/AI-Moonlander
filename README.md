# AI Moonlander

In this repository, I apply reinforcement learning (RL) to make an agent controlling a spacecraft land on the moon in a simulation environment!

The agent's goal is to land the lunar module between the yellow flags, avoid crashing into the rocks around, and complete the landing as fast as possible!

To do this, it has come up with a pattern for the decisions, or *policies*, it will take at each time throughout the landing. At any time, it can do nothing, or fire the left, bottom, or right rocket engines, so 4 possible actions, at least in the discrete action space environment.

Hence, the agent will simulate many rounds of landings using various action policies. Through its observations of unsuccessful and successful options, and searches for better policies, it will settle on the optimal policy.

## Model Implementation

The algorithms implemented so far for the moonlander are the Deep-Q Network (DQN), REINFORCE policy gradient, Advantage Actor-Critic (A2C) policy gradient.

The DQN algorithm, implemented in TensorFlow, uses a deep neural network to predict the value of taking each of the 4 actions, or essentially a state to action-value mapping. The primary part of this is running various training samples using the current policy and learning from successes failures to get better.
- By knowing an optimal mapping, the moonlander agent picks the action with the highest value to always get the highest possible reward. To do this, it uses a dual-network architecture containing a target network and an online network, as the true optimal state is not known.
- Each training sample is taken from a large data structure of replay memory to learn from both present and past episodes, and it uses the Bellman Equation to determine Mean-Squared-Error loss between the applied online network and temporary target network prediction.
- For further training stability, the target network is periodically updated every few training time steps to match the online network using a soft-update convex combination formula for gradual updates.
- Additionally, to explore actions at the start, it uses an epsilon greedy decay to balance exploration and exploitation. At each episode, there will be a probability of $$\epsilon$$ that the agent takes a random action in training (it never happens in inference!), which should intuitively be higher at the start (when it knows less about the environment) and lower later on (as it knows more).

The REINFORCE policy gradient algorithm, implemented in PyTorch, uses a single policy network that takes the state of the agent in the environment, and outputs a distribution over actions. It updates parameters using present rewards to maximize expected cumulative reward in stochastic gradient ascent.
- It takes actions using the policy network, and adjusts on gradient of discounted reward vs baseline as well as policy parameters, based on the formula of the policy gradient theorem. It hopes to increase probability of actions leading to higher reward. This procedure is also known as being on-policy. 
- It also uses Monte-Carlo methods, and updates the policy only after completing an entire episode, rather than after each individual time step. This helps to reduce variance.
- To make the algorithm more stable, I introduced a baseline function with an exponentially moving average of agent rewards. This baseline helps reduce the variance of the gradient estimate by subtracting a value from the reward before applying the update, i.e. computing the advantage function. The baseline does not change the expected value of the gradient but makes the updates less noisy.

The A2C algorithm, also implemented in PyTorch, is *also* an on-policy method using two neural networks: an actor and a critic. The actor selects actions based on the current policy, while the critic evaluates the selected actions by estimating the state value function.
- The actor network updates policy using an *advantage function* $$A^{\pi}(s, a)$$ which is the difference between state-action return $$Q(s, a)$$ and state value $$V(s)$$, where state value is given by the critic network.
- The critic network updates parameters using Temporal Difference (TD) mean-squared-error of difference between estimated value and a bootstrapped estimation of the total cumulative reward.
- In my implementation, I had the actor and critic in different neural networks to ensure flexibility, guaranteed and independent behavior, as well as easier convergence. However, some implementations have them share parts of their networks for efficiency and simplicity.

I implemented the Proximal Policy Optimization (PPO) algorithm in PyTorch to use clipped [surrogate](https://www.mathworks.com/help/gads/surrogate-optimization-algorithm.html) objectives to ensure that updates do not make the model worse. It is the go-to algorithm of today, combining both replay memory and actor-critic policy updates to make training progression robust & stable. While Trust-Region Policy Optimization (TRPO) uses similar techniques, PPO is computationally cheaper, hence its strong popularity. 

Additionally, I achieved strong model performance in continuous action spaces, where the agent can control the strength of each engine's power input, by applying the Deep Deterministic Policy Gradient (DDPG) algorithm in PyTorch.

## Demonstration

Demo of a successful landing by a model trained using the REINFORCE vanilla policy gradient method:

https://github.com/user-attachments/assets/1fe87cd9-be23-47b9-9dbb-ab5235fb352c

In comparison, the PPO landing is much smoother (and faster!), thanks to its stability and variance minimization:

https://github.com/user-attachments/assets/7d0f386d-988c-4871-aaea-d9dfb8e056f8
