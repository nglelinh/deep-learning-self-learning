---
layout: post
title: 20-01 Reinforcement Learning Fundamentals
chapter: '20'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter20
lesson_type: required
---

# Reinforcement Learning: Learning from Interaction

![Reinforcement Learning Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/800px-Reinforcement_learning_diagram.svg.png)
*Hình ảnh: Sơ đồ Reinforcement Learning với agent, environment, actions và rewards. Nguồn: Wikimedia Commons*

## 1. Concept Overview

Reinforcement learning represents a fundamentally different learning paradigm than the supervised and unsupervised learning we've studied. Instead of learning from labeled examples or discovering patterns in unlabeled data, RL agents learn by interacting with an environment, taking actions, observing outcomes, and receiving rewards. The agent's goal is to discover a policy—a strategy for choosing actions—that maximizes cumulative reward over time. This trial-and-error learning with delayed rewards mirrors how humans and animals learn many skills: we try actions, experience consequences, and gradually improve our behavior to achieve desired outcomes.

Understanding RL requires appreciating what makes it challenging compared to supervised learning. In supervised learning, we have correct answers for every input—the model learns from examples of optimal behavior. In RL, we only have rewards indicating how good outcomes were, not which specific actions were optimal. The agent must explore different actions to discover which lead to high rewards, creating an exploration-exploitation tradeoff: should we exploit known good actions or explore potentially better alternatives? Moreover, rewards are often delayed—an action's consequences might not be apparent until many steps later (in chess, moves early in the game affect victory or defeat much later). Credit assignment becomes challenging: which of the many actions taken contributed to the final reward?

The mathematical framework of Markov Decision Processes provides elegant formalization of sequential decision-making under uncertainty. States represent the environment's configuration, actions are choices available to the agent, transitions describe how actions change states (potentially stochastically), and rewards provide learning signal. The Markov property—that the future depends only on the present state, not the full history—simplifies analysis while being approximately true for many real-world problems when states are chosen appropriately. Value functions quantify the expected future reward from states or state-action pairs, providing targets for learning. Policies map states to actions, with the optimal policy selecting actions maximizing expected cumulative reward.

The connection to deep learning through Deep Reinforcement Learning enables RL to scale to high-dimensional state spaces like images and large action spaces. Neural networks approximate value functions or policies, learning from experience through gradient descent. This combination has achieved remarkable success: AlphaGo mastering Go through self-play, Atari game playing from pixels, robotic manipulation learning from trial and error, and sophisticated language model alignment through reinforcement learning from human feedback (RLHF). Understanding RL fundamentals provides foundation for these deep RL methods covered in the next chapter.

## 2. Mathematical Foundation

A Markov Decision Process (MDP) is defined by the tuple $$(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$:

- $$\mathcal{S}$$: State space (all possible environment configurations)
- $$\mathcal{A}$$: Action space (choices available to agent)
- $$\mathcal{P}$$: Transition function $$P(s_{t+1}|s_t, a_t)$$ (dynamics)
- $$\mathcal{R}$$: Reward function $$R(s_t, a_t, s_{t+1})$$ (feedback)
- $$\gamma \in [0,1)$$: Discount factor (balancing immediate vs future rewards)

The agent follows a policy $$\pi(a|s)$$—probability of action $$a$$ in state $$s$$. The goal is finding optimal policy $$\pi^*$$ maximizing expected return:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^\infty \gamma^t r_t\right]$$

where $$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$$ is a trajectory.

### Value Functions

The state value function quantifies expected return starting from state $$s$$:

$$V^\pi(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]$$

The action-value function (Q-function) includes the first action:

$$Q^\pi(s,a) = \mathbb{E}_{\pi}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, a_0 = a\right]$$

These satisfy Bellman equations expressing recursive relationships:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]$$

The optimal value functions satisfy Bellman optimality equations:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')]$$

The optimal policy is greedy with respect to $$Q^*$$: $$\pi^*(s) = \arg\max_a Q^*(s,a)$$.

### Q-Learning

Q-learning learns $$Q^*$$ without knowing transition probabilities through temporal difference learning:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

The update uses observed reward $$r_t$$ and estimated future value $$\max_{a'} Q(s_{t+1}, a')$$ to improve current estimate $$Q(s_t, a_t)$$. This bootstrapping—using one estimate to improve another—enables learning from experience without environment model.

## 3. Example / Intuition

Consider training an agent to play a simple game: navigate a 5×5 grid to reach a goal. States are positions (25 states), actions are {up, down, left, right}, reward is +10 at goal, -1 elsewhere (encouraging reaching goal quickly).

Initially, Q-values are random. The agent starts at (0,0), takes random actions. After wandering, it accidentally reaches goal at (4,4), receiving +10 reward. Q-learning updates:

$$Q((4,3), \text{right}) \leftarrow Q((4,3), \text{right}) + \alpha[10 + 0 - Q((4,3), \text{right})]$$

The Q-value for "go right from position next to goal" increases. Next time the agent reaches (4,3), it's more likely to go right (if using ε-greedy policy based on Q-values).

After more episodes, values propagate backward. Q((4,2), right) increases because it leads to (4,3) which now has high Q-value. Eventually, optimal Q-values form gradient pointing toward goal from every state, and the agent learns to navigate directly to goal from any starting position.

## 4. Code Snippet

```python
import numpy as np

class GridWorld:
    """Simple grid environment for RL demonstration"""
    
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)
        self.state = (0, 0)
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """Execute action, return (next_state, reward, done)"""
        actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up,down,left,right
        dx, dy = actions[action]
        
        x, y = self.state
        new_x = max(0, min(self.size-1, x + dx))
        new_y = max(0, min(self.size-1, y + dy))
        
        self.state = (new_x, new_y)
        reward = 10 if self.state == self.goal else -1
        done = self.state == self.goal
        
        return self.state, reward, done

# Q-Learning
env = GridWorld(size=5)
Q = np.zeros((5, 5, 4))  # Q(state, action)

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount
epsilon = 0.1  # Exploration

print("Training Q-Learning agent...")

for episode in range(500):
    state = env.reset()
    total_reward = 0
    
    for step in range(100):
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = Q[state].argmax()
        
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Q-learning update
        best_next_action = Q[next_state].max()
        Q[state + (action,)] += alpha * (reward + gamma * best_next_action - Q[state + (action,)])
        
        state = next_state
        
        if done:
            break
    
    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {total_reward}")

print("Learned optimal policy!")
```

## 5. Related Concepts

RL connects to control theory, operations research, and economics through optimal decision-making under uncertainty. Dynamic programming provides algorithms for computing optimal policies when environment dynamics are known. RL extends this to unknown dynamics, learning through interaction.

RL relates to supervised learning through imitation learning and inverse RL. Rather than learning from rewards, agents can learn from demonstrations (supervised), or infer reward functions from expert behavior (inverse RL).

## 6. Fundamental Papers

**["Reinforcement Learning: An Introduction" (2018)](http://incompleteideas.net/book/the-book-2nd.html)**  
*Authors*: Richard Sutton, Andrew Barto  
The definitive RL textbook, establishing mathematical foundations and core algorithms. Essential reading for anyone studying RL.

**["Playing Atari with Deep Reinforcement Learning" (2013)](https://arxiv.org/abs/1312.5602)**  
*Authors*: Volodymyr Mnih et al.  
DQN showed deep learning + RL could learn to play Atari games from pixels, launching deep RL revolution. Combined Q-learning with deep neural networks, experience replay, and target networks.

## Common Pitfalls and Tricks

Exploration-exploitation tradeoff is crucial. Pure exploitation (always best known action) never discovers better alternatives. Pure exploration (random actions) doesn't use learned knowledge. ε-greedy, softmax policies, or UCB-based methods balance both.

## Key Takeaways

Reinforcement learning trains agents through interaction with environments, learning policies that maximize cumulative rewards through trial and error. MDPs formalize sequential decision-making with states, actions, transitions, and rewards. Value functions estimate expected future returns, providing targets for learning. Q-learning learns optimal action-values through temporal difference updates, enabling learning without environment model. The exploration-exploitation tradeoff requires balancing discovering new strategies versus using known good ones. RL's delayed reward and credit assignment challenges make it harder than supervised learning but enable applications where supervision is unavailable or expensive.

