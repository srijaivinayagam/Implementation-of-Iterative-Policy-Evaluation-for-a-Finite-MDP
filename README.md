# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python


import gymnasium as gym
import numpy as np
     

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
     

# Access the unwrapped environment to use the transition model
env = env.unwrapped
     

# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n
     

# Parameters
gamma = 0.99
theta = 1e-8
     

# Random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions
     

from typing import Tuple
import numpy as np

# Initialize value function
V = np.zeros(n_states)

# -------------------------------------------------
# Policy Evaluation Function
# -------------------------------------------------

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
  prev_V = np.zeros(len(policy))
  iterations = 0
  while True:
    V = np.zeros(len(policy))
    for s in range(len(policy)):
      for a in range(env.action_space.n):
        action_prob = policy[s][a]
        for prob, next_state, reward, done in env.P[s][a]:
          V[s] += action_prob * prob * (reward + gamma * prev_V[next_state] * (not done))
    iterations += 1
    if np.max(np.abs(prev_V - V)) < theta:
      break
    prev_V = V.copy()
  return V, iterations

# Run policy evaluation
V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name: Srijai.v")
print("Register Number: 212224240015212225040426 ")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("Name:Srijai.v ")
print("Register Number: 212225040426 ")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()

```

---

## Output

```text
---------------------------------------------------------
gamma = 0.99

Name: Srijai.v
Register Number: 212225040426
Number of iterations: 71

State-Value Function:
[0.0123561  0.01042443 0.01933841 0.00947773 0.01478703 0.
 0.03889444 0.         0.03260246 0.08433763 0.13781085 0.
 0.         0.17034482 0.43357944 0.        ]
Name: Srijai.v
Register Number: 212225040426

State-Value Function as 4x4 Grid:
[[0.0124 0.0104 0.0193 0.0095]
 [0.0148 0.     0.0389 0.    ]
 [0.0326 0.0843 0.1378 0.    ]
 [0.     0.1703 0.4336 0.    ]]

------------------------------------------------------------
gamma = 0.69

Name: Srijai.v
Register Number: 212225040426
Number of iterations: 26

State-Value Function:
[0.00049587 0.00065634 0.00265268 0.0006986  0.00122657 0.
 0.01137027 0.         0.00538812 0.0246208  0.06326194 0.
 0.         0.07407924 0.33074484 0.        ]
Name: Srijai.v
Register Number: 212225040426

State-Value Function as 4x4 Grid:
[[0.0005 0.0007 0.0027 0.0007]
 [0.0012 0.     0.0114 0.    ]
 [0.0054 0.0246 0.0633 0.    ]
 [0.     0.0741 0.3307 0.    ]]

```
---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

```text
For γ = 0.99, the state-value function is higher, indicating that future rewards have a greater influence on state values, and convergence takes 71 iterations.



For γ = 0.69, the state values are generally lower due to greater discounting of future rewards, and convergence is faster, taking 26 iterations.


```




---


