# Tabular RL (MDPs, Dynamic Programming, SARSA, Q-learning)

This block is where "real RL" starts. Bandits had no state, you pick an arm and get a reward.

Here we introduce **Markov Decision Processes (MDPs)**: states, actions, transitions, rewards, and long-term planning.

We will look at two big families of methods:

- **Dynamic Programming (DP)**: planning when you know the model. So you can query $P(s^{\prime}|s,a)$ (transition probability) and $R(s,a)$ (rewards).
- **Temporal-Difference (TD) Control**: learning when you don’t know the model, only samples from interaction.

This is based on the same core idea: the **Bellman equations**.
DP uses full-width backups (expectations over all next states). TD uses sample backups (single transitions). 

---

## What to read (recommended order)

### 1) MDP foundations (definitions)
Start here:

- `docs/02_tabular/mdp.md`

It introduces Markov property, MDP tuple, policies, value functions, and Bellman equations.

### 2) DP + TD control
Then read:

- `docs/02_tabular/theory.md`

This explains:
- value iteration / policy iteration (DP)
- SARSA / Q-learning / Expected SARSA (TD control)

and how they relate under a unified view. 

---

## What to run (hands-on order)

### A) DP on a tiny Gridworld (planning)
Run:

- `examples/02_tabular/train_gridworld_dp.py`

What it does:
- creates a small tabular MDP where we can explicitly compute backups
- runs value iteration and/or policy iteration
- prints (and/or visualises) the learned value function and greedy policy

This is a clean way to see Bellman backups working.

### B) TD control on CliffWalking (learning by interaction)
Run:

- `examples/02_tabular/train_cliffwalking.py`

What it does:
- uses Gymnasium’s CliffWalking
- trains and compares SARSA, Q-learning, and Expected SARSA
- plots episode returns so you can see the behavioural difference:
  SARSA often learns a safer route because it learns the value of the exploratory policy,
  while Q-learning pushes toward the greedy optimal path.

Tip: if curves look noisy, run multiple seeds and average.

---

## Where the code lives

This repo is split so you can study in layers:

### `src/rl_algorithms_guide/tabular/` - the implementations (importable code)
This folder contains the tabular algorithms you can import and reuse:

- `gridworld.py`: a tiny MDP with explicit transitions/rewards (for DP)
- `value_iteration.py`: Bellman optimality backups on $V(s)$
- `policy_iteration.py`: evaluation + greedy improvement loop
- `sarsa.py`: on-policy TD control
- `q_learning.py`: off-policy TD control
- `expected_sarsa.py`: lower-variance on-policy TD target

Importable means you can do things like:

```python
from rl_algorithms_guide.tabular.q_learning import QLearningAgent
```

like we did with bandits.

The scripts in `examples/` just use these modules.

### `examples/02_tabular/` - runnable scripts (experiments / plots)

These are short scripts you run from the command line.
They glue together:

* an environment (Gridworld or Gymnasium)
* an agent from `src/`
* a training loop + logging/plotting

Think of them as mini-labs.

### `tests/` - small checks that protect correctness

Tabular RL is easy to get subtly wrong (terminal bootstrapping, indexing, etc.).
The tests are intentionally small, but they catch the most common mistakes early.

### What the DP tests check

The DP tests verify two core correctness properties:

- **Greedy policy extraction**: the policy returned by value iteration/policy iteration is greedy with respect to 
  the returned value function `V`.
  For each state, the chosen action is one of the actions with maximal one-step lookahead value.

- **Terminal handling**: in our Gridworld setup, terminal states are absorbing with reward 0, so `V(terminal) = 0`.

These tests are intentionally small, but they catch common bugs like wrong transition indexing, 
bootstrapping past terminal states, or extracting a non-greedy policy.

---

## Common gotchas (worth knowing early)

* **Terminal bootstrapping**: when `done=True`, your target should not include $\gamma Q(s^{\prime},a^{\prime})$.
* **Exploration matters**: if $\varepsilon$ is too small, you may never learn; if it’s too big, learning looks noisy forever.
* **SARSA vs Q-learning is not "better vs worse"**: they optimise different things during learning. CliffWalking is the classic example where that difference shows up clearly. 

---

## Next

After tabular RL, deep RL is basically "same ideas, but values/policies are approximated with neural nets".

So the next step is:

* `docs/03_dqn/`
