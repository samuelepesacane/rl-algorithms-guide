# Reinforcement Learning Algorithms Guide (beginner-friendly)

This repo is a hands-on guide to Reinforcement Learning (RL).

The idea is simple:

- every topic has a short explanation (`docs/`)
- a minimal implementation (`src/`)
- and a small script you can run in a few minutes (`examples/`) that produces plots

I'm writing this for people who want to get comfortable with RL. The explanations focus on intuition and practical details. I'll keep the math light unless it's really necessary.

## What's included

This repo aims to cover (with runnable code + notes):

- **Bandits**: $\varepsilon$-greedy, Upper Confidence Bound (UCB)
- **Dynamic Programming**: value iteration, policy iteration
- **Tabular control**: SARSA (State-Action-Reward-State-Action), Q-learning (and Expected SARSA)
- **Deep RL (value-based)**: Deep Q-Network (DQN)
- **Policy gradients**: REINFORCE (Monte Carlo policy gradient), Actor-Critic
- **On-policy deep RL**: Proximal Policy Optimization (PPO)
- **Off-policy deep RL**: Soft Actor-Critic (SAC), TD3 (Twin-Delayed Deep Deterministic Policy Gradient)
- **Model-based RL**: Dyna / MPC (Model Predictive Control) plus one modern exemplar, such as Dreamer or MBPO (Model-based policy optimization), and I'll implement it on a small, simplified setup (toy-scale but concept-faithful)
- **Offline RL**: Behavioral Cloning (BC), Conservative Q-Learning (CQL), Implicit Q-Learning (IQL)

Some modules will be work in progress while I build them out, but the goal is that everything above ends up explained and runnable.

## Quickstart

After installing the package (see below), you can run your first experiment:

```bash
python examples/01_bandits/train_compare.py
```

This will compare $\varepsilon$-greedy and UCB on a simple bandit setup and save plots into:

* `assets/plots/`

If you're not sure where to start, open:

* `docs/00_getting_started.md`
* then continue with `docs/01_bandits/`

## How to navigate this repo

This repo is split into three main parts. They work together:

### `docs/`: the explanations (the "book")

This is where you read and learn.

- It's a learning path, so it's numbered on purpose.
- Each section usually contains:
  - a short **README**: what you're about to learn and what to run
  - a **theory.md**: a simple explanation of the algorithm (intuition first, then a bit of math if needed)

If you feel lost, start here. The docs tell you what to run and what to look at.

---

### `src/rl_algorithms_guide/`: the actual implementations (the "library")

This is where the algorithms are implemented.

For example, inside you'll find code like:
- `bandits/epsilon_greedy.py`
- `tabular/q_learning.py`
- later: `dqn/`, `ppo/`, etc.

You usually don't run files from `src/` directly.
Instead, the scripts in `examples/` import this code and use it.
This folder is a normal Python package, so you can import modules from it. Indeed, Python treats `rl_algorithms_guide` like any other installed package.

So you can do imports like:

```python
from rl_algorithms_guide.bandits.epsilon_greedy import EpsilonGreedyAgent
```

This is useful because:

* your code stays organized (implementations in one place, experiments in another)
* example scripts don't need messy relative imports
* later you can reuse these implementations in other projects

---

### `examples/`: runnable experiments (the "demos")

This is where you run things.

Each script in `examples/` is meant to be:

* small
* runnable in a few minutes
* focused on one idea

They usually:

1. create an environment (or a toy problem)
2. create an agent from `src/rl_algorithms_guide/`
3. train it for a short time
4. save plots into `assets/plots/`

So the usual workflow is:

* read the relevant page in `docs/`
* run the script in `examples/`
* if you're curious, open the implementation in `src/` and see how it works

That's the loop this repo is built around.

---

### `tests/`: quick sanity checks 
This repo includes a small test suite in `tests/`.

The tests are not meant to prove "this algorithm is optimal" or anything like that.
They are simply there to catch easy mistakes, like:
- a wrong update rule (sign errors, off-by-one issues)
- UCB not pulling each arm once at the start
- bandit rewards behaving differently than expected in deterministic settings

### How to run tests

From the repo root (with your virtual environment active):

```bash
pytest -q
````

If `pytest` is not installed, install the dev dependencies:

```bash
pip install -e ".[dev]"
```

### When you should run them

A good habit is:

* run tests after changing any core implementation under `src/`
* run tests before committing, if you changed logic (not just comments/docs)

If a test fails, don’t panic. It usually means you changed something small and a check needs to be updated, or you found a real bug early (which is exactly why the tests exist).

For now only a test on bandits is present. I plan to make another one for tabular methods later.

---

## Repo structure

I'll add the structure of this repo later, when there are more files.

## Install

Create a virtual environment (recommended), then from the repo root:

```bash
pip install -e .
```

### Optional extras

Some sections use extra dependencies (deep RL, SB3, offline RL). You can install everything with:

```bash
pip install -e ".[all]"
```

## Notes

* This repo uses **Gymnasium** for environments and **PyTorch** for deep RL.
* Tabular methods and bandits are implemented with **NumPy**.
* I'll upload new methods as soon as I can. I am a bit busy with work, so I'll proceed slowly (sorry).

## License

MIT (see `LICENSE`)
