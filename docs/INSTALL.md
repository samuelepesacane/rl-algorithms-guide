# Installation per block

Not every block needs the same dependencies. A bandit script only needs NumPy,
while a deep RL block needs PyTorch, and a library baseline needs Stable-Baselines3
on top of that. So instead of making everyone install everything, the heavier
dependencies are split into optional "extras" (defined in `pyproject.toml`), and
you install only the ones a block actually uses.

This file is the lookup table for that: find your block, install what it needs, run the script.

## The base install

Always start here, from the repo root (a virtual environment is recommended):

```bash
pip install -e .
```

This pulls in the core stack (NumPy, Matplotlib, Gymnasium), which is already
enough for the bandit and tabular blocks. Everything below is an optional extra
you add on top.

## What each block needs

Most blocks can be run two ways, so the table has two columns, and both matter.
The "from scratch" column is the implementation you build and learn from, written
out so you can see every piece of the algorithm. The "library baseline" column runs
the same algorithm through a real library (Stable-Baselines3 or d3rlpy), and it is
just as much part of the point: in practice you will not always reimplement
something when a mature library already does it better and faster, so these scripts
show you how to reach for one and how to read its results. You only need a block's
baseline extra if you want to run that side.

| Block | From scratch | Library baseline |
|---|---|---|
| 01 Bandits | base install | (none) |
| 02 Tabular | base install | (none) |
| 03 DQN | `.[deep]`, or `.[deep,box2d]` for LunarLander | `.[sb3]`, or `.[sb3,box2d]` for LunarLander |
| 04 Policy gradients / Actor-Critic | `.[deep]` | `.[sb3]` |
| 05 TRPO / PPO | `.[deep,box2d]` | `.[sb3,sb3-contrib,box2d]` |
| 06 DDPG / TD3 / SAC | `.[deep]` | `.[sb3]` |
| 07 Model-based | base install for Dyna-Q, `.[deep]` for MPC and MBPO | `.[sb3]` |
| 08 Offline RL | `.[deep]` | `.[offline]` |

Here `.[x]` is shorthand for the full command `pip install -e ".[x]"`.

A few notes on the rows that are not obvious:

- LunarLander (used in blocks 03 and 05) is a Box2D environment, so it needs the
  `box2d` extra on top of whatever else the block uses. CartPole and Pendulum are
  classic-control environments and need nothing extra.
- Block 05's TRPO baseline lives in sb3-contrib, not in core SB3, which is why that
  row adds `sb3-contrib`. Installing it pulls SB3 in with it, so you do not also
  need to list `sb3` separately.
- Block 08 has no offline RL in SB3, so its baseline uses d3rlpy, which is the
  `offline` extra.
- Block 07 is split: Dyna-Q runs on the tabular Gridworld and needs nothing beyond
  the base install, while MPC and MBPO use a neural dynamics model on Pendulum and
  need `deep`.

## Running the tests

The test suite needs pytest (the `dev` extra) plus whatever the code under test
imports, so for the neural blocks add `deep` as well:

```bash
pip install -e ".[dev,deep]"
pytest -q
```

## Just give me everything

If you would rather not think about it, one command installs every optional extra:

```bash
pip install -e ".[all]"
```

That is the simplest path. The per-block split above only exists so you can skip
installing PyTorch or Box2D when you are still on the early blocks.
