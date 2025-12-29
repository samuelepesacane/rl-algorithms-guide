# Getting started

This repo is meant to be read like a path, not like a reference manual.

You'll usually do three things for each topic:

1. read a short note in `docs/`
2. look at the minimal code in `src/`
3. run a small script in `examples/` and inspect the plots

If you do those three steps, RL starts to feel much less "mystical" and more fun.

Tip: most scripts have CLI flags. Run `--help` to see options:

```bash
python examples/01_bandits/train_compare.py --help
```
---

## 1) Setup

### Create a virtual environment

Pick one:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install the package

From the repo root:

```bash
pip install -e .
```

If you want *all* optional dependencies (PyTorch, SB3, etc.):

```bash
pip install -e ".[all]"
```

---

## 2) How to run examples

Examples are simple Python scripts. You run them from the repo root, like:

```bash
python examples/01_bandits/train_compare.py
```

Most examples save plots into:

* `assets/plots/`

If nothing shows up, check:

* you're running from the repo root
* your environment is activated
* dependencies installed correctly

---

## 3) How to navigate the repo

* `docs/`
  The learning path. Numbered on purpose, so you can follow it in order. You can find theory and how to run examples.

* `src/rl_algorithms_guide/`
  The implementations. This is a normal Python package, so you can import modules from it.

* `examples/`
  Small runnable scripts that use the package and save plots into `assets/plots/`.

* `tests/`
  A few scripts for a quick sanity checks (update rules, basic behavior), not heavy benchmarks.

---

## 4) Suggested learning order

You can jump around, but if you want a smooth ramp:

1. **Bandits** (`docs/01_bandits/`)

   * exploration vs exploitation
   * why greedy strategies fail
   * UCB as a "smart exploration" baseline

2. **Tabular MDPs** (`docs/02_tabular/`)

   * value iteration / policy iteration
   * SARSA vs Q-learning on a small environment

3. **Deep RL**

   * start with DQN, then policy gradients
   * later: PPO / SAC / TD3

4. **Model-based + Offline RL**

   * useful, but easier once the basics are solid

---

## 5) A note on expectations

RL is noisy.

Even correct implementations can look "bad" sometimes:

* learning curves vary with random seeds
* some algorithms are sensitive to hyperparameters
* small changes in environment settings can matter

When something doesn't work, that's not a failure, it's part of the subject.
Try changing one thing at a time and keep notes.

---

## Next

Go to:

* `docs/01_bandits/README.md`

and run your first experiment:

```bash
python examples/01_bandits/train_compare.py
```
