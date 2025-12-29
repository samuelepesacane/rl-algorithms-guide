# Bandits ($\varepsilon$-greedy and UCB)

Bandits are the smallest possible RL problem.

There's no state, no transitions, no "episode".
You just pick an action and get a reward.

That's why bandits are a great place to start.
You can focus on the one thing that shows up everywhere in RL:

**exploration vs exploitation.**

---

## What you'll learn here

- Why "always pick the best-looking action" can fail
- How $\varepsilon$-greedy explores in a simple way
- How UCB explores in a more targeted way
- How to read two standard bandit plots:
  - average reward over time
  - % optimal action over time

---

## Where things are

Implementation (NumPy):

- `src/rl_algorithms_guide/bandits/bandit_envs.py`
- `src/rl_algorithms_guide/bandits/epsilon_greedy.py`
- `src/rl_algorithms_guide/bandits/ucb.py`

Runnable script:

- `examples/01_bandits/train_compare.py`

Theory note:

- `docs/01_bandits/theory.md`

---

## Run it

From the repo root:

```bash
python examples/01_bandits/train_compare.py
```

This should save plots into:

* `assets/plots/`

If you want to play with it, open the script and change:

* number of arms (`k`)
* number of steps
* $\varepsilon$ values
* UCB exploration constant (`c`)
* number of runs (higher = smoother curves)

### Optional: decaying $\varepsilon$-greedy

If you want an $\varepsilon$ that starts high and decreases over time, you can include the decaying variant:

```bash
python examples/01_bandits/train_compare.py --include-eps-decay \
  --eps-start 1.0 --eps-end 0.1 --eps-decay-steps 500 --eps-schedule linear
```

---

## What to look for

You'll usually see something like:

* pure greedy ($\varepsilon$ = 0) can get stuck early
* $\varepsilon$-greedy improves with some exploration
* UCB often reaches strong performance faster, because it explores "uncertain" arms on purpose

Don't worry if your curves look a bit different: randomness matters.

---

## Quick sanity checks (tests)

If you want to make sure the bandit code is behaving as expected, there's a small test file:

- `tests/test_bandits.py`

Run all tests from the repo root:

```bash
pytest -q
```

---

## Next

After bandits, go to:

* `docs/02_tabular/`

That's where the real Markov Decision Process (MDP) part starts (states, transitions, value functions).
