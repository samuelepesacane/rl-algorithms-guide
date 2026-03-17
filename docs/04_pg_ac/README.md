# 04 — Policy Gradients (REINFORCE + Actor-Critic)

This block is your first step into **policy optimization**.

So far (bandits + tabular + DQN), we mostly learned a value function and then acted greedy.
Here we learn the **policy directly**.

The key shift is:

- Value-based RL: learn $Q(s,a)$, then pick $a=\arg\max_a Q(s,a)$
- Policy gradient RL: learn $\pi_\theta(a \mid s)$, and update $\theta$ to increase expected return

This is the foundation for PPO (next block), SAC, and most modern continuous-control RL.

---

## What you will learn

1) **Stochastic policies**
- what $\pi_\theta(a \mid s)$ means
- why stochastic policies are useful (exploration, continuous actions, robustness)

2) **Trajectories and objectives**
- how a policy induces a distribution over trajectories $p_\theta(\tau)$
- why the objective is an expectation: $J(\theta)=\mathbb{E}[G_0]$

3) **The policy gradient**
- the log-likelihood trick
- why gradients only go through the policy, not the environment dynamics

4) **REINFORCE (Monte Carlo policy gradient)**
- compute returns from full episodes
- update the policy using $\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t$

5) **Baselines and advantages**
- what a baseline is and why it reduces variance
- what an advantage function is: $A(s,a)=Q(s,a)-V(s)$
- how to reduce variance without changing the expected gradient

6) **Actor-Critic**
- the actor (policy) + critic (value) pattern
- how TD error can be used as an advantage estimate

7) **A2C (Advantage Actor-Critic)**
- why parallel environments reduce gradient variance
- n-step returns and bootstrapping across multiple workers
- the entropy bonus: what it is, why it helps, and how to tune it
- A3C (concept only): why A2C is implemented instead

---

## Files in this block

### Docs

- `docs/04_pg_ac/theory.md`  
  Derivations and intuition:
  stochastic policies, trajectories, REINFORCE, baselines, advantage, actor-critic.

### Implementation (importable package code)

- `src/rl_algorithms_guide/pg_ac/reinforce.py`  
  REINFORCE (Monte Carlo policy gradient) with optional running-mean baseline.

- `src/rl_algorithms_guide/pg_ac/actor_critic.py`  
  Minimal actor-critic (categorical policy + value critic) using TD error as advantage.

- `src/rl_algorithms_guide/pg_ac/a2c.py`  
  A2C with N parallel environments (SyncVectorEnv), n-step returns, and entropy bonus.

- `src/rl_algorithms_guide/common/utils.py`  
  Small utilities: discounted returns, normalization, running-mean baseline.

- `src/rl_algorithms_guide/common/plotting.py`  
  Shared plotting utilities including `pad_curves_with_last_value` and `save_lines_with_bands`.

### Runnable examples (what you actually run)

- `examples/04_pg_ac/train_reinforce_cartpole.py`  
  REINFORCE on CartPole.

- `examples/04_pg_ac/train_actor_critic_cartpole.py`  
  Actor-Critic on CartPole (usually learns faster / more smoothly).

- `examples/04_pg_ac/train_a2c_cartpole.py`  
  A2C on CartPole with configurable number of parallel environments and rollout length.

All scripts save plots to:

- `assets/plots/`

---

## How to run

From the repo root, after installing:

```bash
pip install -e .
```


### REINFORCE (episode-based updates)

```bash
python examples/04_pg_ac/train_reinforce_cartpole.py
```

Useful knobs:

```bash
python examples/04_pg_ac/train_reinforce_cartpole.py --episodes 800 --lr 1e-3
python examples/04_pg_ac/train_reinforce_cartpole.py --no-baseline
python examples/04_pg_ac/train_reinforce_cartpole.py --no-adv-norm
```

### Actor-Critic (step-based TD updates)

```bash
python examples/04_pg_ac/train_actor_critic_cartpole.py
```

Useful knobs:

```bash
python examples/04_pg_ac/train_actor_critic_cartpole.py --total-steps 150000
python examples/04_pg_ac/train_actor_critic_cartpole.py --lr-actor 3e-4 --lr-critic 1e-3
python examples/04_pg_ac/train_actor_critic_cartpole.py --entropy-coef 0.01
```

### A2C (parallel environments, n-step returns)

```bash
python examples/04_pg_ac/train_a2c_cartpole.py
```

Useful knobs:

```bash
python examples/04_pg_ac/train_a2c_cartpole.py --num-envs 8 --rollout-steps 5
python examples/04_pg_ac/train_a2c_cartpole.py --num-envs 1 --rollout-steps 1   # = Actor-Critic
python examples/04_pg_ac/train_a2c_cartpole.py --entropy-coef 0.0               # no entropy bonus
python examples/04_pg_ac/train_a2c_cartpole.py --no-adv-norm
```

You should see:

* episodic return improving over time (learning curve)
* typically: Actor-Critic is less noisy than REINFORCE

---

## What plots to expect

You'll usually save (mean $\pm$ std over seeds):

* **return vs episode** (learning curve)
* **policy loss** (sanity signal; can be noisy)
* for actor-critic and A2C: **value loss** (should stay finite and decrease)
* for A2C: **policy entropy** (starts high near a random policy, should gradually decrease as learning progresses but not collapse to zero if `entropy-coef > 0`)

CartPole is intentionally chosen because it trains quickly and makes it easy to debug.

---

## Common pitfalls (worth reading before debugging)

* **REINFORCE is noisy**: it uses Monte Carlo returns, so variance is high.
  If learning looks unstable, that's not necessarily a bug.

* **Baseline confusion**:
  a baseline should *not* change the expected gradient, only reduce variance.
  If you use the value function as baseline, it should only depend on $s$, not $a$.
  In this repo the REINFORCE example uses a super simple baseline (running mean of episode return).

* **On-policy means on-policy**:
  these algorithms assume your data was generated by the current policy.
  If you start mixing old experience (replay buffer style), you are doing something else.

* **Learning rate matters**:
  policy gradients can blow up if your steps are too large.
  If returns go to nonsense fast, try a smaller `--lr` / `--lr-actor`.

* **Normalization helps**:
  advantage normalization is a common trick to make updates less fragile.

* **A2C gradient clipping**: A2C uses a tighter default `--max-grad-norm 0.5` than the
  single-step Actor-Critic. With N environments and n rollout steps, the combined batch
  can produce large gradients. If training is unstable, tighten the clip or reduce `--lr`.

* **A2C `--num-envs 1 --rollout-steps 1` should behave like Actor-Critic**:
  this is a useful sanity check. If A2C diverges there but Actor-Critic does not, the
  bug is in the rollout buffer or bootstrapping logic, not the parallelism.

* **If reward is sparse, learning can stall**:
  CartPole is dense enough that you should still see progress.

---

## Where this goes next (PPO)

PPO keeps the same ingredients as A2C:

* log probabilities $\log \pi_\theta(a \mid s)$
* advantage estimates $A(s,a)$
* actor + critic
* parallel environments

...but adds one more mechanism: preventing the policy from changing too much in a single update.
In A2C nothing stops a large gradient step from moving the policy far from where the data was collected.
PPO fixes that, which is why it tends to be more stable and sample-efficient.

So if A2C (or policy gradients in general) feels fragile on harder environments, 
that's exactly the problem PPO is designed to solve.
