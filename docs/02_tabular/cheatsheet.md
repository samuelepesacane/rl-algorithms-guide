# 02 - Tabular RL Cheatsheet

A one-page refresher for the tabular block. For the full derivations, the
on-policy vs off-policy discussion, and the worked numeric examples, read
`docs/02_tabular/theory.md` (and `docs/02_tabular/mdp.md` for the MDP
definitions).

Everything here is "store one number per state (or per state-action) in a table
and update it". Three families differ only in **how the update target is
built**: DP uses the full model, MC uses the actual return, TD uses one step
plus a bootstrap.

---

## The one update shape

Almost every method is:

$$
\text{estimate} \leftarrow \text{estimate} + \alpha\,\big(\text{target} - \text{estimate}\big)
$$

The "estimate" is $V(s)$ or $Q(s,a)$. The only thing that changes between
methods is the **target**.

| Family | Target uses | Needs the model? | Waits for episode end? |
|---|---|---|---|
| DP | full expectation over next states | yes | no (it is planning) |
| MC | actual return $G_t$ | no | yes |
| TD | $R_{t+1} + \gamma V(S_{t+1})$ (bootstrap) | no | no |

---

## Dynamic Programming (you know $P$ and $R$)

**Problem it solves:** when the model is known, you can plan the optimal policy
exactly, no sampling needed.

**Policy evaluation backup:**

$$
v_{k+1}(s) \leftarrow \sum_a \pi(a\mid s)\Big[R(s,a) + \gamma\sum_{s'} P(s'\mid s,a)\,v_k(s')\Big]
$$

**Value iteration** folds in the greedy step with a $\max$:

$$
v_{k+1}(s) \leftarrow \max_a \Big[R(s,a) + \gamma\sum_{s'} P(s'\mid s,a)\,v_k(s')\Big]
$$

**Policy iteration** alternates full evaluation and greedy improvement until the
policy stops changing.

**Defining idea:** Bellman backup + iteration. Backups are $\gamma$-contractions
when $\gamma<1$, so they converge to a unique fixed point.

**Watch out for:** DP is a planning tool, it needs $P$ (transition probabilities) and $R$ (rewards). That is why this
block uses a transparent `GridworldMDP` that exposes them.

---

## Monte Carlo (model-free, full returns)

**Problem it solves:** evaluate a policy without a model.

**Update:** $V(S_t) \leftarrow V(S_t) + \alpha\,(G_t - V(S_t))$, where $G_t$ is
the full return to the end of the episode.

**Defining idea:** no bootstrapping. Wait until the episode ends, then use what
actually happened.

**Watch out for:** unbiased but high variance; only works on episodic tasks
because it needs the episode to finish.

---

## TD(0) (model-free, one-step bootstrap)

**Update:**

$$
V(S_t) \leftarrow V(S_t) + \alpha\,\underbrace{\big(R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\big)}_{\delta_t,\ \text{the TD error}}
$$

**Defining idea:** "MC, but you do not wait until the end." Bootstrap from your
current guess of the next state.

**Bias/variance:** lower variance than MC, but biased while $V$ is still wrong.
**n-step TD** ($G_t^{(n)}$, bootstrap after $n$ rewards) is the dial between
TD(0) ($n=1$) and MC ($n\to$ end); **TD($\lambda$)** mixes all $n$ at once.

---

## TD control: SARSA vs Expected SARSA vs Q-learning

All three share $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha(\text{target} - Q(S_t,A_t))$
and differ only in how they treat the **next** action.

| Method | Target | On/off-policy | Learns |
|---|---|---|---|
| SARSA | $R_{t+1} + \gamma\,Q(S_{t+1}, A_{t+1})$ | on | $q_\mu$ (value of the behaviour policy) |
| Expected SARSA | $R_{t+1} + \gamma\sum_{a'}\mu(a'\mid S_{t+1})Q(S_{t+1},a')$ | on | $q_\mu$, less noisy |
| Q-learning | $R_{t+1} + \gamma\max_{a'} Q(S_{t+1},a')$ | off | $q_\ast$ (optimal) |

**The key distinction:** on-policy ($\pi=\mu$) bakes the cost of your own
exploration into the values; off-policy ($\pi\neq\mu$) learns the greedy/optimal
values even while you explore. It is about *which policy the target evaluates*,
not about whether you explore.

**The CliffWalking intuition:** SARSA learns the exploratory policy is risky
near the cliff, so it takes the safer longer route; Q-learning learns the greedy
values and hugs the cliff, eating exploration accidents during training.

**Key knobs (typical):** $\alpha \approx 0.1$ (learning rate), $\gamma \approx 0.99$ (discount factor),
$\varepsilon$ around $0.1$ (often decayed).

---

## Mental traps

- **Terminal handling:** when the episode ends, do not bootstrap, the target is
  just $R_{t+1}$.
- $\alpha$ too large makes values bounce; too small makes learning crawl.
- Random tie-breaking in $\arg\max$ helps early exploration.
- Average over seeds; single tabular runs still vary.

---

## Where to look in the repo

- DP: `src/rl_algorithms_guide/tabular/{gridworld,value_iteration,policy_iteration}.py`,
  `examples/02_tabular/train_gridworld_dp.py`
- TD control: `src/rl_algorithms_guide/tabular/{sarsa,expected_sarsa,q_learning}.py`,
  `examples/02_tabular/train_cliffwalking.py`
- Full theory: `docs/02_tabular/theory.md`, `docs/02_tabular/mdp.md`

**Next block:** `docs/03_dqn/` replaces the table with a neural network, so
Q-learning can scale to large state spaces.
