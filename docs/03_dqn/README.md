# 03 - DQN (DQN, Double DQN, Dueling DQN)

This block is the bridge from **tabular Q-learning** (Block 02) to **deep RL**.

In Block 02 we stored values in tables:
- `Q(s, a)` was a table indexed by a discrete state `s` and action `a`.

That breaks as soon as the state space is too large (or continuous), because you can't store one entry per state-action pair.

So here we approximate:

- `Q(s, a) ≈ Q_θ(s, a)` with a neural network.

Function approximation is basically when in Physics we "assume continuity and pray", but with gradients.

The core idea is still **Q-learning control**:
- bootstrap with a TD target
- improve with a greedy `max` backup

DQN adds two classic stabilizers so this works with neural networks:
- **experience replay**
- **a target network**

We're replacing a Q-table with a neural net because RAM is finite (I am scared of 'memory error' 
for how many times I saw it), but optimism is not.

---

## What you will learn

1) **Core DQN**
- Q-network outputs a vector `Q_θ(s, ·)` over actions
- replay buffer (shuffle experience)
- target network (frozen copy for stable TD targets)
- epsilon-greedy exploration
- TD error + Huber loss

2) **Double DQN**
- reduces Q-value overestimation by decoupling:
  - action selection (online network)
  - action evaluation (target network)

3) **Dueling DQN**
- architecture change:
  - learn `V(s)` and `A(s, a)` separately
  - combine them to produce `Q(s, a)`

4) **Double + Dueling**
- they stack cleanly:
  - dueling changes the network architecture
  - double changes how TD targets are computed

---

## Where the code is

**Core implementation**
- `src/rl_algorithms_guide/dqn/dqn.py`

**Example scripts**
- `examples/03_dqn/train_dqn_cartpole.py`
- `examples/03_dqn/train_dqn_lunarlander.py`

---

## How to run

From the repo root:

### CartPole-v1 (fast sanity check)

```bash
python examples/03_dqn/train_dqn_cartpole.py --algo dqn
python examples/03_dqn/train_dqn_cartpole.py --algo double
python examples/03_dqn/train_dqn_cartpole.py --algo dueling
python examples/03_dqn/train_dqn_cartpole.py --algo double_dueling
```

### LunarLander-v2 (harder, needs more steps)

```bash
python examples/03_dqn/train_dqn_lunarlander.py --algo dqn
python examples/03_dqn/train_dqn_lunarlander.py --algo double
python examples/03_dqn/train_dqn_lunarlander.py --algo dueling
python examples/03_dqn/train_dqn_lunarlander.py --algo double_dueling
```

---

## Outputs

Both scripts save plots under:

* `assets/plots/`

Typical outputs:

* `dqn_*_returns.png` (episode returns)
* `dqn_*_loss.png` (training loss / TD loss)
* `dqn_*_epsilon.png` (epsilon schedule)

If returns go up, you’re learning. If loss goes to NaN, you're learning... a valuable lesson.

---

## Reading order

If you're new to DQN, the fastest path is:

1. `docs/03_dqn/theory.md` (what the algorithm is doing and why)
2. `src/rl_algorithms_guide/dqn/dqn.py` (how it is implemented)
3. run CartPole (quick feedback loop)
4. run LunarLander (harder + noisier, but more realistic)
