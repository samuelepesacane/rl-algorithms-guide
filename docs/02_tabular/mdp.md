# Markov Decision Processes (MDPs)

This file is a compact reference for the basic objects in Reinforcement Learning:
Markov processes, Markov reward processes, and Markov decision processes.

---

## 1) The Markov property (the key assumption)

RL environments are often modelled using states.

A state is called Markov if:

> the future is independent of the past given the present

Formally, a state $S_t$ is Markov if and only if:

$$
P(S_{t+1} \mid S_t) = P(S_{t+1} \mid S_1, \dots, S_t)
$$

the probability of transition to $S_{t+1}$, given $S_t$, is equal to the probability of transition to $S_{t+1}$, given all the past states, from $S_1$ to $S_t$.

This means:
- the state must contain all information that matters for predicting the future
- once you know the state, the history can be ignored

This is why in RL you spend time thinking about
"what should I put in the state?"

---

## 2) Markov Process (Markov Chain)

A Markov Process has states and transitions.

It’s defined by a tuple:

$$
\langle \mathcal{S}, P \rangle
$$

- $\mathcal{S}$: finite set of states
- $P$: transition matrix with entries

$$
P_{ss^{\prime}} = P(S_{t+1} = s^{\prime} \mid S_t = s)
$$

Each row sums to 1.

But there is no "goal" yet, because there are no rewards.

---

## 3) Markov Reward Process (MRP)

A Markov Reward Process adds rewards and discounting.

It’s defined by:

$$
\langle \mathcal{S}, P, R, \gamma \rangle
$$

- $R(s)$ is the expected immediate reward:

$$
R(s) = \mathbb{E}[R_{t+1} \mid S_t = s]
$$

- $\gamma \in [0, 1]$ is the discount factor

So now we have rewards associated with states, and a discount factor that it's useful when we want to look at future rewards (not only immediate).

### Return

The return from time $t$ is the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

Why discount?

- it prevents infinite returns in cyclic tasks (when rewards are bounded)
- it makes the math cleaner
- it matches "preference for sooner rewards" in many real settings
- if $\gamma = 0$ I look only at the immediate rewards
- if $\gamma = 1$ all rewards have the same importance (no discount). In this case, returns are still fine in episodic tasks (finite episodes), but in continuing tasks you typically need $\gamma < 1$

In RL, what we want to maximise is the expected return, that is the expected cumulative rewards.

---

## 4) Episodic vs continuing tasks (finite vs infinite horizon)

In RL, tasks are often classified by whether they terminate.

### Episodic tasks (finite horizon)

An **episodic** task has episodes that end in a **terminal state**.\
A terminal state is an absorbing end-of-episode state: once you enter it, the episode stops and there are 
no further rewards/transitions (equivalently, the value function of the terminal state is 0, $V(\text{terminal}) = 0$, for the remaining return).\
There exists a terminal time $T$ such that the episode ends at $S_T$.

- Horizon: **finite** (the return sums over a finite number of rewards)
- Return:
  $$
  G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}
  $$
- Important consequence:
  even with $\gamma = 1$, the return is still finite (because the episode ends),
  assuming rewards are bounded.

Examples: Gridworld with goal, CliffWalking, most games with "game over" (or "You died" in Dark Souls).

### Continuing tasks (infinite horizon)

A **continuing** task does **not** have terminal states (it can run forever).
There is no terminal time $T$.

- Horizon: **infinite**
- Return (discounted formulation):
  $$
  G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
  $$
- Important consequence:
  to keep $G_t$ finite in general, we typically need $\gamma < 1$
  (with bounded rewards).

Examples: robot control, process control, keep balancing forever, many real-world systems.

### Finite vs infinite horizon

- **Finite horizon** usually means episodic (there is a terminal time $T$).
- **Infinite horizon** usually means continuing (no terminal state), so we use:
  - discounted return with $\gamma < 1$, or
  - an alternative objective like average reward (advanced topic).

### Practical note (important in code)

Even for continuing problems, implementations often use timeouts/truncation
(e.g. stop after 1000 steps). This creates artificial episode boundaries for training,
but it is not the same as a true terminal state in the MDP.

---

## 5) Value function and the Bellman equation (MRP)

The state-value function tells you the long-term value of a state:

$$
v(s) = \mathbb{E}[G_t \mid S_t = s]
$$

that is the expected return from state $s$.

The Bellman equation is just the "one-step decomposition" of that function:

$$
v(s) = R(s) + \gamma \sum_{s^{\prime}} P_{ss^{\prime}} v(s^{\prime})
$$

so the value function is given by the immediate rewards plus the discounted value function of the next state (note that the sum is taken over all next state).
Try to put the formula of the return inside the value function and see if you can obtain the Bellman equation. I really recommend doing this, it will make everything more clear.

In matrix form:

$$
\mathbf{v} = \mathbf{R} + \gamma P \mathbf{v}
$$

This is a linear system. It can be solved directly:

$$
\mathbf{v} = (I - \gamma P)^{-1}\mathbf{R}
$$

However, this direct solution is expensive for large problems.

In practice, we use iterative methods to solve Bellman equation, such as Dynamic programming.

---

## 6) Markov Decision Process (MDP)

An MDP is an MRP where an agent chooses actions.

It’s defined by:

$$
\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
$$

- $\mathcal{S}$: states
- $\mathcal{A}$: actions
- transition dynamics depend on the action:

$$
P^a_{ss^{\prime}} = P(S_{t+1} = s^{\prime} \mid S_t = s, A_t = a)
$$

- rewards can also depend on action:

$$
R^a_s = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]
$$

---

## 7) Policies

A policy tells you how actions are chosen:

$$
\pi(a \mid s) = P(A_t = a \mid S_t = s)
$$

Important detail: in standard MDP theory, policies are stationary (they depend on the current state, not time or history).

---

## 8) Value functions in an MDP

Given a policy $\pi$:

### State-value

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]
$$

this function tells us how good (the value) is state $s$, that is the expected cumulative return, starting from state $s$ and follow the policy.

### Action-value

$$
q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]
$$

You can read $q_\pi(s,a)$ as:
"If I’m in $s$ and take $a$ now, how good is that, assuming I follow $\pi$ afterwards?"

---

## 9) Bellman expectation equations (MDP)

The "one-step decomposition" shows up again.

For the state-value function:

$$
v_\pi(s) = \sum_a \pi(a \mid s)\left[
R^a_s + \gamma \sum_{s^{\prime}} P^a_{ss^{\prime}} v_\pi(s^{\prime})
\right]
$$

The state-value function can again be decomposed into immediate reward plus discounted value of successor state.

For the action-value function:

$$
q_\pi(s,a) = R^a_s + \gamma \sum_{s^{\prime}} P^a_{ss^{\prime}} \sum_{a^{\prime}} \pi(a^{\prime} \mid s^{\prime}) q_\pi(s^{\prime},a^{\prime})
$$

These equations are the backbone of:
- Dynamic Programming
- Temporal-Difference learning
- modern deep RL (just with function approximation)

---

## 10) Optimal value functions and optimality equations

### What optimal means

Instead of evaluating one policy, we ask: what is the best achievable performance?

The optimal value function specifies the best possible performance in the MDP.

So, we define the optimal values as:

$$
v_{\ast}(s) = \max_\pi v_\pi(s)
\qquad
q_{\ast}(s,a) = \max_\pi q_\pi(s,a)
$$

note that the max is taken over all policies.

* $v_{\ast}(s)$: best possible expected return starting from state $s$
* $q_{\ast}(s,a)$: best possible expected return starting from state $s$, taking action $a$, and behaving optimally after

An MDP is considered "solved" when we know $v_{\ast}$ or $q_{\ast}$, because we can derive an optimal policy from them.

The Bellman optimality equations:

$$
v_{\ast}(s) = \max_a \left[
R^a_s + \gamma \sum_{s^{\prime}} P^a_{ss^{\prime}} v_{\ast}(s^{\prime})
\right]
$$

and

$$
q_{\ast}(s,a) = R^a_s + \gamma \sum_{s^{\prime}} P^a_{ss^{\prime}} \max_{a^{\prime}} q_{\ast}(s^{\prime},a^{\prime})
$$

Once you know $q_{\ast}$, a greedy optimal policy is immediate:

$$
\pi_{\ast}(s) = \arg\max_a q_{\ast}(s,a)
$$

There always exists a deterministic optimal policy for a finite MDP.

A useful identity to remember:

$$
v_{\ast}(s)=\max_a q_{\ast}(s,a)
$$

---

## 11) How do we actually compute optimal values?

If you know the model (you know $P$ and $R$), you can plan with Dynamic Programming:

* Policy Iteration:

  1. policy evaluation (compute $v_\pi$)
  2. policy improvement (make $\pi$ greedy w.r.t. current value)
* Value Iteration: repeatedly apply the Bellman optimality backup until values converge

These methods are exactly built around the Bellman expectation/optimality equations above. 

If you don’t know $P$ and $R$, you’ll estimate values from experience. This is where Monte Carlo and Temporal-Difference methods enter.

These methods are explained in `docs/02_tabular/theory.md`

---

## 12) A quick note on extensions (optional reading)

The basic MDP assumes fully observable state.

If the environment is partially observable, you get a POMDP: you don't see the true state directly, only observations.
A standard trick is to work with a belief state (a probability distribution over states), which becomes Markov again.

For now, you can treat this as later topics.
