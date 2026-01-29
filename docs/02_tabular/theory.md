# Tabular RL Theory (DP + Monte Carlo + TD Control)

This note goes with the code in `src/rl_algorithms_guide/tabular/`.

It focuses on three families of ideas you keep seeing in RL:

- **Dynamic Programming (DP)**: planning when you know the model.
- **Monte Carlo (MC)**: learning values from full episodes (no bootstrapping).
- **Temporal-Difference (TD)**: learning values from partial experience (bootstrapping).

If you want the clean definitions page (Markov property, MDP tuple, Bellman equations),
read this first:

- `docs/02_tabular/mdp.md`

---

## 0) The big picture

Most RL algorithms are variations of the same pattern:

1. **Estimate values** (how good is a state / action?)
2. **Improve the policy** (choose better actions more often)

The main difference between DP, MC, and TD is how the "update target" is built:

- **DP**: uses the full expectation over next states (needs the model).
- **MC**: uses the actual return you observe by running to the end of the episode.
- **TD**: uses a one-step sample + a bootstrap estimate of what comes after.

In RL, an update almost always looks like:

$$
\text{estimate} \leftarrow \text{estimate} + \alpha \, (\underbrace{\text{target}}_{\text{what we push toward}} - \text{estimate})
$$

Here "estimate" is usually either a state value $V(s)$, or an action value $Q(s,a)$.

So when I say "how the update target is built", I mean:

* What number do we use as the target to push $V$ or $Q$ toward?

* Where does that number come from (model vs experience, full return vs bootstrap)?

A useful word you'll see everywhere is backup:

> A backup updates an estimate (like $V(s)$ or $Q(s,a)$) by backing up information from the future.

DP = full backup, MC/TD = sample-based backups.
 
---

## 1) Dynamic Programming (planning with a known model)

### What dynamic programming means in RL

Dynamic Programming is a general method to solve complex problems by breaking them into smaller subproblems, solving those, and combining the results.

DP is especially useful when two properties hold:

- **Optimal substructure**: the optimal solution can be built from optimal solutions of smaller subproblems (principle of optimality)
- **Overlapping subproblems**: the same subproblems show up repeatedly, so you want to reuse results

MDPs satisfy both:

- the **Bellman equations** give the recursive decomposition (the subproblems)
- the **value function** is the "cache" that stores solutions for reuse

In RL terms:

- you have an MDP
- you know the model transition probabilities $P(s^{\prime} \mid s,a)$ and rewards $R(s,a,s^{\prime})$
- you repeatedly apply Bellman backups until values (and then the policy) stop changing

A short summary is:

**DP = Bellman backups + iteration (using the full model).**

### A tiny example (what a Bellman backup actually does)

Consider a toy MDP with two states:

- State **A**: non-terminal
- State **B**: terminal, value is 0

From state $A$, you have one action that:
- gives reward +1
- moves to $B$ with probability 1

Let the discount be $\gamma = 0.9$.

We start with an initial guess (we assign an initial value to those state):

- $V_0(A) = 0$
- $V_0(B) = 0$

Now apply **one Bellman backup** to update the value of $A$:

$$
V_{1}(A) \leftarrow R(A) + \gamma \sum_{s'} P(s' \mid A)\, V_0(s')
$$

Here, the only next state is $B$, so the sum collapses:

$$
V_{1}(A) = 1 + 0.9 \cdot V_0(B) = 1 + 0.9 \cdot 0 = 1
$$

If we apply the backup again:

$$
V_{2}(A) = 1 + 0.9 \cdot V_1(B) = 1
$$

So in this toy example, values stop changing immediately (remember that $B$ is terminal). The backup found the correct answer in one step.

**What just happened?**  
The backup took future information (the value of the next state) and pushed it one step backward.

---

### Same idea with a loop (why iteration matters)

Now imagine state $A$ transitions back to itself:

- from $A$, reward is +1
- next state is $A$ again (probability 1)

Then the Bellman equation says:

$$
V(A) = 1 + \gamma V(A)
$$

The true solution is:

$$
V(A) = \frac{1}{1-\gamma} = \frac{1}{1-0.9} = 10
$$

But if we start from $V_0(A)=0$ and apply backups:

- $V_1(A) = 1 + 0.9 \cdot 0 = 1$
- $V_2(A) = 1 + 0.9 \cdot 1 = 1.9$
- $V_3(A) = 1 + 0.9 \cdot 1.9 = 2.71$
- ...

It gradually converges toward 10.

That's why DP is "backup + iteration": when future depends on itself (loops),
you need repeated backups to propagate values until they stabilise.

In `train_gridworld_dp.py`, we will do exactly these backups, but for every state in the grid, many times.

---

### When can you use DP?

DP assumes you know the environment model:

- transition probabilities $P(s^{\prime} \mid s,a)$
- expected rewards $R(s,a)$ (or $R(s,a,s^{\prime})$)

So DP is mainly a **planning tool**.

That's why we use a small transparent `GridworldMDP` in this repo. It exposes $P$ and $R$ explicitly, so DP is easy to understand.

---

## 2) Policy evaluation (DP prediction)

### Goal

Given a fixed policy $\pi$, compute its state-value function $v_\pi(s)$.

The Bellman expectation equation says:

$$
v_\pi(s) =
\sum_a \pi(a \mid s)
\left[
R(s,a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s,a)\, v_\pi(s^{\prime})
\right]
$$

### Iterative policy evaluation (the DP way)

You can solve the Bellman equation as a linear system for tiny problems.
But DP usually does this instead:

1. start with some guess $v_0(s)$ (often all zeros)
2. apply the Bellman expectation backup repeatedly to produce $v_{k+1}$ from $v_k$
3. stop when updates are small

A synchronous update is:

$$
v_{k+1}(s) \leftarrow
\sum_a \pi(a \mid s)
\left[
R(s,a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s,a)\, v_k(s^{\prime})
\right]
$$

Synchronous means every state uses the previous table $v_k$.

In code, that typically means either:
- using a separate array `V_new`, or
- being careful if you update in-place.

---

## 3) Policy improvement (the greedy step)

Once you have $v_\pi$, you can build a better policy by acting greedily.

Define a one-step lookahead action-value (using the model):

$$
q_\pi(s,a) = R(s,a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s,a)\, v_\pi(s^{\prime})
$$

Then improve greedily:

$$
\pi^{\prime}(s) = \arg\max_a q_{\pi}(s,a)
$$

This is not just a heuristic: greedy improvement gives a policy that is never worse (it improves or keeps the same value).

---

## 4) Policy iteration (DP control)

Policy iteration alternates:

1) **Policy evaluation**: compute $v_{\pi}$ for current $\pi$  
2) **Policy improvement**: set $\pi \leftarrow \text{greedy}(v_{\pi})$

Repeat until the policy stops changing.

In this repo, this is implemented in:
- `src/rl_algorithms_guide/tabular/policy_iteration.py`

### Modified policy iteration (a practical idea)

Full policy evaluation can be expensive.
A common trick is:

- evaluate only a few sweeps of evaluation (or stop early)
- then improve
- repeat

This blends policy iteration and value iteration.

It's also a good mental bridge to deep RL later. In practice we often do approximate evaluation.

---

## 5) Value iteration (DP control)

Value iteration skips full evaluation and applies the optimality backup directly:

$$
v_{k+1}(s) \leftarrow
\max_a \left[
R(s,a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s,a)\, v_k(s^{\prime})
\right]
$$

Intuition that usually helps: start with $v_0 = 0$.

After values converge, extract a greedy policy:

$$
\pi_{\ast}(s) = \arg\max_a \left[
R(s,a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s,a)\, v_{\ast}(s^{\prime})
\right]
$$

In this repo:
- `src/rl_algorithms_guide/tabular/value_iteration.py`

**Good to know:** during value iteration, intermediate $v_k$ may not correspond to any real policy.
That's normal. The algorithm is still guaranteed to converge.

---

## 6) Why do these DP methods converge? (intuition)

The reason "backup + iteration" works is that, when $\gamma < 1$, Bellman backups are stable:
each sweep shrinks the difference between two value estimates.

Intuition:
- the future is discounted by $\gamma$
- so any mistake you make about the future gets multiplied by at most $\gamma$
- meaning repeated backups can't "blow up", they settle to a unique answer

That unique answer is exactly the value function you want:
- for a fixed policy, the backups converge to $v_{\pi}$
- for the optimality backup, they converge to $v_{\ast}$

In more formal terms, Bellman backup operators are $\gamma$-contractions when $\gamma < 1$, which guarantees a unique fixed point.

---

## 7) DP: synchronous vs asynchronous

So far we described synchronous DP (sweep all states).

You can also do **asynchronous** DP:
update states one at a time, in any order.

Practical forms you'll see:
- in-place updates (reuse the same array instead of `V_new`)
- prioritized sweeping (update "important" states first)
- real-time DP (update only states you encounter)

We don't implement these yet, but the idea matters because it shows up again later
(e.g. replay buffers and prioritized replay are "DP-flavoured" ideas).

---

## 8) Monte Carlo (model-free prediction)

Now assume you do **not** know $P$ and $R$. You can only collect experience by acting.

Monte Carlo (MC) evaluates a policy by averaging **complete returns**.

### MC evaluation idea

Run episodes following $\pi$. For each visited state $S_t$, you can compute the return:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-t-1} R_T
$$

Then update toward that sampled return:

$$
V(S_t) \leftarrow V(S_t) + \alpha \left(G_t - V(S_t)\right)
$$

**Key feature:** MC does not bootstrap. It waits until the end, then uses the real return.

Note: TD is easier to understand if you first understand what MC is doing.
TD is basically "MC, but you don't wait until the end".

---

## 9) TD learning (model-free prediction)

As in MC, we do not know $P$ and $R$.\
We only observe experience tuples like:

$$
(S_t, A_t, R_{t+1}, S_{t+1})
$$

The goal in prediction is still estimate $V(s)$ for a fixed policy $\pi$.

---

### TD(0): the one-step TD method

TD learns from partial experience.\
The simplest TD method is called **TD(0)**.\
It is one-step because it looks one step ahead and then bootstraps from your current estimate of the next state value.\
Instead of waiting until the end of the episode to compute the full return,
it builds a learning target using:

- the reward you just saw: $R_{t+1}$
- plus your current guess of what comes next: $V(S_{t+1})$ (estimate of the next state)

That second part is called bootstrapping: you use an estimate to help update another estimate.

The TD(0) target is:

$$
\text{target}_{\text{TD(0)}} = R_{t+1} + \gamma V(S_{t+1})
$$

So the update becomes:

$$
V(S_t) \leftarrow V(S_t) + \alpha \left(
R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
\right)
$$

The quantity inside parentheses is the **TD error**:

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

Why this is useful:

- It updates **online** (after every step).
- It doesn't need the episode to finish.
- It's usually lower-variance than Monte Carlo because it only depends on one transition.

Why it's called "(0)":
- It's the special case of TD($\lambda$) with $\lambda = 0$ (explained below).

---

### MC vs TD (the practical differences)

A simple way to remember:

- **MC**:
  - must wait until termination (episodic tasks)
  - target is "true return" (the full return to the end of the episode) $\to$ unbiased sample of the true return, but high variance
  - doesn't need the Markov property as strongly (works even if your "state" is imperfect)

- **TD**:
  - updates online, step-by-step and in continuing tasks
  - target uses a bootstrap estimate $\to$ lower variance, but introduces bias because it bootstraps from an estimate
  - exploits the Markov structure (usually more efficient when the state is truly Markov)

#### What bias and variance mean here

In RL, we use targets to update value estimates (like $G_t$ for MC, or $R_{t+1} + \gamma V(S_{t+1})$ for TD).

- **Variance** answers: if I repeat the same experiment with different random episodes, how much does my target bounce around?
  High variance usually means noisier learning curves and slower, less stable learning.

- **Bias** answers: is my target systematically shifted away from the true value, even after many samples?
  Bias can make learning confidently wrong if the bootstrap estimate is poor.

Why this maps to MC vs TD:

- **MC** uses the full return $G_t$. That return is an unbiased sample of the true expected return,
  but it can vary a lot depending on how the rest of the episode unfolds $\to$ **high variance**.

- **TD** uses a bootstrapped target $R_{t+1} + \gamma V(S_{t+1})$.
  This is less noisy because it only looks one step ahead, but if $V$ is inaccurate early on,
  the target inherits that error $\to$ **some bias**.

A good mental image: MC updates can be noisy but honest, TD updates can be smoother but trust your own current guess.

---

### n-step TD: a bridge between TD(0) and MC

TD(0) uses **1 step** of real rewards before bootstrapping.

You can generalise this idea to **n-step returns**:

- look ahead $n$ rewards,
- then bootstrap from $V(S_{t+n})$.

The n-step return is:

$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
$$

Then the update is:

$$
V(S_t) \leftarrow V(S_t) + \alpha\left(G_t^{(n)} - V(S_t)\right)
$$

Two important extremes:

- If $n = 1$, then $G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$ $\to$ TD(0).
- If $n$ goes to the end of the episode, bootstrapping disappears $\to$ it becomes Monte Carlo.

So n-step TD lets you trade off:
- small $n$ $\to$ more bootstrapping (more bias, less variance)
- large $n$ $\to$ less bootstrapping (less bias, more variance)

**Naming clarification (important):**

- **TD(0)** is the one-step TD method. The "(0)" does not mean "0 steps".\
  It comes from the TD($\lambda$) family: TD(0) is the special case $\lambda = 0$.

- If you use an **n-step return** (like $n=6$), that method is called n-step TD.\
  People do not call it TD(n), and it is not TD(0).

So:
- **TD(0)** = 1-step TD update  
- **n-step TD** = choose a fixed horizon $n$ and bootstrap at $t+n$  
- **TD($\lambda$)** = instead of choosing one fixed $n$, mix many n-step targets together (next section, after the example)

---

### Numerical example: TD(0) vs 4-step TD on the same trajectory

This example is about prediction (estimating a value function $V(s)$ for a fixed policy).
We compare:

- **TD(0)** (1-step TD)
- **4-step TD** (n-step TD with $n=4$)

The point is to see how the targets differ numerically.

---

### 1) Setup

We observe a trajectory (episode fragment) under a fixed policy:

$$
S_0=A,\; S_1=B,\; S_2=C,\; S_3=D,\; S_4=E,\; S_5=F
$$

Rewards received after each transition:

- $A \to B$: $R_1 = +1$
- $B \to C$: $R_2 = 0$
- $C \to D$: $R_3 = +2$
- $D \to E$: $R_4 = -1$
- $E \to F$: $R_5 = +0.5$

Hyperparameters:

- discount: $\gamma = 0.9$
- step size: $\alpha = 0.1$

Current value estimates (our guesses, not true values):

$$
V(A)=0.5,\; V(B)=0.2,\; V(C)=0.0,\; V(D)=-0.1,\; V(E)=1.0,\; V(F)=0.4
$$

Note: these numbers are just an example of a current estimate table.\
In practice, you often start with $V(s)=0$ (zeros initialization) for all states, 
or random/small values (sometimes used to break symmetry).

---

### 2) TD(0): update $V(A)$ at time $t=0$

### TD(0) target

TD(0) uses a 1-step target:

$$
\text{target}_{\text{TD(0)}} = R_{t+1} + \gamma V(S_{t+1})
$$

At $t=0$, $S_0=A$, $S_1=B$, so:

$$
\text{target}_{\text{TD(0)}} = R_1 + \gamma V(B)
= 1 + 0.9 \cdot 0.2
= 1 + 0.18
= 1.18
$$

### TD(0) update

$$
V(A) \leftarrow V(A) + \alpha(\text{target} - V(A))
$$

$$
V(A) \leftarrow 0.5 + 0.1(1.18 - 0.5)
= 0.5 + 0.1(0.68)
= 0.5 + 0.068
= 0.568
$$

After TD(0), $V(A)$ becomes 0.568.\
After this, we need to update $V(B)$ for $t=1$. It is the same thing that we just did for $V(A)$ but now we have:

$$
V(B) \leftarrow V(B) + \alpha(R_{2} + \gamma V(C) - V(B))
$$

---

### 3) 4-step TD: update $V(A)$ at time $t=0$

### What 4-step TD means

The 4-step return starting at time $t$ is:

$$
G_t^{(4)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \gamma^4 V(S_{t+4})
$$

So it uses:
- 4 real rewards,
- then bootstraps from the value estimate 4 steps ahead.

At $t=0$, we need $R_1,R_2,R_3,R_4$ and $V(S_4)=V(E)$.\
That means: you can only form this target after observing 4 transitions (up to state $E$).

### 4-step target $G_0^{(4)}$

$$
G_0^{(4)} = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 + \gamma^4 V(E)
$$

Compute term by term:

- $R_1 = 1$
- $\gamma R_2 = 0.9 \cdot 0 = 0$
- $\gamma^2 R_3 = 0.9^2 \cdot 2 = 0.81 \cdot 2 = 1.62$
- $\gamma^3 R_4 = 0.9^3 \cdot (-1) = 0.729 \cdot (-1) = -0.729$
- $\gamma^4 V(E) = 0.9^4 \cdot 1.0 = 0.6561$

Sum:

$$
G_0^{(4)} = 1 + 0 + 1.62 - 0.729 + 0.6561 = 2.5471
$$

### 4-step TD update

$$
V(A) \leftarrow 0.5 + 0.1(2.5471 - 0.5)
= 0.5 + 0.1(2.0471)
= 0.5 + 0.20471
= 0.70471
$$

After 4-step TD, $V(A)$ becomes 0.70471.\
After this, we can compute $V(B)$ for $t=1$ (I do this just to make n-step TD more clear, 
but it's the same idea as for $V(A)$).\
For $t=1$, we update $S_1=B$ using $R_2,R_3,R_4,R_5$ and $V(S_5)=V(F)$.\
That requires seeing one more transition (up to $F$).

$$
G_1^{(4)} = R_2 + \gamma R_3 + \gamma^2 R_4 + \gamma^3 R_5 + \gamma^4 V(F)
$$

Plug numbers:

* $R_2 = 0$
* $\gamma R_3 = 0.9 \cdot 2 = 1.8$
* $\gamma^2 R_4 = 0.81 \cdot (-1) = -0.81$
* $\gamma^3 R_5 = 0.729 \cdot 0.5 = 0.3645$
* $\gamma^4 V(F) = 0.6561 \cdot 0.4 = 0.26244$

Sum:

$$
G_1^{(4)} = 0 + 1.8 - 0.81 + 0.3645 + 0.26244 = 1.61694
$$

Update $V(B)$ (current $V(B)=0.2$):

$$
V(B) \leftarrow 0.2 + 0.1(1.61694 - 0.2)
= 0.2 + 0.1(1.41694)
= 0.2 + 0.141694
= 0.341694
$$

$V(B)$ moves from 0.2 $\to$ 0.341694.

---

### 4) Compare TD(0) vs 4-step TD for $V(A)$

The only difference is the target:

- TD(0) target:  
  $$
  1.18
  $$
  uses just $R_1$ and bootstraps immediately from $V(B)$

- 4-step target:  
  $$
  2.5471
  $$
  uses $R_1,R_2,R_3,R_4$ and bootstraps later from $V(E)$

So in this trajectory:

- TD(0) update: $V(A): 0.5 \to 0.568$
- 4-step update: $V(A): 0.5 \to 0.70471$

#### Why the 4-step target is much larger here

Because it sees the positive reward $R_3=+2$ inside its 4-step window.\
TD(0) does not see that reward yet; it only looks one step ahead.

---

### 5) Summary: where n-step methods sit between TD and MC

- TD(0): fast, online, low variance, but relies heavily on the current estimates.
- Larger n (like 4-step): incorporates more real rewards, less immediate bootstrapping.
- If n reaches the end of the episode, bootstrapping disappears $\to$ it becomes Monte Carlo for that time step.

So n-step TD is a knob between TD(0) and MC.

---

### TD($\lambda$): mix all n-step returns (concept only)

Earlier we saw two extremes for the learning target:

- **TD(0)**: look 1 step ahead, then bootstrap  
  (fast updates, low variance, but more bias early on)

- **Monte Carlo**: look all the way to the end of the episode  
  (no bootstrap bias, but high variance and delayed updates)

TD($\lambda$) is a clean way to get something in between.

#### The core idea

Instead of committing to one fixed $n$, like TD(0) (1-step TD) or 5-step TD,
TD($\lambda$) builds a target that is a **mixture of all n-step returns**:

- some weight on 1-step return
- some weight on 2-step return
- some weight on 3-step return
- ...
- and so on

The parameter $\lambda \in [0,1]$ controls how much you trust:
- short lookahead targets (more bootstrap)
- vs longer lookahead targets (more real return)

So TD($\lambda$) doesn't force you to pick one single $n$, but blends them, with weights controlled by $\lambda \in [0,1]$.

#### What the weights mean

A standard definition of the $\lambda$-return is:

$$
G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
$$

This is just a weighted average.

The weights are:

- weight on $G_t^{(1)}$ is $(1-\lambda)$
- weight on $G_t^{(2)}$ is $(1-\lambda)\lambda$
- weight on $G_t^{(3)}$ is $(1-\lambda)\lambda^2$
- ...

So they form a geometric sequence:
short returns get more weight when $\lambda$ is small,
long returns get more weight when $\lambda$ is large.

##### Tiny example of weights

If $\lambda = 0.5$:

- 1-step return gets weight: $0.5$
- 2-step return gets weight: $0.25$
- 3-step return gets weight: $0.125$
- ...

So most of the target is still short lookahead.

If $\lambda = 0.9$:

- 1-step return weight: $0.1$
- 2-step: $0.09$
- 3-step: $0.081$
- ...

Now long returns keep a lot of weight, so you're closer to Monte Carlo.

#### How the update looks

Once you define the target $G_t^\lambda$, you update like usual:

$$
V(S_t) \leftarrow V(S_t) + \alpha\left(G_t^\lambda - V(S_t)\right)
$$

#### What $\lambda$ does (the intuition)

- $\lambda = 0$  
  Only the 1-step return survives $\to$ **TD(0)**

- $\lambda \to 1$ (episodic tasks)  
  Longer returns dominate $\to$ approaches **Monte Carlo** (and in continuing tasks, 
  you typically rely on $\gamma < 1$ to keep returns finite)

So TD($\lambda$) is literally a dial between TD and MC.

We are not implementing TD($\lambda$) in this repo right now, but it's worth knowing because:
- it explains where TD(0) comes from (it's just $\lambda=0$)
- multi-step targets show up again later (DQN variants, actor-critic tricks, etc.)

---

### Eligibility traces (how TD($\lambda$) is done online)

The $\lambda$-return definition above is nice on paper,
but it's awkward to compute online because it seems to require mixing many n-step returns (Computing $G_t^\lambda$ directly is not convenient online).

Eligibility traces are the practical trick that makes TD($\lambda$) work online:
they keep track of how much credit recent states should receive.

#### The problem traces solve

If you only update the current state $S_t$, learning can be slow:
reward information takes time to move backward to earlier states.

Traces let you do something more natural:
> when a reward arrives, update not only the current state, but also recently visited states, with decreasing strength.

That is "credit assignment over time".

#### What a trace is

For each state $s$, keep a number $e(s)$ called its trace.
Think of $e(s)$ as:

> how "eligible" state $s$ is to receive updates right now.

Rules of thumb:

- when you visit a state, its trace increases (it becomes highly eligible)
- as time passes, traces decay (older states become less eligible)

A typical decay looks like:

$$
e(s) \leftarrow \gamma \lambda \, e(s)
$$

So traces fade faster when:
- $\gamma$ is smaller (far future matters less)
- $\lambda$ is smaller (we prefer shorter lookahead)

#### The online TD($\lambda$) update (high level)

At each time step:

1) Compute the usual TD(0) error:

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

2) Update traces (conceptually):
- decay all traces by $\gamma\lambda$
- bump up the trace for the current state

3) Update values for *all* states proportionally to their trace:

$$
V(s) \leftarrow V(s) + \alpha \, \delta_t \, e(s)
$$

#### Intuition: why this matches TD($\lambda$)

- If $\lambda = 0$, traces die immediately:
  only the current state gets updated $\to$ TD(0)

- If $\lambda$ is large, traces last longer:
  many past states get updated $\to$ closer to Monte Carlo-style "spread credit backward"

When you get a reward at the end of an episode, traces let you 'send some of that learning signal' backward to the last few states.

So eligibility traces are basically the online mechanism behind the idea "mix short and long lookaheads".

Again: concept-only here, no implementation in this block.

---

#### A tiny numeric example (what traces do)

A helpful mental image: traces are like footprints.\
Every state you visit leaves a footprint, and the footprint fades over time. 
When a learning signal arrives (the TD error), you update recent states more and older states less.

Imagine you visit three states in a row:

$$
S_0 = A,\quad S_1 = B,\quad S_2 = C
$$

and then at time $t=2$ you receive a reward (so you compute a TD error $\delta_2$).

Let's pick simple values:

- learning rate: $\alpha = 0.1$
- discount: $\gamma = 0.9$
- trace parameter: $\lambda = 0.8$

Then the trace decay factor is:

$$
\gamma \lambda = 0.9 \cdot 0.8 = 0.72
$$

We'll use the common accumulating traces intuition:
- every time you visit a state, its trace gets bumped up (think "+1")
- then traces decay each step by multiplying by $0.72$

Start with all traces at 0:

- at $t=0$ (visit $A$):  
  $e(A)=1$
- at $t=1$ (move to $B$):
  - decay old traces: $e(A) \leftarrow 0.72 \cdot 1 = 0.72$
  - bump $B$: $e(B)=1$
- at $t=2$ (move to $C$):
  - decay: $e(A) \leftarrow 0.72 \cdot 0.72 = 0.5184$
  - decay: $e(B) \leftarrow 0.72 \cdot 1 = 0.72$
  - bump $C$: $e(C)=1$

So right when we compute $\delta_2$, the traces are:

- $e(C)=1$ (most recent state)
- $e(B)=0.72$
- $e(A)=0.5184$ (older state, smaller trace)

Now we apply the TD($\lambda$) update:

$$
V(s) \leftarrow V(s) + \alpha\, \delta_2\, e(s)
$$

Suppose the TD error is $\delta_2 = +2$ (positive surprise).

Then each state gets updated by a different amount:

- $\Delta V(C) = 0.1 \cdot 2 \cdot 1 = 0.2$
- $\Delta V(B) = 0.1 \cdot 2 \cdot 0.72 = 0.144$
- $\Delta V(A) = 0.1 \cdot 2 \cdot 0.5184 \approx 0.104$

**What this shows:**
- the newest state gets the biggest update
- earlier states still get credit, but less and less as they get further in the past
- that's exactly "send the learning signal backward", smoothly controlled by $\lambda$

If you set $\lambda = 0$, then $\gamma\lambda = 0$ and traces die immediately, 
only the current state would get updated $\to$ TD(0).

---

## 10) From prediction to control: Generalised Policy Iteration

Control means you don't just evaluate a fixed policy, you improve it.

The general loop is:

- **(some) evaluation**
- **(some) improvement**
- repeat

This is called **Generalised Policy Iteration (GPI)**.

DP, MC control, and TD control are all "GPI with different evaluation machinery".

One important detail: control needs exploration.
A common choice is an $\epsilon$-greedy behaviour policy over $Q$.

---

## 11) TD Control (learning action-values without a model)

So far, TD methods were prediction: estimate values for a fixed policy.

Control means we also want to improve the policy while learning.

The common pattern is the same GPI loop:

1) estimate values (approximately)
2) act more greedily with respect to those values
3) repeat

In control, we want a policy that chooses good actions.

A common approach is to learn action-values:

- $Q(s,a)$ estimates how good it is to take action $a$ in state $s$.

Then we act with an exploration strategy (often $\epsilon$-greedy).

---

### Reminder: Policies (what we mean by a policy)

A policy is a rule for choosing actions in each state.

- Deterministic:
  $$
  a = \pi(s)
  $$
- Stochastic:
  $$
  \pi(a \mid s) = P(A_t=a \mid S_t=s)
  $$

In tabular control, we often do not store $\pi$ directly.\
Instead, we learn action-values and derive a policy from them.

---

### Why action-values $Q(s,a)$ are useful in control

We learn:

- $Q(s,a)$: how good it is to take action $a$ in state $s$, assuming we keep following some policy afterwards.

If we knew perfect action-values, the optimal greedy choice would be:

$$
\pi_{\text{greedy}}(s) = \arg\max_a Q(s,a)
$$

So learning a good $Q$ is enough to get a good policy.

---

### Exploration: why we don't act greedily all the time

Early in learning, $Q$ is wrong.\
If we always take $\arg\max$ from a wrong table, we may never discover better actions.

So during training we usually behave with an exploratory policy, often $\epsilon$-greedy:

- with prob $1-\epsilon$: choose $\arg\max_a Q(s,a)$
- with prob $\epsilon$: choose a random action

This gives a practical balance between:
- **exploitation**: use what we currently believe is best
- **exploration**: try other actions to learn better values

---

### The generic TD control update pattern

All TD control methods update an action-value estimate with:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big(\text{target} - Q(S_t, A_t)\Big)
$$

Same shape, different target.

The crucial difference between SARSA, Expected SARSA, and Q-learning (we are going to see them below) is:

> **what they use as the target for the next step**  

in particular, how they treat the next action at $S_{t+1}$.

That difference is exactly where the terms on-policy and off-policy come from.

---

### on-policy vs off-policy (behaviour vs target policy)

In TD control, we learn from experience tuples $(S_t, A_t, R_{t+1}, S_{t+1})$, 
and we are not just predicting values, we are improving a policy while learning action-values $Q(s,a)$.

But there is a subtle point:

> The experience tells you what action you took.
> The update tells you what policy you are learning the value of.
> Same data, different targets, potentially different value functions.

To make this precise, we always separate two policies:

---

#### 1) Behaviour policy $\mu$: the policy that generates the data

$\mu$ is the policy you actually follow to interact with the environment.
It answers: **"How did we pick actions while collecting this trajectory?"**

So if you say: "during training I act $\epsilon$-greedily w.r.t. $Q$",
then that $\epsilon$-greedy rule is your behaviour policy $\mu$.

Formally, the trajectory distribution depends on $\mu$:

$$
A_t \sim \mu(\cdot \mid S_t)
$$

- In practice, $\mu$ is often $\epsilon$-greedy w.r.t. the current $Q$:
  - with probability $1-\epsilon$: pick $ \arg\max_a Q(s,a) $
  - with probability $\epsilon$: pick a random action

---

#### 2) Target policy $\pi$: the policy whose value function you want to learn

$\pi$ is the policy you are trying to evaluate/improve.
It answers: **"Which policy is this update estimating $q_\pi$ for?"**

This matters because $Q(s,a)$ is not just "how good is $(s,a)$" in isolation.
It is:

$$
Q(s,a) \approx q_\pi(s,a) = \mathbb{E}_\pi\left[G_t \mid S_t=s, A_t=a\right]
$$

i.e., the expected return **if you take $a$ now and then continue following $\pi$**.

So whenever you update $Q(s,a)$, you are implicitly choosing a $\pi$
(the "continue following ..." part).

---

### The key question (the real meaning of on/off-policy)

After observing $(S_t, A_t, R_{t+1}, S_{t+1})$, your update must decide:

> From $S_{t+1}$ onward, which policy am I assuming the agent will follow?

- If the update assumes **the same policy that generated the data** ($\pi=\mu$),
  then you are learning the value of your *actual behaviour*.
  This is **on-policy** learning.

- If the update assumes a **different policy** ($\pi \neq \mu$),
  then you are learning the value of another policy
  (often greedier / more optimal) while still collecting data with $\mu$.
  This is **off-policy** learning.

Formally:

- **On-policy:** $\pi = \mu$  
  Learn $q_\mu$ (values of the behaviour policy).

- **Off-policy:** $\pi \neq \mu$  
  Learn $q_\pi$ for some other target policy $\pi$ (often $\pi_{\text{greedy}}$).

---

### How you can *see* the difference in the target

The difference shows up exactly in the target.

- **On-policy targets** use the next action distribution under $\mu$  
  (either by sampling $A_{t+1}\sim \mu$ like SARSA,
  or averaging over $\mu$ like Expected SARSA).

- **Off-policy targets** use the next action according to the target policy $\pi$,
  even if $\mu$ generated the data.
  In Q-learning, $\pi$ is the greedy policy, so the target uses a max.

This is why it is not about exploration vs no exploration.
You can explore in both cases.
It is about **what policy the update is evaluating**.

---

### One intuition that helps a lot

Think of it as two different questions:

- **On-policy:**  
  "Given that I will keep behaving with $\epsilon$-greedy exploration, how good is $(s,a)$?"

- **Off-policy:**  
  "Even if I explore right now, how good is $(s,a)$ assuming that from next step onward I act greedily/optimally?"

Same data, different question, different target.

---

### A quick summary

Exploration forces you to sometimes take non-greedy actions. The key question is:

- Do we want our learned $Q$ to reflect the consequences of those exploratory actions as part of the policy?
  - If yes $\to$ **on-policy** (SARSA, Expected SARSA)
- Or do we want to learn the value of the greedy/optimal policy even while exploring?
  - If yes $\to$ **off-policy** (Q-learning)

| Method | Behaviour policy $\mu$ | Target policy $\pi$ | What it tends to learn                |
|---|---|---|---------------------------------------|
| SARSA | $\epsilon$-greedy | same as behaviour | $q_\mu$ (value of exploratory policy) |
| Expected SARSA | $\epsilon$-greedy | same as behaviour | $q_\mu$ (but less noisy)              |
| Q-learning | $\epsilon$-greedy | greedy | $q_{\ast}$ (optimal action-values)    |

Now we'll see three ways to define the target: 
- sample under $\mu$ (SARSA) 
$$
\text{target}_{\text{SARSA}} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})
$$
- expectation under $\mu$ (Expected SARSA)
$$
\text{target}_{\text{Expected SARSA}} = R_{t+1} + \gamma \sum_{a^{\prime}} \mu(a^{\prime} \mid S_{t+1})\, Q(S_{t+1}, a^{\prime})
$$
- max under greedy $\pi$ (Q-learning)
$$
\text{target}_{\text{Q}} = R_{t+1} + \gamma \max_{a^{\prime}} Q(S_{t+1},a^{\prime})
$$

---

## SARSA (on-policy TD control)

### Update rule

SARSA updates using the action it **actually takes next**:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[
R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
\right]
$$

The TD error is:

$$
\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
$$

### Why this is on-policy

Because $A_{t+1}$ is sampled from the **same** policy you use to behave:

- you choose $A_t \sim \mu(\cdot \mid S_t)$
- you choose $A_{t+1} \sim \mu(\cdot \mid S_{t+1})$
- and your target contains $Q(S_{t+1}, A_{t+1})$

So SARSA is learning:

> How good is $(s,a)$ if I keep behaving with this same $\epsilon$-greedy policy?

That's exactly $q_\mu$.

### Practical consequence

Because SARSA's target includes the *actual next action*, if exploration sometimes takes risky/bad actions, SARSA will feel that risk in its estimates.

This often makes SARSA:
- **more conservative**
- sometimes safer in environments where exploratory actions can be costly (classic example: cliff walking)

In this repo:
- `src/rl_algorithms_guide/tabular/sarsa.py`

---

## Q-learning (off-policy TD control)

### Update rule

Q-learning updates using the **greedy** action at the next state in the target:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[
R_{t+1} + \gamma \max_{a^{\prime}} Q(S_{t+1}, a^{\prime}) - Q(S_t, A_t)
\right]
$$

TD error:

$$
\delta_t = R_{t+1} + \gamma \max_{a^{\prime}} Q(S_{t+1}, a^{\prime}) - Q(S_t, A_t)
$$

### Why this is off-policy

Even if you behave $\epsilon$-greedily (to explore), the update target assumes that from $S_{t+1}$ onward you will take the greedy action.

So:
- behaviour: $\mu$ (exploratory, $\epsilon$-greedy)
- target: $\pi$ (greedy)

Hence $\pi \neq \mu$: **off-policy**.

### What Q-learning is really doing

It is trying to push $Q$ toward satisfying the Bellman optimality equation:

$$
q_{\ast}(s,a) = \mathbb{E}\big[R_{t+1} + \gamma \max_{a^{\prime}} q_{\ast}(S_{t+1},a^{\prime}) \mid S_t=s, A_t=a\big]
$$

In tabular settings, under standard conditions, Q-learning converges to $q_{\ast}$.

In deep RL, this off-policy bootstrapping + function approximation can become unstable
(that's why DQN needs tricks like replay buffers plus target networks).

In this repo:
- `src/rl_algorithms_guide/tabular/q_learning.py`

---

## A tiny example that makes SARSA vs Q-learning "click”

We focus on one update step at time $t$.\
Suppose we observed:

- $S_t = s$, $A_t = a$
- reward: $R_{t+1} = 0$
- next state: $S_{t+1} = s^{\prime}$
- learning rate: $\alpha = 1$ (to make numbers simple)
- discount: $\gamma = 1$

Assume current estimates:

- $Q(s,a) = 0$
- At $s^{\prime}$, two actions exist: $a_1, a_2$
- $Q(s^{\prime}, a_1) = 10$ (great)
- $Q(s^{\prime}, a_2) = -10$ (terrible)

Now assume your behaviour policy is $\epsilon$-greedy and this time it explores at $s^{\prime}$,
choosing the bad action:

- $A_{t+1} = a_2$

Before computing targets, let's see the two policies:

- Behaviour policy $\mu$: we are acting $\epsilon$-greedily, and at $s^{\prime}$ we happened to explore,
  so it sampled $A_{t+1}=a_2$.
- Target policy $\pi$:
  - for SARSA: $\pi=\mu$ (keep following the same $\epsilon$-greedy policy)
  - for Q-learning: $\pi=\pi_{\text{greedy}}$ (assume greedy behaviour from $s^{\prime}$ onward)

So even though the data is the same, the updates are estimating two different quantities:
- SARSA estimates $q_\mu(s,a)$
- Q-learning pushes toward $q_{\ast}(s,a)$ (optimal values)

### SARSA target (on-policy)

SARSA uses the actual next action:

$$
\text{target}_{\text{SARSA}} = R_{t+1} + \gamma Q(s^{\prime}, A_{t+1})
= 0 + 1 \cdot Q(s^{\prime}, a_2) = -10
$$

Update with $\alpha=1$:

$$
Q(s,a) \leftarrow -10
$$

**Interpretation:**  
SARSA learns: "If I keep behaving $\epsilon$-greedily, sometimes I'll take $a_2$ at $s^{\prime}$, and that's bad."  
So it becomes more pessimistic/conservative.

### Q-learning target (off-policy)

Q-learning ignores the exploratory action and uses the greedy one:

$$
\text{target}_{\text{Q}} = R_{t+1} + \gamma \max_{a^{\prime}} Q(s^{\prime},a^{\prime})
= 0 + 1 \cdot \max(10, -10) = 10
$$

Update with $\alpha=1$:

$$
Q(s,a) \leftarrow 10
$$

**Interpretation:**  
Q-learning learns: "Even if I explore now, what matters is the value if I act greedily from $s^{\prime}$."  
So it pushes toward the optimal policy's values.

### This is the entire on-policy vs off-policy difference (again)

- SARSA target uses $Q(s^{\prime}, A_{t+1})$ where $A_{t+1} \sim \mu$ $\rightarrow$ learns $q_\mu$ (**on-policy**)\
  SARSA is on-policy because its target assumes the agent will keep following the same policy used to act ($\pi=\mu$),
  so the update learns values that include the consequences of exploration.
- Q-learning target uses $\max_{a^{\prime}} Q(s^{\prime},a^{\prime})$ (greedy action from $\pi$) $\rightarrow$ learns $q_{\ast}$ (**off-policy**)\
  Q-learning is off-policy because its target assumes a different policy from the one used to act ($\pi \neq \mu$):
  you may act $\epsilon$-greedily, but the update evaluates the greedy/optimal continuation from $s^{\prime}$.

---

### Expected SARSA (on-policy, less noise)

SARSA uses the next action it actually sampled (the next action that actually happens), $A_{t+1}$:

$$
\text{SARSA target} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})
$$

But if you explore (e.g. with $\epsilon$-greedy), then $A_{t+1}$ is random.
So the SARSA target can bounce around a lot (high variance),
especially early in training, depending on which exploratory action was sampled.

**Expected SARSA** keeps the same on-policy idea, but instead of using one sampled next action,
it uses the *expected* next value under the behaviour policy $\mu$.

---

#### Update rule

Expected SARSA uses:

$$
\text{Expected SARSA target} =
R_{t+1} + \gamma \sum_{a^{\prime}} \mu(a^{\prime} \mid S_{t+1})\, Q(S_{t+1}, a^{\prime})
$$

So the update is:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[
R_{t+1} + \gamma \sum_{a^{\prime}} \mu(a^{\prime} \mid S_{t+1}) Q(S_{t+1}, a^{\prime}) - Q(S_t, A_t)
\right]
$$

This is often less noisy than SARSA because you average over actions.

---

#### What expected means in practice

Expected here means:

> take the expectation over the next action $a^{\prime}$ that your behaviour policy $\mu$ would choose at $S_{t+1}$.

It does not mean an expectation over next states (that would require knowing the model).
You still use the actual sampled next state $S_{t+1}$. The only averaging is over actions.

So the Expected SARSA target is:

$$
R_{t+1} + \gamma \sum_{a^{\prime}} \mu(a^{\prime} \mid S_{t+1})\, Q(S_{t+1}, a^{\prime})
$$

You can read the sum as a weighted average:

- take each possible next action $a^{\prime}$
- weight its $Q(S_{t+1}, a^{\prime})$ by how likely your behaviour policy is to take it
- add them up

A useful view is:

$$
\mathbb{E}_{a^{\prime} \sim \mu(\cdot \mid S_{t+1})}[Q(S_{t+1}, a')] = \sum_{a^{\prime}} \mu(a^{\prime} \mid S_{t+1})\, Q(S_{t+1}, a^{\prime}) = \sum_{a^{\prime}} p_{a^{\prime}} \, Q_{a^{\prime}}
$$

this is a dot product between the probability vector and the Q-values.

---

#### Why it is still on-policy

Expected SARSA evaluates the same policy that generated the data:

- behaviour policy: $\mu$ (e.g. $\epsilon$-greedy)
- target policy: $\pi = \mu$

Even though it doesn't use the sampled $A_{t+1}$, it still uses $\mu(\cdot \mid S_{t+1})$ in the target.
So Expected SARSA is learning $q_\mu$, like SARSA, but with less variance.

---

#### If $\mu$ is $\epsilon$-greedy, what are the probabilities?

Let:
- $m$ = number of actions
- $a^{\ast} = \arg\max_a Q(S_{t+1}, a)$ be the greedy action

Then an $\epsilon$-greedy policy assigns:

$$
\mu(a^{\ast} \mid s) = 1 - \epsilon + \frac{\epsilon}{m}
$$

and for every other action $a \neq a^{\ast}$:

$$
\mu(a \mid s) = \frac{\epsilon}{m}
$$

So the expectation becomes:

$$
\sum_{a^{\prime}} \mu(a^{\prime} \mid s^{\prime}) Q(s^{\prime},a^{\prime})
=
\left(1-\epsilon+\frac{\epsilon}{m}\right)Q(s^{\prime},a^{\ast}) + \sum_{a \neq a^{\ast}}\frac{\epsilon}{m}Q(s^{\prime},a)
$$

This makes the behaviour very intuitive:

- if $\epsilon \to 0$, almost all weight goes to the greedy action $a^{\ast}$ $\to$ close to Q-learning's greedy next value
- if $\epsilon \to 1$, you average almost uniformly over all actions $\to$ very exploratory evaluation

Quick note, if there are $k$ greedy actions, then for each greedy action we have:
$$
\mu(a^{\ast} \mid s) = \frac{1 - \epsilon}{k} + \frac{\epsilon}{m}
$$

---

#### A tiny numeric example

Suppose at $S_{t+1}$ you have $m=4$ actions with:

$$
Q(S_{t+1}, \cdot) = [10,\ 2,\ 0,\ -1]
$$
Greedy action is the first one (value $10$), and $\epsilon = 0.1$.

Then:
$$
\mu(a^*) = 1 - 0.1 + \frac{0.1}{4} = 0.925,
\qquad
\mu(a \neq a^*) = \frac{0.1}{4} = 0.025
$$

Expected next value:

$$
0.925\cdot 10 + 0.025\cdot 2 + 0.025\cdot 0 + 0.025\cdot (-1) = 9.275
$$

So Expected SARSA uses this smoothed number ($9.275$) as the “next value” instead of:
- SARSA: one sampled $Q(S_{t+1},A_{t+1})$ (noisier)
- Q-learning: $\max_a Q(S_{t+1},a)=10$ (greedy, off-policy)

---

#### Relationship to SARSA and Q-learning

- **SARSA**: sample the next action $A_{t+1}$ under $\mu$ $\Rightarrow$ on-policy, but noisy.\
  Learn what happens when I keep behaving with my exploratory policy.\
  If I sometimes do random actions, SARSA learns a policy that is safe under that reality.
- **Expected SARSA**: average over next actions under $\mu$ $\Rightarrow$ on-policy, less noisy.\
  Instead of one random next action, average over what my policy would do next.
- **Q-learning**: use the greedy $\max_{a^{\prime}} Q(S_{t+1},a^{\prime})$ $\Rightarrow$ off-policy, learns toward $q_{\ast}$.\
  Learn what would happen if I acted greedily from the next state onward.\
  I can explore while learning, but the update is pushing toward the greedy/optimal policy.

In this repo:
- `src/rl_algorithms_guide/tabular/expected_sarsa.py`

---

## 12) Why SARSA and Q-learning behave differently (CliffWalking intuition)

CliffWalking makes the on-policy vs off-policy difference visible.

### What the environment teaches you
There is a short path right next to the cliff.\
It is optimal if you always take the correct greedy moves.

But during learning, you often behave $\epsilon$-greedily.\
Even if the greedy action is "safe", with probability $\epsilon$ you take a random action.

Near the cliff, a single random wrong step can produce a catastrophic penalty.

### Why SARSA looks safer
SARSA learns the value of the behaviour policy $\mu$ (which includes exploration).\
So if your behaviour occasionally takes random actions, SARSA will bake in the fact that:

> being close to the cliff is risky under my current behaviour

That tends to push the learned policy toward a slightly longer but safer route.

### Why Q-learning hugs the cliff
Q-learning learns the value of the greedy target policy.\
Its update assumes that from the next state onward you behave optimally (via the max operator),
even though in reality you will still explore during training.

So Q-learning is pulled toward the shortest greedy route,
even if that route performs poorly during training due to exploration accidents.

In summary:
- **Q-learning** learns values for the greedy target policy.
  It often learns a path close to the cliff because that's the shortest path.

- **SARSA** learns values for the exploratory behaviour policy.
  If your behaviour is $\epsilon$-greedy, SARSA knows it might occasionally take a random step,
  so it tends to prefer safer routes with less catastrophic downside.

In this repo, you can see both effects clearly, especially if you print policies and look at the cliff-fall-rate plot:
- `examples/02_tabular/train_cliffwalking.py`

---

## 13) Practical pitfalls

- **Terminal handling**:
  when the episode ends, don't bootstrap.
  The target becomes just $R_{t+1}$, not $R_{t+1} + \gamma \cdot \text{something}$.

- **Exploration matters more than you think**:
  too little $\epsilon$ can freeze learning early;
  too much $\epsilon$ can prevent convergence to a good policy.

- **Learning rate $\alpha$**:
  if $\alpha$ is too large, values bounce around;
  if $\alpha$ is too small, learning is painfully slow.

- **Tie-breaking in argmax**:
  if several actions have the same value early on, random tie-breaking helps exploration.

- **Don't trust a single run**:
  tabular methods can still vary by seed.
  Averaging multiple runs gives much more reliable curves.

---

## 14) Where this block appears in the repo

**DP on a tiny transparent MDP**
- `src/rl_algorithms_guide/tabular/gridworld.py`
- `src/rl_algorithms_guide/tabular/value_iteration.py`
- `src/rl_algorithms_guide/tabular/policy_iteration.py`
- `examples/02_tabular/train_gridworld_dp.py`

**TD control on Gymnasium CliffWalking**
- `src/rl_algorithms_guide/tabular/sarsa.py`
- `src/rl_algorithms_guide/tabular/q_learning.py`
- `src/rl_algorithms_guide/tabular/expected_sarsa.py`
- `examples/02_tabular/train_cliffwalking.py`

---

## 15) Bridge to the next blocks

Once state spaces get large, tabular tables stop working.
The next steps are:

- replace tables with function approximators (neural nets) $\to$ **DQN**
- move beyond value-based methods $\to$ **policy gradients / actor-critic**
