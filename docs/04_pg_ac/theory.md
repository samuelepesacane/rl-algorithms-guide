# Policy Gradients (REINFORCE + Actor-Critic)

This note goes with the code in `src/rl_algorithms_guide/pg_ac/`.

In the previous blocks, we mostly learned **values** (like $V(s)$ or $Q(s,a)$) and then acted greedy.
Here we switch perspective: we learn the **policy directly**.

The price we pay is that gradients are noisy.
So this block is mostly about:

- how to write the policy objective cleanly
- how to take its gradient without differentiating through the environment
- how to reduce variance so learning actually works

---

## 0) Why policy gradients?

Value-based methods pick actions via something like:

$$
a = \arg\max_a Q(s,a)
$$

That's clean for **discrete** actions, but becomes awkward when:
- actions are **continuous** (no easy argmax)
- you want a **stochastic** policy (useful for exploration and robustness)
- you want to optimize a policy class directly (Gaussian, squashed Gaussian, etc.)

Policy gradients do something conceptually simpler:

> Increase the probability of actions that lead to high return, decrease the probability of actions that lead to low return.

Everything else in this document is just making that idea correct and practical.

---

## 1) What we optimize: stochastic policies and trajectories

### 1.1 Stochastic policy

A stochastic policy is a conditional distribution:

$$
\pi_\theta(a \mid s)
$$

- **Discrete** actions: categorical distribution (softmax over logits).
- **Continuous** actions: often Gaussian with mean from a network, e.g.
  $a \sim \mathcal{N}(\mu_\theta(s), \Sigma)$.

The key quantity we will backprop through is:

$$
\log \pi_\theta(a \mid s)
$$

because policy gradients are gradients of log-probabilities.

---

### 1.2 Trajectories and the induced distribution $p_\theta(\tau)$

An episode / rollout / trajectory is:

$$
\tau = (s_0,a_0,r_1,s_1,a_1,r_2,\dots,s_{T})
$$

Assume:
- initial state distribution $\rho_0(s_0)$
- environment transition dynamics $P(s_{t+1}\mid s_t,a_t)$

Then the probability of a whole trajectory under $\pi_\theta$ is:

$$
p_\theta(\tau)
=
\rho_0(s_0)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t)
$$

Important detail:
- the environment term $P(\cdot)$ does **not** depend on $\theta$,
- only the policy term $\pi_\theta(a_t\mid s_t)$ depends on $\theta$.

So we can compute policy gradients **without** differentiating through the environment.

---

## 2) Returns and the objective $J(\theta)$

Define the (discounted) reward-to-go from time $t$:

$$
G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}
$$

The standard objective is expected return from the start:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[G_0\right]
$$

On CartPole, rewards are typically +1 per step until termination, so $G_0$ is basically the episode length
(up to discounting). That makes it a great debugging environment since "return goes up" is easy to interpret.  
Note: in the training scripts we plot the undiscounted episode return (plain sum of rewards),
because it is easy to interpret on CartPole. The policy update, instead, uses discounted reward-to-go
returns $G_t$.

---

## 3) From the objective to the gradient

### From expected return to a usable gradient

We have a clean goal:

$$
\text{maximize } J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[G_0(\tau)]
$$

Now we want to improve the policy, that is change the network weights $\theta$ so that $J(\theta)$ goes up.

That means we need a direction to move $\theta$. In calculus, that direction is the gradient:

$$
\nabla_\theta J(\theta)
$$

If this were supervised learning, we would have something like:

$$
J(\theta) = \mathbb{E}[f_\theta(x)]
$$

and we could just differentiate $f_\theta$ directly.

In RL the tricky part is that $\theta$ is not only inside some function, but it also is inside
the distribution we are sampling from:
- we sample trajectories $\tau$ from $p_\theta(\tau)$
- and then we evaluate their return $G_0(\tau)$

$\theta$ affects which trajectories are likely to happen, not just how we score them.  
So we can't just treat this like a normal supervised-learning expectation where only the inside depends on $\theta$.

To analyse this situation better, we will rewrite the expectation as an explicit sum/integral over trajectories.

---

### 3.1 Write the expectation as a sum/integral over trajectories

An expectation w.r.t. $p_\theta(\tau)$ is:

- discrete case: $J(\theta) = \sum_\tau p_\theta(\tau)\,G_0(\tau)$
- continuous case: $J(\theta) = \int p_\theta(\tau)\,G_0(\tau)\,d\tau$

We'll write the integral form (it covers both):

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[G_0(\tau)] = \int p_\theta(\tau)\,G_0(\tau)\,d\tau
$$

- $p_\theta(\tau)$ = how likely the policy is to generate trajectory $\tau$
- $G_0(\tau)$ = how good that trajectory was
- the integral = average goodness over all possible trajectories

---

### 3.2 Differentiate: the gradient hits the trajectory probability

Take the gradient:

$$
\nabla_\theta J(\theta)
= \nabla_\theta \int p_\theta(\tau)\,G_0(\tau)\,d\tau
$$

Assuming we can swap gradient and integral (standard in this context):

$$
\nabla_\theta J(\theta)
= \int \nabla_\theta p_\theta(\tau)\,G_0(\tau)\,d\tau
$$

Key observation:
- $G_0(\tau)$ is a number once the trajectory is fixed
- the $\theta$-dependence is in **$p_\theta(\tau)$**

So we reduced the problem to: *how do we deal with $\nabla_\theta p_\theta(\tau)$?*

---

### 3.3 The log-likelihood trick (a.k.a. score function identity)

We want $\nabla_\theta J(\theta)$, but the expectation depends on $\theta$ through $p_\theta(\tau)$.

A useful identity is:

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)
$$

Why it's true:

$$
\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
\quad \Rightarrow \quad
\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)
$$

Substitute this into the gradient:

$$
\nabla_\theta J(\theta)
= \int p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)\,G_0(\tau)\,d\tau
$$

Now rewrite the integral back as an expectation:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[
\nabla_\theta \log p_\theta(\tau)\,G_0
\right]
$$

This is already a usable form, because it says:

> sample trajectories $\tau$, compute $G_0$, compute $\nabla_\theta \log p_\theta(\tau)$, average.

But we still need to simplify $\log p_\theta(\tau)$.

Now, let's expand $\log p_\theta(\tau)$ and watch the environment disappear into the void.

From the trajectory probability formula above we have:

$$
p_\theta(\tau)
=
\rho_0(s_0)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t)
$$

Take logs (products become sums):

$$
\log p_\theta(\tau)
=
\log\rho_0(s_0) +
\sum_{t=0}^{T-1}\log\pi_\theta(a_t\mid s_t) +
\sum_{t=0}^{T-1}\log P(s_{t+1}\mid s_t,a_t)
$$

Now take $\nabla_\theta$:

- $\rho_0$ does not depend on $\theta$
- $P(\cdot)$ does not depend on $\theta$
- only $\pi_\theta$ depends on $\theta$

So:

$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

Plug this back into the gradient:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}\left[
\left(\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right)\,G_0
\right]
$$

That's the basic policy gradient estimator.  

---

### 3.4 What this formula means

Inside the expectation, for each time step $t$:

- $\nabla_\theta \log\pi_\theta(a_t\mid s_t)$ says:
  how should I change $\theta$ to make the taken action $a_t$ more likely in state $s_t$?
- $G_0$ says:
  was this whole trajectory good or bad?

So the update does exactly trial-and-error:

- if $G_0$ is large $\rightarrow$ increase probability of actions in that trajectory
- if $G_0$ is small $\rightarrow$ decrease probability of actions in that trajectory

This is correct, but the estimator has high variance.

---

## 4) Variance problem: why the correct estimator is painful

The estimator above uses **the same multiplier $G_0$ for every action** in the trajectory.
That means:

- early actions get credited/blamed for things they could not control
- lucky initial states can dominate updates
- random events late in the episode can flip the sign of credit for earlier actions

With enough samples it averages out, but it may require a lot of samples.

So the next sections are about making the estimator lower variance without changing its expectation (or by introducing controlled bias).

---

## 5) Causality and reward-to-go (the first big fix)

### 5.1 The idea

Action $a_t$ cannot affect rewards that happened before time $t$.

So instead of weighting $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ by the full return $G_0$,
we weight it by **reward-to-go** $G_t$:

$$
\nabla_\theta J(\theta)
= \mathbb{E}\left[
\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t
\right]
$$

This usually reduces variance a lot.

### 5.2 Why it's still unbiased

The part of the return that comes from rewards before $t$ is independent of $a_t$ once you condition on the past.
When you push the expectation through carefully (law of iterated expectations),
those "past reward" terms multiply an expectation of $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ which becomes zero.
So dropping them does not change the expected gradient, but it removes noise.

You do not need to re-derive this every time; just remember: **causality lets you use reward-to-go**.

---

## 6) Discounting: "absolute" vs "relative" discounting

Now we know that for an action at time $t$, only rewards after $t$ can be caused by that action,
so we weight $\nabla_\theta \log \pi_\theta(a_t\mid s_t)$ by a reward-to-go.

That fixes the big problem (variance from blaming early actions for past rewards).

If this fixed everything we could wrap this up and move on to REINFORCE and Actor-Critic.  
As you can see, as the name of this section suggests, we still have other things to say before wrap this up.

Now there is a more subtle question:

> Once we use reward-to-go, how should the discount $\gamma$ be applied?

This matters because it changes how much learning signal late time steps receive.  
In episodic tasks, late actions are often the ones most directly responsible for the final outcome
(e.g., the last few steps before CartPole falls), so shrinking their gradients too much can slow learning.

In policy gradients, there are two equally valid ways to place the discount power,
depending on whether discounting is measured from:


1) the **episode start** (a global clock) $\to$ **Absolute discounting**
2) the **current time step** (a local clock)$\to$ **Relative discounting**

remember that the reward-to-go is defined as:
$$
G_t
=
\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}
$$

This definition is the local-clock one (discount starts from now, from time $t$).

---

### 6.1 Relative discounting (local clock)

This is exactly the reward-to-go we already use:

$$
G_t^{\text{local}}
=
\sum_{t^{\prime}=t}^{T-1}\gamma^{t^{\prime}-t} r_{t^{\prime}+1}
\quad\;\;(\text{same as } G_t)
$$

This is the same as the $G_t$ we defined before $\left(\text{the one with } \sum_{k=0}^{T-t-1}\right)$,
we just wrote the index in a different way.  
Interpretation: from step $t$, rewards further in the future count less.  
So at any point in the episode, the policy prefers actions that lead to sooner rewards from that point onward.

This is also the form you naturally get if you compute returns in code via backward recursion:

- start from the end with $G=0$
- go backwards: $G \leftarrow r_{t+1} + \gamma G$

In practice, you almost always want this relative form.  
It's called relative discounting because rewards are discounted relative to the current time step $t$.  

With this discounting we can write the gradient as:

$$
\nabla_\theta J(\theta)
= \mathbb{E}\left[
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,
\left(\sum_{t^{\prime}=t}^{T-1}\gamma^{t^{\prime}-t}r_{t^{\prime}+1}\right)
\right]
$$

---

### 6.2 Absolute discounting (global clock)

Here discounting is measured from the start of the episode, using the episode time index directly:

$$
G_t^{\text{global}}
=
\sum_{t^{\prime}=t}^{T-1}\gamma^{t^{\prime}} r_{t^{\prime}+1}
$$

So rewards are discounted not only by how far they are in the future, but also by how late we are in the episode.  
This tends to make the policy care much more about early parts of the episode than late parts,
even if those late states are important.

The important connection is:

$$
G_t^{\text{global}} = \gamma^t\,G_t^{\text{local}}
$$

So the global-clock version is just the local-clock version multiplied by an extra $\gamma^t$.

This extra factor can make late-step gradients smaller when episodes are long and $\gamma<1$.

With this discounting we can write the gradient as:

$$
\nabla_\theta J(\theta)
= \mathbb{E}\left[
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,
\left(\sum_{t^{\prime}=t}^{T-1}\gamma^{t^{\prime}}r_{t^{\prime}+1}\right)
\right]
$$

---

### 6.3 Tiny numeric example

Let $T=3$, $\gamma=0.9$, rewards $(r_1,r_2,r_3)=(0,0,1)$ (only the final reward is non-zero).

Local clock:

- $G_0^{\text{local}} = 0.9^2\cdot 1 = 0.81$
- $G_1^{\text{local}} = 0.9^1\cdot 1 = 0.9$
- $G_2^{\text{local}} = 1$

Global clock (multiply by $\gamma^t$):

- $G_0^{\text{global}} = 0.9^0\cdot 0.81 = 0.81$
- $G_1^{\text{global}} = 0.9^1\cdot 0.9 = 0.81$
- $G_2^{\text{global}} = 0.9^2\cdot 1 = 0.81$

So:
- with the local clock, later steps get larger weights (0.81 $\to$ 0.9 $\to$ 1),
- with the global clock, later steps get an extra shrink factor $\gamma^t$.

---

### Practical rule of thumb

- If you want to match the objective $J(\theta)=\mathbb{E}[\sum_t \gamma^t r_{t+1}]$ as literally as possible,
  then the global-clock form $G_t^{\text{global}}$ is the most faithful.
  (or compute $G_t^{\text{local}}$ and multiply by $\gamma^t$)

- In many episodic tasks (like CartPole), people often use $G_t^{\text{local}}$ directly (no extra $\gamma^t$),
  because it keeps late-step learning signals from becoming too small.

Both are reasonable. The important thing is to be consistent about which convention you are using.

---

## 7) Policy gradients as weighted maximum likelihood

Up to now we derived a gradient expression like:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}\left[
\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\text{(some score for step $t$)}
\right]
$$

That is mathematically correct, but it can still feel abstract. So, in this section, we will show that
policy gradients are basically **maximum likelihood**, except the labels are the actions we sampled,
and each sample is **weighted** by how good it turned out.

After this, the code pattern becomes obvious: compute log-probs, multiply by weights, average, backprop.

---

### 7.1 Supervised learning: maximum likelihood

In supervised learning you have data pairs $(s, a)$:
- $s$ is the input (a state, features, an image, etc.)
- $a$ is the target label (the correct action/class)

You pick a probabilistic model $\pi_\theta(a\mid s)$ and you want it to give high probability to the correct labels.

The standard objective is maximum likelihood:

$$
\max_\theta\;\mathbb{E}[\log \pi_\theta(a\mid s)]
$$

In deep learning we usually do gradient descent, so we minimize negative log-likelihood:

$$
\mathcal{L}_{\text{NLL}}(\theta)
=
-\mathbb{E}[\log \pi_\theta(a\mid s)]
$$

If you have a batch, the loss is just an average:

$$
\mathcal{L}_{\text{NLL}}(\theta)
\approx
-\frac{1}{N}\sum_{i=1}^N \log \pi_\theta(a^{(i)}\mid s^{(i)})
$$

So in code, it looks like maximize log-prob of the target action.

---

### 7.2 The RL problem: we don't have labels

In RL we also have states and actions, but we don't have correct labels.

We only know what happened after we took actions:
- sometimes it was good (high return)
- sometimes it was bad (low return)

So the best we can do is:

> Make the actions that led to good outcomes **more likely**,  
> and the actions that led to bad outcomes **less likely**.

This is the same as supervised learning, except that we need a number that says
how good (or bad) was this action, so we can scale the update.

That number is the weight.

---

### 7.3 Weighted maximum likelihood: the policy gradient loss

Take the supervised loss and add a weight $\hat{w}$ (weighted maximum likelihood):

$$
\mathcal{L}_{\text{WML}}(\theta)
=
-\mathbb{E}\left[\log \pi_\theta(a_t\mid s_t)\,\hat{w}_t\right]
$$

This is exactly the standard policy gradient loss (just written as Weighted Maximum Likelihood Estimation):

$$
\mathcal{L}_{\text{PG}}(\theta)
=
-\mathbb{E}\left[\log \pi_\theta(a_t\mid s_t)\,\hat{w}_t\right]
$$

where $\hat{w}_t$ is some estimate of how good that decision was:
- if $\hat{w}_t$ is **positive**, gradient descent will try to **increase** $\log\pi_\theta(a_t\mid s_t)$,
  so the action becomes more likely
- if $\hat{w}_t$ is **negative**, gradient descent will try to **decrease** $\log\pi_\theta(a_t\mid s_t)$,
  so the action becomes less likely
- if $\hat{w}_t = 0$, that sample contributes nothing

So the whole algorithm boils down to choose a good weight.

---

### 7.4 What can the weight be?

Different policy gradient methods differ mostly in the choice of $\hat{w}_t$.

Common choices:

- **REINFORCE (Monte Carlo)**: use return / reward-to-go
  $$
  \hat{w}_t = G_t
  $$

- **Baseline form**: subtract a baseline to reduce variance
  $$
  \hat{w}_t = G_t - b(s_t)
  $$

- **Advantage form**: use an advantage estimate (often the cleanest choice)
  $$
  \hat{w}_t = \hat{A}_t
  $$

In all cases, the loss is the same shape; only the weights change.

---

### 7.5 A tiny example

Suppose at some state $s$ your policy picked action $a$.

- Episode A: it worked well, $\hat{w}=+3$
- Episode B: it worked badly, $\hat{w}=-2$

Then the loss contributes:

- Episode A: $-\log\pi_\theta(a\mid s)\cdot 3$  $\to$ pushes $\pi_\theta(a\mid s)$ **up**
- Episode B: $-\log\pi_\theta(a\mid s)\cdot (-2)$ $\to$ pushes $\pi_\theta(a\mid s)$ **down**

So the same action can be reinforced or discouraged depending on the outcome.
That is exactly what we want.

---

### 7.6 How this maps to code

This is the pattern you will see in every policy gradient implementation in this repo:

1) compute log-probabilities for the taken actions:

   - `logp = log πθ(a_t | s_t)`

2) compute weights (returns / advantages):

   - `weights = G_t`  (or `G_t - baseline`, or `A_t`)

3) compute a scalar loss and backprop:

$$
\mathcal{L} = -\mathbb{E}[\log p \cdot \text{weights}]
$$

In code:

- `loss = -(logp * weights).mean()`
- `loss.backward()`

---

### 7.7 One very important gotcha: do NOT backprop through the weights

The weights are computed from rewards and value estimates.
They are a training signal, not something we want to optimize through.

So in practice you treat them as constants:
- no gradient should flow through $G_t$,
- no gradient should flow through $\hat{A}_t$ when updating the actor.

In PyTorch terms, you typically ensure weights are detached (or computed under `torch.no_grad()`).

That is exactly what the code in this block will do.

---

## 8) Baselines: reduce variance without changing the expected gradient

Up to now, our policy update looks like:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}\left[
\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t
\right]
$$

This is still noisy: two identical state-action choices can get very different $G_t$ just because
the rest of the episode went differently.


> Can we change the weight $G_t$ into something less noisy **without changing what the gradient means on average**?

The trick is surprisingly simple: subtract something that does not depend on the action.

---

### 8.1 The baseline trick

Imagine you are judging how good an action was at state $s_t$.

- $G_t$ is like a "final score from now on".
- But some states are **already** good or bad regardless of what action you take:
  - near the goal: almost any action will still get decent return,
  - near failure: almost any action will still get poor return.

So instead of using the raw score $G_t$, we compare it to a reference score for that state:

> "Was this outcome better or worse than what I typically get from here?"

That reference is called a **baseline**.

Formally, pick any baseline $b_t$ that does **not** depend on the action.
It can be state-dependent, like $b(s_t)$, or even a single scalar shared across the episode, and use:

$$
G_t - b(s_t)
$$

Then the policy gradient becomes:

$$
\nabla_\theta J(\theta)
= \mathbb{E}\left[
\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\left(G_t - b(s_t)\right)
\right]
$$

Important:
- subtracting a baseline **does not change the expected gradient**
- but it can reduce variance a lot

In this repo's REINFORCE implementation, the baseline is a **simple scalar running mean**
(Exponential Moving Average, EMA, of the episode return). That means $b_t=b$ is the same for every time step in the episode.
It is still a valid baseline because it does not depend on the action.
EMA is a running average where recent episodes count more than older ones.

If $R_k$ is the episode return at episode $k$, an EMA baseline is typically updated like:

$$
b_k = (1-\beta),b_{k-1} + \beta,R_k
$$

* $b_k$: current baseline estimate
* $R_k$: current episode return
* $\beta \in (0,1]$: smoothing factor (bigger $\beta$ = reacts faster)

So EMA of the episode return means: a smooth baseline that tracks the typical return over time, without jumping around too much.

---

### 8.2 Why this is safe (the baseline does not bias the gradient)

The reason is: the baseline does not depend on the action, and policy gradients have a special property:

> Averaged over actions sampled from the policy,  
> $\nabla_\theta \log \pi_\theta(a\mid s)$ sums to zero.

Take the baseline term only:

$$
\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\right]
$$

Condition on a fixed state $s$ (so $b(s)$ is just a constant):

$$
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
\left[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\right]
=
b(s)\,
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
\left[\nabla_\theta\log\pi_\theta(a\mid s)\right]
$$

Now use the identity:

$$
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
\left[\nabla_\theta\log\pi_\theta(a\mid s)\right]
=
\sum_a \pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)
=
\sum_a \nabla_\theta \pi_\theta(a\mid s)
=
\nabla_\theta \sum_a \pi_\theta(a\mid s)
=
\nabla_\theta 1
=
0
$$

So the baseline term is zero in expectation, meaning it does not change the expected gradient.
It only removes noise.

---

### 8.3 A tiny example (why variance goes down)

Suppose you are at the same state $s$ twice:

- Trajectory A: you eventually get $G_t = 50$
- Trajectory B: you eventually get $G_t = 10$

If from this state you "usually" get about $30$, set $b(s)=30$.

Then your learning signals become:
- A: $G_t - b(s) = 50 - 30 = +20$ (good, reinforce)
- B: $G_t - b(s) = 10 - 30 = -20$ (bad, discourage)

Instead of dealing with huge raw returns, you deal with a centered signal around 0.
That is exactly what reduces variance.

---

## 9) Advantage: the centered learning signal you actually want

Section 8 said: "subtract any state-only baseline to reduce variance".

Now the natural question is:

> What is the best baseline to subtract?

Intuitively, the best baseline at state $s$ is:

> The return you expect **on average** from that state, if you keep following your policy.

That is exactly the **value function**.

---

### 9.1 The value baseline

Define the on-policy state value:

$$
V^\pi(s) = \mathbb{E}\left[G_t \mid s_t=s\right]
$$

So if we choose:

$$
b(s_t)=V^\pi(s_t)
$$

then the learning signal becomes:

$$
G_t - V^\pi(s_t)
$$

Interpretation:
- $V^\pi(s_t)$ is "what I normally get from here"
- $G_t$ is "what I actually got this time"
- the difference says whether the taken action helped or hurt relative to normal

This difference is an estimate of the **advantage** (it is a Monte Carlo estimate of the advantage).

---

### 9.2 What "advantage" means

Define:
- $Q^\pi(s,a)$: expected return if you take action $a$ in $s$, then follow $\pi$ $\to$ how good if I take $a$ now and then follow $\pi$
- $V^\pi(s)$: expected return if you follow $\pi$ from $s$ (averaging over actions) $\to$ how good is this state on average under $\pi$

Then:

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

Meaning:
- $A^\pi(s,a) > 0$ : this action is better than average for this state
- $A^\pi(s,a) < 0$ : this action is worse than average for this state
- $A^\pi(s,a) = 0$ : this action is exactly average

So advantage is not "how good the state is".
It is "how much better this action is than my default behavior in this state".

That is why advantage is the perfect weight for policy gradients:
- $A>0$ $\to$ increase probability of that action
- $A<0$ $\to$ decrease probability of that action

---

### 9.3 The clean policy gradient form (what we actually want to approximate)

If we could compute the true advantage, the nicest form of the policy gradient is:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}\left[
\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^\pi(s_t,a_t)
\right]
$$

In practice, we cannot compute $A^\pi$ exactly, so we estimate it:

- Monte Carlo: $\hat{A}_t \approx G_t - V_\phi(s_t)$
- TD(0): $\hat{A}_t \approx \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

Actor-Critic uses the TD version, which we will introduce in Section 11.

---

### 9.4 A tiny example (what advantage does that raw return doesn't)

Suppose from state $s$ your policy usually gets about $V^\pi(s)=30$.

Two different actions happen:

- You take $a_1$ and get $G_t=35$ → advantage $= +5$ (slightly better than usual)
- You take $a_2$ and get $G_t=10$ → advantage $= -20$ (much worse than usual)

Even though both returns are "numbers", advantage tells you the only thing the policy update cares about:

> Did this action beat the baseline or not?

So the policy learns "prefer $a_1$ over $a_2$ in this state" without being distracted by the absolute scale of returns.

---

## 10) REINFORCE (Monte Carlo policy gradient)

At this point we have all the building blocks:

- a stochastic policy $\pi_\theta(a\mid s)$
- a policy gradient loss that looks like weighted maximum likelihood
- weights that can be returns, baselines, or advantages

REINFORCE is the "minimal" algorithm that puts these pieces together with the most direct choice:

> Use **Monte Carlo** returns as the weight.

No critic, no bootstrapping, just run an episode, see how it went, and reinforce the actions that worked.

---

### 10.1 What REINFORCE is doing

REINFORCE repeats this loop:

1) **Collect an episode on-policy**  
   Run the current policy $\pi_\theta$ in the environment and record:
   $$(s_t, a_t, r_{t+1}) \quad \text{for } t=0,\dots,T-1$$

2) **Compute reward-to-go (Monte Carlo returns)**  
   For each time step $t$, compute:

   $$
   G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}
   $$

   In code this is usually done by going backwards:
   - start with $G=0$,
   - update $G \leftarrow r_{t+1} + \gamma G$.

3) **Turn it into a weighted log-likelihood loss**  
   Use the same skeleton from Section 7:

   $$
   \mathcal{L}_{\text{REINFORCE}}(\theta)
   =
   -\mathbb{E}\left[\sum_{t=0}^{T-1} \log\pi_\theta(a_t\mid s_t)\,G_t\right]
   $$

   Optionally, use a baseline:

   $$
   \mathcal{L}(\theta)
   =
   -\mathbb{E}\left[\sum_{t=0}^{T-1} \log\pi_\theta(a_t\mid s_t)\,(G_t - b(s_t))\right]
   $$

4) **Update the policy parameters**  
   Gradient descent on the loss is equivalent to gradient ascent on return.

   Gradient-ascent update:

   $$
   \theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t
   $$

---

### 10.2 Why REINFORCE is 'correct' (and why it can still feel bad)

**Why it is correct:**  
Monte Carlo returns are unbiased estimates of the expected return.
So the policy gradient estimator is unbiased.

**Why it can still feel bad in practice:**  
The weights $G_t$ can vary wildly from episode to episode, especially early in training.
So the update direction is very noisy.

---

### 10.3 Pros and cons

Pros:
- **unbiased** gradient estimator
- minimal moving parts (easy to implement, easy to debug)
- a great baseline algorithm for understanding policy gradients

Cons:
- **high variance** (needs many episodes)
- the simplest version updates only after the episode ends
- learning can be slow or unstable when rewards are sparse

So the natural next step is:

> Keep the same policy-gradient loss shape, but replace noisy Monte Carlo returns with a **learned baseline**.

That is Actor-Critic.

---

## 11) Actor-Critic: reduce variance with a learned value function

REINFORCE's problem is not that it is 'wrong'.  
It is that the learning signal $G_t$ is too noisy.

Actor-Critic keeps the same core idea:

> The actor still does weighted maximum likelihood.  
> We just build a critic that provides a better (lower-variance) weight.

You can think of the Actor as the same as REINFORCE, but now we add a helper: the critic.
The critic will let us use smart weights instead of the full return.

So Actor-Critic splits responsibilities:

- **Actor**: the policy $\pi_\theta(a\mid s)$ chooses actions
- **Critic**: a value estimator $V_\phi(s)$ predicts how good the state is (expected future return)

The critic is trained to answer:

> From this state, how much return remains if I keep following the current policy?

A simple analogy:

Think of a robot (or a game character) learning like a kid learning to play a game.  
It has two helpers inside its head:
- Actor = the one who chooses what to do
- Critic = the one who judges if that choice was good or bad (better or worse than expected)

So:
- Actor says: I'll do this!
- Critic says: "Nice move (gg)" or "Bad move (git gud)"

That's why it's called Actor-Critic.

---

### 11.1 The critic: TD error as an advantage estimate

A simple way to estimate "how good was this action compared to normal?" is:

- predict what you expected before acting: $V_\phi(s_t)$
- observe a reward and what you expect after acting: $r_{t+1} + \gamma V_\phi(s_{t+1})$

Their difference is the **TD error**:

$$
\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

This is exactly TD(0) from the tabular block, but now $V_\phi$ is a neural network.

Interpretation:
- if $\delta_t$ is positive, things turned out better than the critic expected
- if $\delta_t$ is negative, things turned out worse

This makes $\delta_t$ a cheap, online estimate of advantage (the simplest advantage estimate is the one-step TD error):

$$
\hat{A}_t \approx \delta_t
$$

So the actor loss keeps the same 'weighted log-likelihood' shape:

$$
\mathcal{L}_{\text{actor}}(\theta)
=
-\mathbb{E}\left[\log\pi_\theta(a_t\mid s_t)\,\delta_t\right]
$$

You can think of it as REINFORCE, but with "smart weights" instead of $G_t$.

---

### 11.2 How the critic learns (bootstrapping)

The critic is trained by regression: make $V_\phi(s_t)$ match a (bootstrap) target.

A one-step bootstrap target is:

$$
y_t = r_{t+1} + \gamma V_\phi(s_{t+1})
$$

Then we minimize squared error (Value loss):

$$
\mathcal{L}_{\text{value}}(\phi)
=
\mathbb{E}\left[(V_\phi(s_t) - y_t)^2\right]
$$

Terminal handling:
- if $s_{t+1}$ is terminal, there is no future value,
  so the target becomes:
  $$
  y_t = r_{t+1}
  $$

This bootstrapping is the key difference from REINFORCE:
- REINFORCE waits to see the full future return (Monte Carlo)
- Actor-Critic uses a value estimate for the future

---

### 11.3 Putting it together (what one training step looks like)

At time step $t$:

1) actor samples $a_t \sim \pi_\theta(\cdot\mid s_t)$  
2) environment returns $(r_{t+1}, s_{t+1})$  
3) critic computes $\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$  
4) actor updates using weight $\delta_t$  
5) critic updates to reduce $(V_\phi(s_t) - y_t)^2$

So actor and critic are trained together, on-policy.

---

### 11.4 Bias vs variance

- REINFORCE: unbiased, but high variance (weights are Monte Carlo returns)
- Actor-Critic: lower variance, but introduces bias if $V_\phi$ is inaccurate

In practice, that bias/variance trade-off is usually worth it:
Actor-Critic tends to learn much faster and more stably than plain REINFORCE.

---

## 12) Making actor-critic "really work": n-step and ($\gamma,\lambda$) advantage

So far we have seen two extremes for the actor weight $\hat{A}_t$:

- **Monte Carlo (REINFORCE-style):** uses full reward-to-go $G_t$  
  - low bias (in the limit), but high variance
- **1-step TD (Actor-Critic):** uses TD error $\delta_t$  
  - lower variance, but can be biased if the critic is inaccurate

> Is there a middle ground where we keep variance reasonable **without** trusting the critic too much?

Yes: use more than one step of real rewards before bootstrapping.

---

### 12.1 The problem as a spectrum (two endpoints)

**Endpoint A: 1-step TD (fast, but trusts the critic a lot)**

$$
\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

You only use one real reward and then immediately bootstrap from $V_\phi$.

**Endpoint B: full Monte Carlo (slow, but doesn't trust the critic)**

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots
$$

You use all real future rewards; no bootstrapping needed.

If you use $\delta_t$ (1-step TD), you reduce variance but can add bias.
If you use full Monte Carlo returns, you remove bias but add variance.

We want something in between.

---

### 12.2 n-step returns: wait a bit, then bootstrap

A common compromise is **n-step advantage estimates**, which mix real rewards for $n$ steps then bootstrap.

Pick an integer $n \ge 1$.

Instead of bootstrapping after 1 step, bootstrap after $n$ steps:

$$
G_t^{(n)}
=
\sum_{k=0}^{n-1}\gamma^k r_{t+k+1} +
\gamma^n V_\phi(s_{t+n})
$$

- if $n=1$, this reduces to 1-step TD target
- if $n$ goes all the way to the end of the episode, this becomes Monte Carlo

Now build an advantage estimate by subtracting the baseline:

$$
\hat{A}_t^{(n)} = G_t^{(n)} - V_\phi(s_t)
$$

- small $n$: fast updates, low variance, more bias
- large $n$: slower, higher variance, less bias

So $n$ is a knob that moves you along the bias/variance spectrum.

---

### 12.3 $(\gamma,\lambda)$ advantage: mix many n-step returns

Choosing a single $n$ is already useful, but we can do even better.

Idea:

> Instead of picking one horizon $n$, average many horizons.  
> Weight short horizons more (more stable), but still include longer horizons (less bias).

This produces a family of estimators controlled by $\lambda \in [0,1]$:

- $\lambda$ small $\to$ emphasize short-horizon bootstraps (lower variance, higher bias)
- $\lambda$ large $\to$ emphasize longer-horizon returns (lower bias, higher variance)

So we blend many n-step estimates with an exponentially decaying weight. This estimator is controlled
by a parameter $\lambda\in[0,1]$.

A convenient way to compute it uses TD errors:

$$
\hat{A}^{(\gamma,\lambda)}_t
=
\delta_t + \gamma\lambda\,\hat{A}^{(\gamma,\lambda)}_{t+1}
$$

This recursion says:
- start with the immediate TD error $\delta_t$
- then add a discounted, $\lambda$-scaled 'next-step advantage'

If $\lambda=0$:
- $\hat{A}_t^{(\gamma,0)} = \delta_t$ (pure 1-step TD)

If $\lambda \to 1$:
- you approach a Monte Carlo-like advantage (using long-horizon information)

We do **not** implement $(\gamma,\lambda)$ advantage in Block 04 to keep code minimal,
but you will see this idea again in PPO-style methods later.

---

#### 12.3.1 Tiny example (how $\lambda$ changes the advantage)

Suppose for three consecutive time steps the TD errors are:

$$
\delta_0 = 1,\quad \delta_1 = 1,\quad \delta_2 = 1
$$

and assume the episode ends after $t=2$, so the next advantage is zero.

Let $\gamma = 0.9$.  
The recursion is:

$$
\hat{A}^{(\gamma,\lambda)}_t
=
\delta_t + \gamma\lambda\,\hat{A}^{(\gamma,\lambda)}_{t+1}
$$

We compute backwards.

---

**Case 1: $\lambda = 0$ (pure 1-step TD)**

Then $\gamma\lambda = 0$, so the recursion becomes:

$$
\hat{A}_t = \delta_t
$$

So:

- $\hat{A}_2 = 1$
- $\hat{A}_1 = 1$
- $\hat{A}_0 = 1$

This is the most local / shortest-horizon version.

---

**Case 2: $\lambda = 0.5$ (mix short and longer horizons)**

Now $\gamma\lambda = 0.9 \cdot 0.5 = 0.45$.

Backward recursion:

- $\hat{A}_2 = 1$
- $\hat{A}_1 = 1 + 0.45 \cdot 1 = 1.45$
- $\hat{A}_0 = 1 + 0.45 \cdot 1.45 = 1 + 0.6525 = 1.6525$

Now each step includes some information from later TD errors.

---

**Case 3: $\lambda = 1$ (long-horizon / Monte Carlo-like)**

Now $\gamma\lambda = 0.9$.

Backward recursion:

- $\hat{A}_2 = 1$
- $\hat{A}_1 = 1 + 0.9 \cdot 1 = 1.9$
- $\hat{A}_0 = 1 + 0.9 \cdot 1.9 = 1 + 1.71 = 2.71$

This carries much more future information, so the advantage at early steps becomes larger.

---

### What this example shows

- **$\lambda=0$**: only the immediate TD error matters (low variance, more bias)
- **larger $\lambda$**: you accumulate more future TD errors (less bias, more variance)
- So $\lambda$ is a smooth knob between **short-horizon** and **long-horizon** credit assignment

---

## 13) Advantage normalization (practical trick, slightly biased)

Even with good advantage estimates, training can be unstable if advantage magnitudes vary a lot:

- early in training, returns can be very noisy
- a few outlier episodes can produce huge advantages
- then the policy update becomes too large, causing oscillations

A simple practical trick is:

> Center and scale the advantage values within each batch.

In other words, normalize advantages within a batch.

---

### 13.1 What normalization does

Normalization makes the batch of advantages look like:

- mean around 0
- standard deviation around 1

So your update step size becomes more consistent across iterations,
which often stabilizes learning.

---

### 13.2 The formula

First, compute advantage estimates $\hat{A}_t$.

Then, given a batch of advantage estimates $\{\hat{A}_t\}$, compute:

- batch mean $\mu$
- batch standard deviation $\sigma$

Then normalize:

$$
\hat{A}^{\prime}_t = \frac{\hat{A}_t - \mu}{\sigma + \varepsilon}
$$

Then you use $\hat{A}^{\prime}_t$ as the weight in the actor loss.

---

### 13.3 Is it "correct"?

This trick is technically biased because the normalization uses statistics computed from the sampled batch itself.
So the expectation of the update is not exactly the same.

In practice, it often helps a lot (stabilizes training) when batch sizes are not tiny.

In this repo, we keep normalization optional and minimal.

---

## 14) On-policy training (why we don't use replay buffers here)

So far, every formula in this document has the same hidden assumption:

> The trajectories used to compute the update were generated by the **same policy** we are updating.

This is what on-policy means.

It sounds like a small detail, but it matters a lot because the core object we differentiate is:

$$
\log \pi_\theta(a_t \mid s_t)
$$

That log-probability is only meaningful for learning if the actions you observed were actually sampled
from (roughly) the same $\pi_\theta$.

> Why can't we just do what DQN does and store experience in a replay buffer?

---

### 14.1 What on-policy means

An on-policy update loop looks like:

1) Use the current policy $\pi_\theta$ to collect data:
   $$(s_t, a_t, r_{t+1}, s_{t+1})$$

2) Compute weights for those same steps:
   - returns $G_t$
   - advantages $\hat{A}_t$ (baseline / TD error / n-step, etc.)

3) Compute the actor loss using the log-prob under the current policy:
   $$
   \mathcal{L}_{\text{actor}}(\theta)
   =
   -\mathbb{E}\left[\log\pi_\theta(a_t\mid s_t)\,\hat{A}_t\right]
   $$

4) Update $\theta$, then throw away the old data (or use it only very briefly)

So the data and the policy stay "in sync".

---

### 14.2 What goes wrong with a replay buffer (the mismatch problem)

Suppose you store transitions from an older policy $\pi_{\theta_{\text{old}}}$ in a replay buffer.

Later, your current policy is $\pi_{\theta_{\text{new}}}$.

If you now compute an update like:

$$
-\log\pi_{\theta_{\text{new}}}(a_t\mid s_t)\,\hat{A}_t
$$

but the action $a_t$ was sampled from $\pi_{\theta_{\text{old}}}$, you are mixing two different things:

- the **log-prob under the current policy** (what you are differentiating)
- outcomes generated by a **different policy** (what produced the data)

This can cause unstable or even wrong updates, because you are trying to "explain" data that did not come from
your current distribution.

In other words:
- DQN can reuse old data because it learns a value function off-policy
- plain policy gradients (REINFORCE / basic Actor-Critic) assume on-policy sampling

---

### 14.3 How make policy gradients work with old data

It is possible to use old data, but you need extra machinery, such as:

- **importance sampling** ratios to correct for policy mismatch
- **off-policy critics** and specialized algorithms

Those ideas are powerful, but they add complexity and distract from the core policy gradient picture.

So in Block 04 we stay strictly on-policy:
- collect fresh rollouts
- compute weights
- update
- repeat

---

## 15) Summary: what each algorithm is doing

This block is small on purpose.
Both algorithms share the same skeleton:

> **Policy (actor) update = weighted maximum likelihood**  
> increase probability of sampled actions if their weight is positive

They differ mainly in how they compute the weight.

---

### 15.1 REINFORCE (Monte Carlo policy gradient)

What it does:
- Collect a full episode using the current policy
- Compute Monte Carlo reward-to-go returns $G_t$
- Update the policy by weighting log-probs (negative log-likelihood) with $G_t$ (optionally minus a baseline)

Policy loss shape:

$$
\mathcal{L}_{\text{REINFORCE}}(\theta)
= -\mathbb{E}\left[\sum_{t=0}^{T-1}\log\pi_\theta(a_t\mid s_t)\,G_t\right]
$$

Optional baseline form (same expected gradient, lower variance):

$$
\mathcal{L}(\theta)
= -\mathbb{E}\left[\sum_{t=0}^{T-1}\log\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\right]
$$

Example:
- "I played an episode. If it went well, I make those actions more likely."

---

### 15.2 Actor-Critic (value critic)

What it does:
- Collect transitions on-policy (often in short rollout chunks, not necessarily full episodes)
- Learn a critic $V_\phi(s)$, via TD targets, to predict expected future return
- Actor uses the TD error $\delta_t$ as a low-variance advantage estimate
- Update actor and critic together
- Updates can happen every step or every short rollout chunk

TD error:

$$
\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Actor loss shape:

$$
\mathcal{L}_{\text{actor}}(\theta)
= -\mathbb{E}\left[\log\pi_\theta(a_t\mid s_t)\,\delta_t\right]
$$

Critic loss shape:

$$
\mathcal{L}_{\text{value}}(\phi)
=
\mathbb{E}\left[\left(V_\phi(s_t) - (r_{t+1} + \gamma V_\phi(s_{t+1}))\right)^2\right]
$$

Example:
- "Instead of waiting for the whole episode to finish, I learn a value baseline.
  The TD error tells me whether the last action was better or worse than expected."

---

## 16) A2C: Advantage Actor-Critic

Actor-Critic updates the policy and value function step by step, using a single worker (a single agent) 
interacting with a single environment.

That works, but there is a practical problem:

> Consecutive transitions are highly correlated.  
> The agent sees $s_t, s_{t+1}, s_{t+2}$ in order, and they all look similar.  
> That correlation makes gradient estimates noisy and learning less stable.

In DQN, the fix was a **replay buffer**: store experience, shuffle, sample randomly.  
But plain policy gradients are on-policy, you cannot reuse old data.

A2C's answer is different:

> Run **multiple independent environments in parallel**, collect transitions from all of them simultaneously, and average their gradients.

Because each environment starts from a different state and follows a different trajectory, the transitions in a batch are much less correlated than those from a single environment.

---

### 16.1 The core idea: parallel workers

In A2C you run $N$ environments at the same time, all sharing the same policy $\pi_\theta$ and critic $V_\phi$.

At each update step:

1) Each worker $i$ runs for $n$ steps and collects a short rollout:
   $$(s_0^{(i)}, a_0^{(i)}, r_1^{(i)}, \dots, s_n^{(i)})$$

2) Compute advantage estimates for each worker using the shared critic.

3) Average the actor and critic losses across all $N \times n$ transitions.

4) Do **one synchronised gradient update** on $\theta$ and $\phi$.

5) All workers continue from their current environment states.

The key word is **synchronised**: all workers finish their rollout, then all gradients are aggregated, then all workers continue. Nobody moves until the update is done.

---

### 16.2 Why this reduces variance

Think about what one gradient estimate looks like.

In single-worker Actor-Critic (Section 11):

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{n} \sum_{t=0}^{n-1} \nabla_\theta \log\pi_\theta(a_t\mid s_t)\,\hat{A}_t
$$

This is just one trajectory. A single lucky or unlucky episode can dominate the update.

In A2C with $N$ workers:

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{N \cdot n} \sum_{i=1}^{N}\sum_{t=0}^{n-1} \nabla_\theta \log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})\,\hat{A}_t^{(i)}
$$

Now you are averaging over $N$ independent trajectories.  
Why does averaging help? Think of it like this: if you flip a coin once, you might get heads (good episode) 
or tails (bad episode) and your one observation is very noisy. If you flip it $N$ times and take the average, 
the result will be much closer to the true probability. The same logic applies here: each worker gives you 
one noisy gradient estimate, but because the environments are independent, their noise partially cancels 
out when you average. Mathematically, the variance of an average of $N$ independent measurements shrinks 
by a factor of $N$ (search 'The law of large numbers' for more info).

---

### 16.3 The advantage estimate in A2C

A2C typically uses **n-step advantage estimates**, one per worker rollout.

For worker $i$, at time $t$, the n-step return is:

$$
G_t^{(n),(i)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k+1}^{(i)} + \gamma^n V_\phi(s_{t+n}^{(i)})
$$

Then the advantage estimate is:

$$
\hat{A}_t^{(i)} = G_t^{(n),(i)} - V_\phi(s_t^{(i)})
$$

This is exactly the same formula as before, the only change is the superscript $(i)$ tracking which worker produced the data.

---

### 16.4 The losses

The actor and critic losses have exactly the same shape as in Section 11.

**Actor loss** (averaged across workers and steps):

$$
\mathcal{L}_{\text{actor}}(\theta)
= -\frac{1}{N \cdot n}\sum_{i=1}^{N}\sum_{t=0}^{n-1}
\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})\,\hat{A}_t^{(i)}
$$

**Critic loss** (mean squared error against n-step targets):

$$
\mathcal{L}_{\text{value}}(\phi)
= \frac{1}{N \cdot n}\sum_{i=1}^{N}\sum_{t=0}^{n-1}
\left(V_\phi(s_t^{(i)}) - G_t^{(n),(i)}\right)^2
$$

**Entropy bonus** (optional, but standard in A2C):

$$
\mathcal{L}_{\text{entropy}}(\theta)
= -\frac{1}{N \cdot n}\sum_{i=1}^{N}\sum_{t=0}^{n-1}
\mathcal{H}\left[\pi_\theta(\cdot\mid s_t^{(i)})\right]
$$

where $\mathcal{H}[\pi] = -\sum_a \pi(a\mid s)\log\pi(a\mid s)$ is the entropy of the policy.

**Why entropy matters.** Entropy measures how "spread out" the policy is across actions. High entropy means 
the policy assigns similar probabilities to many actions. Low entropy (close to zero) means the policy is 
almost deterministic: it almost always picks the same action.

Adding an entropy bonus to the loss encourages the policy to stay spread out. Why would we want that?

Early in training, the policy does not yet know which actions are good. If it collapses to always picking 
one action too soon (low entropy), it will never explore other actions that might be better. 
The entropy bonus acts as a penalty against premature certainty: the policy is punished for becoming too 
confident before it has properly learned.

In practice, the bonus is small (coefficient around $0.01$). It does not prevent the policy from converging. 
It just slows down early collapse so the policy can explore enough to find the good actions first.

So, the entropy bonus encourages the policy to stay exploratory, as it penalises early collapse to a 
deterministic policy, and you weight it by a small coefficient $c_{\text{ent}}$ (often around $0.01$).

The full combined loss is:

$$
\mathcal{L}(\theta, \phi) = \mathcal{L}_{\text{actor}} + c_v \mathcal{L}_{\text{value}} - c_{\text{ent}} \mathcal{L}_{\text{entropy}}
$$

where $c_v$ and $c_{\text{ent}}$ are small coefficients.

Note the minus sign before the entropy term: we subtract it because we want to **maximise** entropy (keep the policy spread out), but our optimizer minimises the loss.

---

### 16.5 What one A2C update step looks like

Let's make the loop concrete.

```
Repeat until done:
  for each worker i = 1 ... N:
    run policy π_θ in environment i for n steps
    store (s, a, r, s_next) for each step

  for each worker i and step t:
    compute n-step return G_t^(i)
    compute advantage A_hat_t^(i) = G_t^(i) - V_phi(s_t^(i))

  compute actor loss  (log-prob * advantage, averaged)
  compute critic loss (MSE against n-step returns, averaged)
  compute entropy bonus (optional)

  one gradient update on θ and φ

  workers continue from where they left off
```

No replay buffer. No data reuse. All workers synchronise at the update step.

---

### 16.6 A2C vs Actor-Critic: what actually changed

Everything from Actor-Critic is still here. The update formulas, the TD-style advantage, and the actor and critic losses are all the same. The only difference is structural:

- **Actor-Critic**: 1 worker, step-by-step TD updates, high correlation between consecutive transitions.
- **A2C**: $N$ parallel workers, short rollout batches, lower correlation, more stable gradients.

If you set $N=1$ and $n=1$ in A2C, you recover the basic Actor-Critic.

This is also why A2C sits naturally between Actor-Critic and PPO (next block) in the progression:
- Actor-Critic introduced the actor + critic split.
- A2C showed that parallel environments stabilise training.
- PPO keeps everything from A2C and adds one more mechanism: preventing the policy from changing too much in a single update.

---

### 16.7 In this repo

Rather than spawning true OS-level parallel processes, we simulate multiple environments using Gymnasium's `SyncVectorEnv`, which runs $N$ environments in the same process and stacks their observations into a batch.

The training script accepts a `--num-envs` flag. Setting `--num-envs 1` makes A2C behave like single-worker Actor-Critic, which is a useful debugging baseline.

---

### 16.8 Tiny example: two workers on CartPole

Suppose $N=2$ workers and $n=3$ rollout steps, $\gamma=0.99$.

**Worker 1** is holding the pole well - short episode, high rewards:

| step | reward |
|------|--------|
| 0    | 1.0    |
| 1    | 1.0    |
| 2    | 1.0    |

**Worker 2** just knocked the pole over - it's picking up a new episode:

| step | reward |
|------|--------|
| 0    | 1.0    |
| 1    | 0.0    |
| 2    | 0.0    |

Say the critic estimates $V(s) \approx 3.0$ for Worker 1's states and $V(s) \approx 0.5$ for Worker 2's terminal states.

**n-step returns** (bootstrapping from $V$ at step 3):

- Worker 1: $G_0^{(3)} = 1 + 0.99 \cdot 1 + 0.99^2 \cdot 1 + 0.99^3 \cdot 3.0 \approx 1 + 0.99 + 0.98 + 2.91 = 5.88$
- Worker 2: $G_0^{(3)} = 1 + 0.99 \cdot 0 + 0.99^2 \cdot 0 + 0.99^3 \cdot 0.5 \approx 1 + 0 + 0 + 0.49 = 1.49$

**Advantages** (subtract current state value):

- Worker 1: $\hat{A}_0^{(1)} = 5.88 - 3.0 = +2.88$  → actions were better than expected, reinforce them
- Worker 2: $\hat{A}_0^{(2)} = 1.49 - 0.5 = +0.99$   → actions were also above expectation, but only slightly

The actor loss averages both advantages across all $N \times n = 6$ transitions. The single gradient step pulls the policy in the direction that both workers agree on (make good actions more likely), while their disagreement on scale partially cancels out.

Notice what happens if only Worker 2 had run: the gradient would be dominated by the bad trajectory and could push the policy in the wrong direction. With both workers averaged, Worker 1's signal stabilises the update.

---

## 17) A3C: asynchronous workers (concept only)

A3C (Asynchronous Advantage Actor-Critic) is the predecessor to A2C.

The core idea is the same, run multiple workers to decorrelate experience, but the workers operate **asynchronously** rather than synchronously.

In A3C:

- each worker has its own **local copy** of the network weights
- each worker runs its own environment, computes its own gradient, and pushes that gradient to a **shared global network**
- the global network is updated immediately upon receiving each gradient, without waiting for other workers
- after each update, the worker pulls the latest global weights and continues

So workers are never synchronised. One worker might be on update 500 while another is still computing gradients from step 10.

---

### Why A3C is not implemented here

A3C requires true multi-threading or multi-processing to be useful. In Python, that means spawning real OS threads or processes that share memory, which adds substantial complexity:

- shared-memory tensor operations are tricky and platform-dependent
- race conditions are hard to debug
- the machinery needed (Python's `multiprocessing`, CUDA-safe queues, etc.) distracts from the RL concepts

More importantly, A2C has been shown to perform comparably to A3C on most benchmarks, without the asynchronous complexity. The variance reduction from parallel environments is what matters most, not whether the updates are synchronous or not.

So this repo implements A2C (synchronous, vectorised environments) and only describes A3C conceptually.

If you want to study A3C in depth, the original paper (Mnih et al., 2016, "Asynchronous Methods for Deep Reinforcement Learning") is a good starting point.

---

## 18) Where this block appears in the repo

- REINFORCE:
  - `src/rl_algorithms_guide/pg_ac/reinforce.py`
  - `examples/04_pg_ac/train_reinforce_cartpole.py`

- Actor-Critic:
  - `src/rl_algorithms_guide/pg_ac/actor_critic.py`
  - `examples/04_pg_ac/train_actor_critic_cartpole.py`

- A2C:
  - `src/rl_algorithms_guide/pg_ac/a2c.py`
  - `examples/04_pg_ac/train_a2c_cartpole.py`
