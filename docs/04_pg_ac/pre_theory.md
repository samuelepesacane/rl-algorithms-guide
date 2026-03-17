# Preliminaries for Policy Gradients (PG)

This file is a short bridge into `docs/04_pg_ac/theory.md`.

Policy gradients (and PPO later) become much easier if there is one consistent picture of:

1) what an MDP generates (a **trajectory**)
2) how a policy induces a distribution over trajectories $p_\theta(\tau)$
3) why the objective is an **expectation** (so we optimize with samples)

The goal here is not to prove everything rigorously, but to make the symbols feel practical.

In `docs/02_tabular/mdp.md`, we explained some theory about MDPs. In this file, we will repeat some of 
those concepts and go beyond.

---

## 0) The goal

We have a policy with parameters $\theta$ (a neural network).\
We want to change $\theta$ so the agent gets more reward on average.

That average reward is an expectation, and expectations are hard to compute exactly, 
so we estimate them with samples (rollouts).

Everything below is just setting up the objects needed to:
1) write the objective as an expectation
2) take its gradient
3) make the gradient estimator not too noisy

So here we give a base to study policy gradients.

---

## 1) MDP recap 

### Time, state, action, reward
The important variables of an MDP are:
- Time steps: $t = 0, 1, 2, \dots$
- $S_t \in \mathcal{S}$: state at time $t$ (the information used to choose actions)
- $A_t \in \mathcal{A}$: action at time $t$
- $R_{t+1} \in \mathbb{R}$: reward observed after acting


**Why $R_{t+1}$ and not $R_t$?**  
Because the reward is usually defined as what you get after taking $A_t$ in $S_t$ and transitioning.
So the tuple at time $t$ is: $(S_t, A_t) \rightarrow (R_{t+1}, S_{t+1})$.

### The timeline of one step

At time $t$:
- you are in a state $S_t$,
- you choose an action $A_t$,
- then the environment gives you:
  - reward $R_{t+1}$
  - next state $S_{t+1}$

So the basic causal arrow is:

$$
(S_t, A_t) \rightarrow (R_{t+1}, S_{t+1})
$$

**In code:**
you call `env.step(action)` and receive `(next_state, reward, done, ...)`.


### Discount and horizon
- $\gamma \in [0,1]$: discount factor
- Episodic tasks end at terminal time $T$ (finite horizon)
- Continuing tasks have no terminal time (infinite horizon), usually with $\gamma < 1$

**What discounting does:**
- It encodes preference for sooner rewards
- It keeps returns finite in continuing tasks (when rewards are bounded)

### MDP

An MDP is:

$$
\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
$$

- $\mathcal{S}$: state space
- $\mathcal{A}$: action space
- $P(s^{\prime} \mid s, a)$: transition dynamics (how the world moves)
- $R$: reward rule (how the environment scores transitions)
- $\gamma$: discount factor

A (stationary) policy is a conditional distribution:

$$
\pi(a \mid s) = P(A_t=a \mid S_t=s)
$$

**Markov property:**  
The state should contain enough information that the future depends on the past only through the present

$$
P(S_{t+1}\mid S_t) = P(S_{t+1}\mid S_1,\dots,S_t)
$$

If the state is not Markov, learning can still work, but the theory and algorithms are designed around this assumption.

---

## 2) Trajectories and Return

### 2.1 What is a trajectory?
A trajectory (rollout/episode) is:

$$
\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)
$$

This is exactly what one run of the policy produces.

So, a trajectory is produced by repeatedly:
1) sampling actions from the policy
2) sampling next states from the environment dynamics

**Why define $\tau$ at all?**\
Because our objective is to know how good the policy is when we run it.\
Running the policy produces a trajectory. So the natural thing to do, to understand how good a policy is,
is to average over its trajectories.

**Why include rewards in $\tau$?**\
Because the RL objective is about returns (section below), which depend on rewards. And we will average over its trajectories.

---

### 2.2 The Return (of the King) and reward-to-go
If someone asks how well did the agent do in this episode, you want one number or a score (like when we get a score in school).

That number is the **return** from the start:

$$
G_0 = \sum_{t=0}^{T-1}\gamma^t r_{t+1}
$$

- $\gamma$ makes later rewards count less (and keeps infinite-horizon sums finite).
- In episodic tasks, the sum is finite anyway.

This will give us the return from the start to the end of the episode.\
Our (and RL) objective will be to maximize expected return.

What if we want the return from time $t$?\
The return from time $t$ is the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}
$$

- Start at time $t$
- Take the next reward $R_{t+1}$ with weight $1$ $\left( k=0 \right)$
- Take the following reward $R_{t+2}$ with weight $\gamma$
- Then $\gamma^2 R_{t+3}$ and so on

$G_t$ is also called reward-to-go, because it is the score for how good things turned out from time $t$ onward.

Why introduce $G_t$ at all (instead of always using $G_0$)?

Because the action $a_t$ cannot influence rewards that already happened before it was chosen ($r_1,\dots,r_t$).\
If we weight the update at time $t$ using the full episode return $G_0$, we inject irrelevant noise (part of the weight comes 
from rewards that $a_t$ couldn't have caused). Using $G_t$ keeps only the future part that $a_t$ could have affected, 
which usually reduces variance and makes learning more stable.

In REINFORCE, $G_t$ is the weight that tells the policy whether action $a_t$ was good or bad.

**Episodic vs continuing:**
- In episodic tasks, the sum ends at terminal time $T$ (so it's finite even if $\gamma=1$).
- In continuing tasks, you typically choose $\gamma<1$ so the infinite sum stays finite.

---

## 3) The RL objective is an expectation

Up to now we defined two things:

- a trajectory $\tau$: what you get when you run the policy once
- a return $G_0$: the score of that trajectory (how many rewards you collected, discounted)

Now we need to turn the sentence "this policy is good" into a mathematical objective we can optimize.

### 3.1 One trajectory is not enough

If you run the same policy twice, you may not get the same score:
- the environment can be stochastic
- the policy itself is stochastic
- even with a deterministic policy, the initial state can be random

So we cannot judge a policy from a single episode.
What we really care about is:

> the average return you get if you keep running the policy again and again.

### 3.2 The expected return (definition of $J(\theta)$)

We represent the policy with a neural network with parameters $\theta$, so we write $\pi_\theta$.

Running $\pi_\theta$ in the environment generates trajectories according to some distribution
(which we will define in the next section). Call that distribution $p_\theta(\tau)$.

Then the standard episodic objective is:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ G_0(\tau) \right]
$$

so the objective is an expectation over trajectories $\tau$ (with a trajectory distribution $p_\theta(\tau)$) 
of the return.

This is the quantity we want to **maximize**.

### 3.3 Why this forces us to use sampling

That expectation is usually impossible to compute exactly:
there are too many possible trajectories, and we do not know the environment dynamics in closed form.

So in practice we estimate it with data:

1) run the policy for $N$ episodes and collect trajectories $\tau^{(1)}, \dots, \tau^{(N)}$
2) compute returns $G_0(\tau^{(i)})$ for each episode
3) average that return across many trajectories (approximate):

$$
J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} G_0^{(i)}
$$

This is the Monte Carlo idea: **use samples (rollouts) to approximate an expectation**.

Now, everything is clear about $J(\theta)$ (I hope). However, we wrote $\tau \sim p_\theta(\tau)$ 
without saying what $p_\theta(\tau)$ actually is.

That is the next step: 
we need an explicit expression for the probability of a trajectory under the policy and the environment.\
Once we have that, we can start asking the next question:

> if we change $\theta$, how does $J(\theta)$ change?

---

## 4) The trajectory distribution $p_\theta(\tau)$

### 4.1 Parameterized policies 

Up to now, we talked about a policy $\pi(a \mid s)$.

That is just a rule that, given a state $s$, tells us a probability for each action $a$.

For policy gradients, we do something specific:

> we represent the policy with a neural network, and we call its weights $\theta$.

So we write the policy as:

$$
\pi_\theta(a \mid s)
$$

This means: **the probabilities of actions are controlled by parameters $\theta$**.

Let's see what a parameterized policy looks like.\
Two common cases:

**Discrete actions (like CartPole):**  
The network outputs some numbers (logits), then we turn them into probabilities with a softmax:

$$
\pi_\theta(a \mid s) = \text{softmax}(f_\theta(s))_a
$$

So if $\theta$ changes, the logits change, and the action probabilities change.

**Continuous actions (not in CartPole, but common later):**  
The network outputs a mean (and maybe a std), defining a Gaussian distribution:

$$
a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)
$$

Again: change $\theta$ → change the distribution of actions.

---

### 4.2 How a policy induces a distribution over trajectories

At this point, we know:
- a trajectory is the sequence we observe when we run the policy
- the policy is now parameterized: $\pi_\theta(a\mid s)$

Now we want a new object:

> a formula for how likely is it to see this exact trajectory?

That formula will be called $p_\theta(\tau)$.

Let's build it step by step.

#### Step 1: probability of the first action and next state

Assume we start from $s_0$.
Two things happen:

1) The agent samples an action from the policy:
   $$
   a_0 \sim \pi_\theta(\cdot\mid s_0)
   $$
   so the probability of choosing a specific action $a_0$ is $\pi_\theta(a_0\mid s_0)$.

2) The environment samples the next state:
   $$
   s_1 \sim P(\cdot\mid s_0, a_0)
   $$
   so the probability of landing in a specific state $s_1$ is $P(s_1\mid s_0, a_0)$.

If we want the probability of seeing both events in that order
(pick $a_0$ and then land in $s_1$), we multiply:

$$
\pi_\theta(a_0\mid s_0)\,P(s_1\mid s_0, a_0)
$$

This multiplication is just the chain rule for probabilities
(one event happens, then the next one happens conditioned on the first).

#### Step 2: second action

Now from $s_1$ the same story repeats:

- choose $a_1$ with probability $\pi_\theta(a_1\mid s_1)$
- transition to $s_2$ with probability $P(s_2\mid s_1, a_1)$

So the probability of seeing this two-step partial trajectory:

$$
(s_0, a_0, s_1, a_1, s_2)
$$

is:

$$
\pi_\theta(a_0\mid s_0)\,P(s_1\mid s_0, a_0)\;
\pi_\theta(a_1\mid s_1)\,P(s_2\mid s_1, a_1)
$$

This is a product across time. Every time step contributes two factors:
one from the policy and one from the environment. So we have found a pattern.

#### Step 3: include the start of the episode

Episodes may start from different initial states.
Let $\rho_0(s_0)$ be the probability of starting in $s_0$.

Then the probability of the two-step partial trajectory becomes:

$$
\rho_0(s_0)\,
\pi_\theta(a_0\mid s_0)\,P(s_1\mid s_0, a_0)\;
\pi_\theta(a_1\mid s_1)\,P(s_2\mid s_1, a_1)
$$

#### Step 4: generalize to the full episode (definition of $p_\theta(\tau)$)

If we keep going until time $T$, we multiply the same kind of factors for each step.
That gives the trajectory distribution:

$$
p_\theta(\tau)
=
\rho_0(s_0)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t, a_t)
$$

- $\rho_0(s_0)$: initial-state distribution (how the episode starts)
- $\pi_\theta(a_t\mid s_t)$: parameterized policy (how the agent chooses actions)
- $P(s_{t+1}\mid s_t,a_t)$: transition matrix (how the environment responds)

This is just a product across time:
- for every time step $t$, you multiply:
  - the probability that the policy chose the action
  - the probability that the environment produced the next state

#### Why changing $\theta$ changes which trajectories you see

Look at the formula above:

- $P(s_{t+1}\mid s_t,a_t)$ is the environment (fixed rules)
- $\rho_0(s_0)$ is how episodes start (also fixed)
- only $\pi_\theta(a_t\mid s_t)$ depends on $\theta$

So if you change $\theta$, you change **every policy factor** in that product.
That changes which action sequences are likely, which changes which state sequences are likely,
which changes which full trajectories are likely (I wrote 'which' too many times).

In short:

> changing $\theta$ changes $\pi_\theta(a_t\mid s_t)$ at every step,  
> so it changes the probability of complete rollouts.

This is the precise meaning of the section's title: the policy induces a distribution over trajectories.

---

### 4.3 Why we care about $p_\theta(\tau)$

At this point we have:

- trajectories $\tau$: what we observe when we run the policy
- returns $G_0(\tau)$: the score of a trajectory
- the objective $J(\theta)$: the average score we want to maximize

Now we want to take the next step:

> if we change the policy parameters $\theta$, how does the objective $J(\theta)$ change?

This is why we introduced $p_\theta(\tau)$.

Recall that:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ G_0(\tau) \right]
$$

So $J(\theta)$ depends on $\theta$ in two ways:

1) **directly**: because the policy inside the trajectory distribution depends on $\theta$
2) **indirectly**: because changing the policy changes which trajectories are likely, and therefore changes the expectation

The key practical takeaway from the formula we derived in Section 4.2 is:

- $\rho_0(s_0)$ (how episodes start) is fixed
- $P(s_{t+1}\mid s_t,a_t)$ (environment dynamics) is fixed
- **only** $\pi_\theta(a_t\mid s_t)$ depends on $\theta$

So if we want to know how $J(\theta)$ changes, we only need to understand how the policy terms change.
To make 'change the policy to get more reward' precise, we compute the gradient $\nabla_\theta J(\theta)$.
Think of the gradient as a compass: it points to how we should nudge $\theta$ to make good trajectories more likely.

In the next file, `theory.md`, we will derive a practical formula for $\nabla_\theta J(\theta)$ that uses only:
- samples (rollouts)
- returns/advantages as weights
- and $\log \pi_\theta(a_t\mid s_t)$

without ever differentiating through $P(\cdot)$.

To get there, we first rewrite the trajectory probability in a form that is easy to differentiate: the log form.

### 4.4 Log form 

In Section 4.2, $p_\theta(\tau)$ is a product over time steps.
Products are inconvenient for two reasons:

1) they are harder to manipulate when we differentiate
2) multiplying many small probabilities is numerically unstable

So we take logs. Logs turn products into sums.

Taking logs of the trajectory probability gives:

$$
\log p_\theta(\tau)
=
\log \rho_0(s_0) + \sum_{t=0}^{T-1}\log \pi_\theta(a_t\mid s_t) + \sum_{t=0}^{T-1}\log P(s_{t+1}\mid s_t,a_t)
$$

**Why logs are so useful:**
- Products over time become a sum, which is easier to manipulate and differentiate
- In code, we compute $\log \pi_\theta(a_t\mid s_t)$ directly (it is stable, and libraries can backprop through it)
- When we differentiate with respect to $\theta$, the only terms that matter are:
  $$
  \sum_{t=0}^{T-1}\log \pi_\theta(a_t\mid s_t)
  $$
  because the other terms don't depend on $\theta$.

---

## 5) Numeric example

This is a tiny 2-step MDP example to make $p_\theta(\tau)$, $G_0$, and $G_t$ feel a bit more than just some abstract theory.

In the next example we will pick one specific rollout $\tau$.  
When we compute $p_\theta(\tau)$ and $G_t$, we will only use the pieces of the MDP that actually happened 
along that rollout.

### 5.1 Setup

**Time / horizon**
- Horizon $T=2$: two decisions, at $t=0$ and $t=1$
- Discount $\gamma=0.9$
- Initial state distribution: $\rho_0(s_0)=1$ (we always start in $s_0$)

**States**
- $s_0$: start state
- $A, B$: middle states (you land in one of them after the first action)
- $\text{terminal}$: episode ends after the second action

So the state flow is:
$$
s_0 \;\rightarrow\; \{A,B\} \;\rightarrow\; \text{terminal}
$$

**Actions**
- At $t=0$ (in $s_0$): $a_0 \in \{\text{L},\text{R}\}$
- At $t=1$ (in $A$ or $B$): $a_1 \in \{\text{U},\text{D}\}$

we pick this rollout:
$$
\tau = (s_0,\ \text{L},\ r_1=1,\ A,\ \text{U},\ r_2=2,\ \text{terminal})
$$

this is the sequence of states, actions and rewards we get in this rollout.  
We now compute all the pieces that we need to obtain $p_\theta(\tau)$.

**Policy probabilities** $\pi_\theta(a\mid s)$

These numbers say: if the agent is in state $s$, how likely is it to pick each action?

- In $s_0$:
  $$
  \pi_\theta(\text{L}\mid s_0)=0.7,\quad \pi_\theta(\text{R}\mid s_0)=0.3
  $$
- In $A$:
  $$
  \pi_\theta(\text{U}\mid A)=0.4,\quad \pi_\theta(\text{D}\mid A)=0.6
  $$
- In $B$:
  $$
  \pi_\theta(\text{U}\mid B)=0.9,\quad \pi_\theta(\text{D}\mid B)=0.1
  $$

We won't need $\pi_\theta(\cdot\mid B)$ because our chosen trajectory won't visit $B$.

**Transition probabilities** $P(s^{\prime}\mid s,a)$

These numbers say: after taking action $a$ in state $s$, how likely is each next state $s^{\prime}$?

- From $s_0$:
  - If you take **L**:
    $$
    P(A\mid s_0,\text{L})=0.9,\quad P(B\mid s_0,\text{L})=0.1
    $$
  - If you take **R**:
    $$
    P(A\mid s_0,\text{R})=0.2,\quad P(B\mid s_0,\text{R})=0.8
    $$

  - We won't use **R** in the chosen trajectory, so its transitions are irrelevant here.
- From $A$ or $B$ at $t=1$ you always go to terminal:
  $$
  P(\text{terminal}\mid A,\text{U})=P(\text{terminal}\mid A,\text{D})=1,\quad
  P(\text{terminal}\mid B,\text{U})=P(\text{terminal}\mid B,\text{D})=1
  $$

**Rewards**

Rewards are not multiplied into $p_\theta(\tau)$.  
They are summed (with discount) into the return $G_0$ and reward-to-go $G_t$.

- After the first transition (this is $r_1$, observed after moving from $s_0$ to $A$ or $B$):
  $$
  r_1 =
  \begin{cases}
  1 & \text{if } s_1=A\\
  0 & \text{if } s_1=B
  \end{cases}
  $$
- After the second action (this is $r_2$, observed after acting in $A$ or $B$):
  - If $s_1=A$:
    $$
    r_2 =
    \begin{cases}
    2 & \text{if } a_1=\text{U}\\
    0 & \text{if } a_1=\text{D}
    \end{cases}
    $$
  - If $s_1=B$:
    $$
    r_2 =
    \begin{cases}
    1 & \text{if } a_1=\text{U}\\
    -1 & \text{if } a_1=\text{D}
    \end{cases}
    $$
  We won't need the $B$ reward cases here.

---

### 5.2 Pick one trajectory and compute its probability

Pick this specific episode:

- Start in $s_0$
- Take action $\text{L}$
- Environment moves to $A$ (so $r_1=1$)
- In $A$, take action $\text{U}$ (so $r_2=2$)
- End in terminal

Written as a trajectory:
$$
\tau = (s_0,\ \text{L},\ r_1=1,\ A,\ \text{U},\ r_2=2,\ \text{terminal})
$$

Its probability under the policy and environment is:
$$
p_\theta(\tau)
=
\rho_0(s_0)\,
\pi_\theta(\text{L}\mid s_0)\,
P(A\mid s_0,\text{L})\,
\pi_\theta(\text{U}\mid A)\,
P(\text{terminal}\mid A,\text{U})
$$

Plug in the numbers:
$$
p_\theta(\tau)=1\cdot 0.7 \cdot 0.9 \cdot 0.4 \cdot 1 = 0.252
$$

With these probabilities, this exact 2-step rollout happens about 25.2% of the time.

---

### 5.3 Return $G_0$ vs reward-to-go $G_1$

Return from the start (whole episode score):
$$
G_0 = r_1 + \gamma r_2 = 1 + 0.9\cdot 2 = 2.8
$$

Reward-to-go from the second decision (what happens after choosing $a_1$):
$$
G_1 = r_2 = 2
$$

**Why we bother with $G_t$:**  
When updating the policy for the second action $a_1$, the earlier reward $r_1$ happened before choosing $a_1$.  
So $a_1$ could not have caused $r_1$. If we used $G_0$ as the weight at $t=1$, we would inject extra noise.  
Using $G_1$ keeps only the future part that $a_1$ could have influenced, which usually reduces variance.

---

## 6) Entropy, cross-entropy, and KL divergence

These three quantities appear repeatedly once you go beyond REINFORCE.
The A2C entropy bonus and the PPO trust-region constraint are both built from the same
basic object, $-\log p(x)$, just assembled differently.

This section introduces all three from scratch and then shows how they connect,
so that when you meet them in `theory.md` the symbols already feel familiar.

---

### 6.1 Surprise and entropy

Pick any probability distribution $p$ over a finite set (for us, the set of actions $\mathcal{A}$).

Define the **surprise** (or self-information) of outcome $a$ under $p$ as:

$$
-\log p(a)
$$

The minus sign is there so that surprise is positive and large when $p(a)$ is small.
If $p(a) = 1$, you knew it was coming: surprise = 0.
If $p(a) = 0.01$, it almost never happens: surprise $= -\log 0.01 \approx 4.6$.

You can think 'surprise' as literally how much are you surprised to see the outcome $a$ under 
probability distribution $p$. If the event is rare, you are a lot surprised to see it (surprise has a big value).
If the event will surely happen, then you will not be surprised to se it (surprise hits the lowest value).

**Entropy** is simply the expected surprise under $p$ itself:

$$
H[p] = -\sum_{a \in \mathcal{A}} p(a) \log p(a)
$$

It answers: *on average, how surprised are you by your own distribution?*

Two extreme cases to build intuition:

- **Uniform distribution** over $|\mathcal{A}|$ actions: every action has probability $1/|\mathcal{A}|$,
  so every outcome is equally surprising.
  Entropy is maximised at $\log |\mathcal{A}|$.
- **Deterministic distribution** (all mass on one action): you are never surprised.
  Entropy = 0.

High entropy means the distribution is spread out and uncertain.
Low entropy means it is peaked and confident.

**Why this matters for RL:**
In A2C (and many other algorithms), we add an entropy bonus to the policy gradient objective:
we literally add $H[\pi_\theta(\cdot\mid s)]$ to the loss.
The reason is that a policy with low entropy has committed strongly to a few actions,
which can cause it to stop exploring.
Maximising entropy keeps the policy spread out, encouraging the agent to try actions
it has not yet evaluated.

---

### 6.2 Cross-entropy

Now suppose you have *two* distributions: a true distribution $p$ and an approximate one $q$.
If the world actually follows $p$, but you are using $q$'s log-probabilities to measure surprise,
the expected surprise you experience is called **cross-entropy**:

$$
H[p,\,q] = -\sum_{a \in \mathcal{A}} p(a) \log q(a)
$$

Notice the asymmetry: you average over $p$ (because that is the true distribution),
but you take logs of $q$ (because that is the model you are using).

**Where you have already seen this:**
Cross-entropy is exactly the loss used for supervised classification.
When you train a neural network to classify images, $p$ is a one-hot vector
(probability 1 on the correct class, 0 on everything else)
and $q$ is your model's softmax output.
The cross-entropy loss penalises $q$ for assigning low probability to the correct label.
So supervised classification training is, at its core, minimising
the expected $-\log q(\text{correct class})$ over the training distribution.

---

### 6.3 KL divergence

**KL divergence** is the *extra* surprise you pay by using $q$ instead of $p$:

$$
D_{\mathrm{KL}}(p \| q)
= \sum_{a \in \mathcal{A}} p(a) \log \frac{p(a)}{q(a)}
$$

The connection to the previous two quantities is just algebra:

$$
D_{\mathrm{KL}}(p \| q)
= \underbrace{-\sum_a p(a) \log q(a)}_{H[p,\,q]}
\;-\;
\underbrace{\Bigl(-\sum_a p(a) \log p(a)\Bigr)}_{H[p]}
= H[p,\,q] - H[p]
$$

In words: cross-entropy = KL divergence + entropy of the true distribution.

**A useful consequence:**
If $p$ is fixed (as it is in supervised learning, where the labels do not change),
then $H[p]$ is a constant that does not depend on your model $q$.
Minimising cross-entropy and minimising KL divergence are therefore exactly the same optimisation problem 
(they differ only by a constant).

Two properties worth remembering:

- $D_{\mathrm{KL}}(p \| q) \geq 0$ always, with equality only when $p = q$ everywhere.
- It is not symmetric: $D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$ in general.
  The order matters, and PPO uses a specific ordering for a specific reason (covered in `theory.md` of block 05).

---

### 6.4 Numeric example

A 3-action policy to make all three quantities concrete (so you are less surprised when you see them).

Suppose $\mathcal{A} = \{a_1, a_2, a_3\}$ and the agent's current policy is:

$$
\pi_{\text{old}} = (0.7,\; 0.2,\; 0.1)
$$

After one gradient update it becomes:

$$
\pi_{\text{new}} = (0.4,\; 0.4,\; 0.2)
$$

**Entropy of $\pi_{\text{old}}$:**

$$
H[\pi_{\text{old}}]
= -(0.7\ln 0.7 + 0.2\ln 0.2 + 0.1\ln 0.1)
\approx -(0.7 \cdot (-0.357) + 0.2 \cdot (-1.609) + 0.1 \cdot (-2.303))
\approx 0.802
$$

**Entropy of $\pi_{\text{new}}$:**

$$
H[\pi_{\text{new}}]
= -(0.4\ln 0.4 + 0.4\ln 0.4 + 0.2\ln 0.2)
\approx -(0.4 \cdot (-0.916) + 0.4 \cdot (-0.916) + 0.2 \cdot (-1.609))
\approx 1.055
$$

The entropy increased: $\pi_{\text{new}}$ is more spread out, which is what the entropy bonus is trying to encourage.

**KL divergence from old to new** (how much the policy changed):

$$
D_{\mathrm{KL}}(\pi_{\text{old}} \| \pi_{\text{new}})
= 0.7\ln\frac{0.7}{0.4} + 0.2\ln\frac{0.2}{0.4} + 0.1\ln\frac{0.1}{0.2}
\approx 0.7(0.560) + 0.2(-0.693) + 0.1(-0.693)
\approx 0.392 - 0.139 - 0.069
\approx 0.184
$$

**Cross-entropy as a sanity check** ($H[p,q] = D_{\mathrm{KL}} + H[p]$):

$$
H[\pi_{\text{old}},\, \pi_{\text{new}}]
= D_{\mathrm{KL}}(\pi_{\text{old}} \| \pi_{\text{new}}) + H[\pi_{\text{old}}]
\approx 0.184 + 0.802
= 0.986
$$

You can verify this directly:
$-\sum_a \pi_{\text{old}}(a)\ln \pi_{\text{new}}(a)
= -(0.7\ln 0.4 + 0.2\ln 0.4 + 0.1\ln 0.2)
\approx -(0.7(-0.916)+0.2(-0.916)+0.1(-1.609))
\approx 0.986$

---

### 6.5 How they show up in practice

These three quantities are not independent ideas.
They are the same $-\log p(x)$ object, combined in slightly different ways for slightly different purposes.

**A2C** adds $H[\pi_\theta(\cdot\mid s)]$ to the objective with a small positive coefficient.
This acts as a soft regulariser: it penalises policies that collapse onto a single action
before the agent has had a chance to find out whether that action is truly good.

**PPO** replaces that soft signal with a hard constraint:
$D_{\mathrm{KL}}(\pi_{\text{old}} \| \pi_{\theta}) < \delta$.
Instead of asking the policy to be spread out in absolute terms,
PPO asks it not to change too much in a single update relative to where it started.
This is called a trust-region constraint and is the main reason PPO tends to be more stable than vanilla policy gradients.
So, PPO uses KL divergence directly to control how much the policy is allowed
to change in a single update. 

The progression is: A2C uses entropy to encourage exploration,
PPO uses KL to control stability.
Both are expressions of the same underlying caution about committing too fast,
just applied at different stages of the training loop.

However, PPO belongs to block 05 (and we are in block 04 now), The details belong there, for now it is enough to know
that KL is the natural tool for measuring "how different is the new policy from the old one",
which is exactly the question PPO is designed to answer.

