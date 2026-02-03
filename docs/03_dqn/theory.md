# DQN Theory (DQN + Double DQN + Dueling DQN)

This note goes with:
- `src/rl_algorithms_guide/dqn/dqn.py`
- `examples/03_dqn/train_dqn_cartpole.py`
- `examples/03_dqn/train_dqn_lunarlander.py`

Assumed background (from earlier blocks):
- MDP objects ($\mathcal{S},\mathcal{A},P,R,\gamma$), value functions $v_\pi, q_\pi$, Bellman equations
- TD learning + TD error, Q-learning target, and the idea of on-policy vs off-policy

---

## 0) The big picture (how DQN fits after Block 02)

In Block 02 (tabular TD control), Q-learning updates a table:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\Big(
R_{t+1} + \gamma \max_{a^{\prime}} Q(S_{t+1},a^{\prime}) - Q(S_t,A_t)
\Big)
$$

That is great when:
- the state space is small and discrete
- you can store values in memory (a lookup table)

But in most interesting environments:
- states are high-dimensional (images, sensors, long vectors)
- there are way too many possible states to index a table

So the core change in DQN is:

$$
Q(s,a) \approx Q_\theta(s,a)
$$

where $Q_\theta$ is a neural network with parameters $\theta$.

**Important:** DQN is not a brand new algorithm. It is still Q-learning at heart (bootstrapping + max over actions).
The "deep" part is just how we represent $Q$.

---

## 1) From table updates to learning a function

A tabular Q-table stores one independent number per $(s,a)$.
A neural network forces sharing:

- if two states look similar, they share network parameters
- learning in one region generalises to other regions

This is the whole point of function approximation:
**generalise from seen states to unseen states.**

(generalise = I hope the model does the right thing on states I didn't show it)

### What the Q-network outputs

In this repo, the Q-network takes a state $s$ and outputs **a vector of Q-values**:

$$
Q_\theta(s,\cdot) \in \mathbb{R}^{|\mathcal{A}|}
$$

So for a discrete action space with $|\mathcal{A}|=4$, the network outputs 4 numbers:

- index 0 = estimated $Q_\theta(s, a=0)$  
- index 1 = estimated $Q_\theta(s, a=1)$  
- etc.

When you actually took action $a$, you select the corresponding entry (a "gather" operation), 
that means you select the Q-value corresponding to the action that was actually taken in that transition.

This design is why DQN is naturally tied to **discrete** action spaces.

### The supervised-learning view

With a Q-table, you update a single cell.\
With a network, you do something that looks like supervised learning:

- input: the current state $s$
- output: a predicted vector of Q-values for all actions, $Q_\theta(s,\cdot) \in \mathbb{R}^{|\mathcal{A}|}$
- training target (label): a single scalar TD target for the action you actually took. 
The TD target is computed from rewards + a bootstrap term

Concretely, for a transition $(s,a,r,s^{\prime},done)$ we build:

$$
y = r + (1-done)\,\gamma \max_{a^{\prime}} Q_{\theta^-}(s^{\prime},a^{\prime})
$$

Then we take the network's prediction for the chosen action $a$:

$$
\hat{y} = Q_\theta(s,a)
$$

and train the network to make $\hat{y}$ close to $y$ (using Huber loss on the TD error $y - \hat{y}$).

So the learning step is:
> treat the TD target $y$ like a supervised label for the taken action, and fit $Q_\theta(s,a)$ to it on minibatches from replay (replayed transitions).

In section 4 we will talk about this target again.

---

## 2) Why the $\max$ is there (control = greedy improvement)

In control, we want optimal action-values $q_{\ast}(s,a)$ and a greedy policy:

$$
\pi_{\ast}(s) = \arg\max_a q_{\ast}(s,a)
$$

The Bellman optimality idea says: the value of a next state should reflect **the best action you could take there**.

That's why the Q-learning/DQN target uses:

$$
\max_{a^{\prime}} Q(\,s^{\prime},a^{\prime}\,)
$$

This is the greedy improvement part of Generalised Policy Iteration:
evaluate approximately, then improve greedily, repeat.

So DQN is still the same greedy control idea from tabular Q-learning.

---

## 3) Why naïve Q-learning with a neural net can explode

If you literally take Q-learning and swap the table with a neural network,
you often get unstable learning.

There are three reasons that interact badly:

1) **Bootstrapping**  
   You update toward a target that uses your own current estimates.

2) **Off-policy learning**  
   Your data comes from a behaviour policy that includes exploration,
   while your target is computed using a greedy backup.

3) **Function approximation**  
   A single gradient step changes many Q-values at once (because parameters are shared).

If those three are present together, training can diverge.

DQN's two classic stabilisers are exactly aimed at this:

- **replay buffer**: reduces correlation and non-stationarity
- **target network**: reduces the moving-target effect

There are other stabilisers people often add: gradient clipping, reward clipping, etc.\
In this repo we keep things minimal.

---

## 4) The DQN learning target (what the network is trying to match)

A transition looks like:

$$
(s, a, r, s^{\prime}, done)
$$

DQN builds a target:

$$
y = r + (1-done)\,\gamma \max_{a^{\prime}} Q_{\theta^-}(s^{\prime}, a^{\prime})
$$

Key details:
- the $\max$ is over actions in the **next state**
- we use a **target network** $Q_{\theta^-}$ (a delayed copy of the online net)
- if `done=1`, we do not bootstrap (the multiplier becomes $0$)

Then we train the online network so that:

$$
Q_\theta(s,a) \approx y
$$

In code, this pick the Q for the taken action step is:

- compute all $Q_\theta(s,\cdot)$
- select the entry for action $a$

**Terminology note:** online network here just means the Q-network we are training ($Q_\theta$).\
The target network ($Q_{\theta^-}$) is a separate frozen copy used only to compute targets.

### Terminal handling

If `done=1`, the episode ended, so the correct target is:

$$
y = r
$$

This single detail matters a lot in practice.
Bootstrapping past terminal states can seriously mess up learning.

### The target network is **not trained**

This is easy to get wrong conceptually, so let's be explicit:

- $Q_\theta$ (online) is trained by gradient descent
- $Q_{\theta^-}$ (target) is **not** trained (no backprop through it)

Instead, it is a **frozen copy** of the online network:
- periodically copy parameters: $\theta^- \leftarrow \theta$

In this repo, we copy $\theta \to \theta^-$ every $N$ environment steps.

---

## 5) TD error in DQN and why Huber loss is used

From Block 02, the TD error is target minus estimate.

In DQN:
- estimate is the Q-value for the taken action: $Q_\theta(s,a)$
- target is the bootstrapped value: $y$

So the TD error is:

$$
\delta = y - Q_\theta(s,a)
$$

Instead of doing a tabular update, we train the network by minimizing a regression loss on $\delta$.

A common stable choice is **Huber loss**:
- behaves like MSE when $|\delta|$ is small (nice smooth gradients)
- behaves like absolute error when $|\delta|$ is large (less sensitive to outliers)

So the learning step is basically:
> make the TD error $\delta$ small on minibatches sampled from replay.

---

## 6) Replay buffer (the correlation problem)

If you train on consecutive transitions, your data is correlated:

- in CartPole, the pole angle changes smoothly, so $s_t$ and $s_{t+1}$ are very similar
- consecutive samples look the same
- your minibatch gradients are dominated by a tiny region of the state space

That causes two practical issues:
1) **inefficient learning**: you keep training on near-duplicates
2) **instability**: your network chases whatever it saw most recently

A replay buffer fixes this by:
- storing many transitions collected over time
- sampling minibatches uniformly at random

So a minibatch contains a mixture of:
- different episodes
- different stages of training
- different parts of the state space

This makes the training signal much closer to the i.i.d. (Independent and Identically Distributed) assumption 
that SGD (Stochastic Gradient Descent) likes.

So at time step $t$:

1. You are in state $s$
2. You choose **one** action $a$ (either greedy or random because of $\epsilon$)
3. The environment gives you $r, s^{\prime}, done$
4. You store that one tuple:

$$
(s, a, r, s^{\prime}, done)
$$

Remember, only the transition that actually happened goes into replay, 
since it is just a memory of experience the agent really collected.


### Tiny analogy
Think of it like training a classifier:
- if you feed 10,000 images of cats in a row, then 10,000 of dogs, training is annoying and unstable
- if you shuffle, training is smoother

Replay is basically shuffle your RL experience.

---

## 7) Target network (the moving target problem)

Suppose you compute the target using the **same** network you are updating:

$$
y = r + \gamma \max_{a^{\prime}} Q_\theta(s^{\prime},a^{\prime})
$$

Now imagine you do one gradient step on $\theta$.
That changes:
- your prediction $Q_\theta(s,a)$ (the thing you're trying to fit)
- and also the target $y$ (because it uses $Q_\theta$)

So the label (target) moves every time you learn.

This feedback loop can create oscillations:
- you fit a target
- the target changes
- you fit again
- the target changes again
- … and you never settle

The target network breaks that loop.\
This is why DQN uses two networks.\
We train the **online network** $Q_\theta$, but we compute the TD targets using a 
**target network** $Q_{\theta^-}$ that is kept fixed for a while.
Every $N$ environment steps (in this repo), we copy $\theta \to \theta^-$ so the targets refresh, 
but they don't move every gradient step.


- online network: $Q_\theta$ $\rightarrow$ updated every gradient step
- target network: $Q_{\theta^-}$ $\rightarrow$ frozen for a while

So during training, for a while:
- the target is computed from a fixed network
- SGD has something stable to chase

---

## 8) Exploration: epsilon-greedy, but with "why the schedule matters"

DQN usually uses $\epsilon$-greedy:

- with probability $1-\epsilon$: take $\arg\max_a Q_\theta(s,a)$
- with probability $\epsilon$: take a random action

### Why $\epsilon$ decays

Early on, the network is basically random.
If you act greedily too early, you lock into bad behaviour and never discover better trajectories.

So you start with high exploration (large $\epsilon$), then decay it.

### A concrete schedule example

If you decay from $\epsilon=1.0$ to $\epsilon=0.05$ over 50k steps:

- step 0: you act almost randomly
- step 25k: you are half-way between random and greedy
- step 50k: you mostly exploit, but still explore sometimes

In the scripts, we plot the epsilon curve so you can sanity-check it.
If the plot is flat (or drops too fast), it often explains bad learning.

---

## 9) Off-policy learning in DQN

From Block 02, the key idea is:
- **behaviour policy** $\mu$: the policy you actually use to collect data
- **target policy** $\pi$: the policy whose value function you are trying to learn

A method is off-policy when $\mu \neq \pi$.

---

### What is $\mu$ in DQN?

In DQN we usually act with $\epsilon$-greedy to explore:

- with probability $1-\epsilon$: take the greedy action
- with probability $\epsilon$: take a random action

So the behaviour policy is:

$$
\mu(\cdot \mid s) = \epsilon\text{-greedy w.r.t. } Q_\theta
$$

That means we sometimes take actions that the greedy policy would not take.

---

### What is $\pi$ (the target policy) in DQN?

The DQN target is built using a greedy backup:

$$
\max_{a^{\prime}} Q_{\theta^-}(s^{\prime},a^{\prime})
$$

This max corresponds to assuming that in the next state $s^{\prime}$ we will act greedily.

So the target policy is the greedy policy, with respect to the current Q estimates (the online network):

$$
\pi(s) = \arg\max_a Q_\theta(s,a)
$$

Remember: the greedy policy is defined w.r.t. the online network $Q_\theta$; 
the target network $Q_{\theta^-}$ is only used to compute more stable TD targets.

So DQN is off-policy, exactly like tabular Q-learning. 
It learns the greedy policy while collecting experience with a more exploratory behaviour policy.

Furthermore, replay makes this even more obviously off-policy because the buffer contains experience 
collected under many past $\epsilon$ values, not just the current one.

Note: you only store (in the replay) what you actually experienced, because you can't observe what would have happened for the 
action you didn't take.\
Replay stores transitions generated by the behaviour policy (the actions you actually executed), 
whether they were greedy or exploratory.

---

### Tiny example: behaviour explores, target learns greedily

Suppose in a state $s$ your current network predicts:

- $Q_\theta(s,a_0) = 10$
- $Q_\theta(s,a_1) = 9$

So the greedy action is $a_0$.

Now, because of exploration, with $\epsilon=0.2$ you might actually take $a_1$ (the non-greedy one).\
That transition still goes into the replay buffer.
So:

- If you picked $a_1$ because exploration triggered $\rightarrow$ you store the $a_1$ transition.
- If you picked the greedy action $a_0$ $\rightarrow$ you store the $a_0$ transition.

You do not store both possible transitions from the same state.

Later, when you train on a minibatch, the target uses:

$$
y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')
$$

Notice what happened:

- the data contains actions taken by $\epsilon$-greedy behaviour ($\mu$)
- the target uses a greedy assumption at the next state (the $\max$), which corresponds to the target policy ($\pi$)

So you are learning "what would happen if I behave greedily",
while still collecting data that includes exploration.

This is exactly the same off-policy idea as tabular Q-learning.
You learn a greedy policy while behaving more randomly to explore.

---

## 10) Putting it together: the full DQN loop

Here is DQN as an operating procedure:

1) Initialise:
   - online network $Q_\theta$
   - target network $Q_{\theta^-} \leftarrow Q_\theta$
   - replay buffer $D$

2) For each environment step:
   - choose $a$ using $\epsilon$-greedy w.r.t. $Q_\theta$
   - execute action, observe $(r, s^{\prime}, done)$
   - store $(s,a,r,s^{\prime},done)$ into replay $D$

3) After warmup (once $D$ has enough variety):
   - sample a minibatch from replay
   - compute targets using the target network
   - compute TD error $\delta = y - Q_\theta(s,a)$
   - update $\theta$ to reduce the Huber loss on $\delta$

4) Every $N$ environment steps:
   - copy $\theta \to \theta^-$

That is exactly what you will see in:
- `examples/03_dqn/train_dqn_cartpole.py`
- `examples/03_dqn/train_dqn_lunarlander.py`

DQN training loop: die $\rightarrow$ respawn $\rightarrow$ update $\rightarrow$ repeat. Eventually even the boss (LunarLander) goes down.

### A tiny example of this loop

Suppose we set:
- `learning_starts = 1000`
- `train_frequency = 1` (train every env step after warmup)
- `target_update_interval = 500` (in env steps)

Then training looks like this:

- Steps 1-1000: only collect transitions into replay (no gradient updates yet).
- Step 1001: collect 1 transition, then sample a minibatch and do 1 gradient update (on the online network).
- Steps 1002-1499: keep collecting + updating every step.
- Step 1500: collect + update, then copy the online weights into the target network.

So after warmup, DQN is online interaction + replay training interleaved:
you keep collecting fresh experience while learning from a shuffled mix of old and new transitions.

---

## 11) Double DQN (why DQN tends to overestimate)

### The overestimation problem

Standard DQN uses:

$$
\max_{a^{\prime}} Q_{\theta^-}(s^{\prime},a^{\prime})
$$

If your Q-values are noisy estimates, the max has a bias:
> taking the maximum of noisy numbers tends to pick a number that is "too high".

This happens even if the noise is unbiased.

If you take the max of noisy estimates, you're selecting the most confident liar.

### A tiny numeric example

Assume the true next-state values are identical:
- all actions are equally good (true value is 0)

But your target network estimates have noise:

- action 0: $-0.2$
- action 1: $+0.1$
- action 2: $-0.05$

The max is $+0.1$, so your target becomes slightly optimistic.

If you repeat this many times, you keep picking the "lucky" action
(the one with positive noise), so optimism accumulates.

That can lead to:
- inflated Q-values
- unstable learning
- worse policies (because the agent becomes confident in wrong actions)

### Double DQN fix (decouple selection and evaluation)

Double DQN changes only one thing: how we compute the next value.

1) Select the best action using the **online** network:

$$
a^{\ast} = \arg\max_{a^{\prime}} Q_\theta(s^{\prime},a^{\prime})
$$

2) Evaluate that chosen action using the **target** network:

$$
Q_{\theta^-}(s^{\prime}, a^{\ast})
$$

So the Double DQN target becomes:

$$
y_{\text{DDQN}} = r + (1-done)\,\gamma\,Q_{\theta^-}(s^{\prime}, \arg\max_{a^{\prime}} Q_\theta(s^{\prime},a^{\prime}))
$$

### Another numeric example

Suppose for the next state $s^{\prime}$ we have:

Online network:
- $Q_\theta(s^{\prime},\cdot) = [1.2,\ 1.1,\ 0.0]$
so it selects $a^{\ast} = 0$.

Target network (slightly different estimates):
- $Q_{\theta^-}(s^{\prime},\cdot) = [0.4,\ 0.9,\ 0.2]$

Now compare:

- DQN bootstrap term uses $\max$ on target net: $\max = 0.9$
- Double DQN bootstrap term uses target at chosen action: $Q_{\theta^-}(s^{\prime},0)=0.4$

So DQN says "next value is 0.9", Double DQN says "next value is 0.4".
If action 0 was selected because of online noise, Double DQN reduces the optimism.

---

## 12) Dueling DQN (yes, another variant of DQN)

Dueling DQN is an architectural idea: it changes the shape of the Q-network, not the learning rule.

Instead of directly outputting $Q(s,a)$, the network outputs two things:

- a **state value** $V(s)$: how good is this state, regardless of what action I take
- an **advantage** $A(s,a)$: how much better/worse is action $a$ compared to a typical action in this state

Mini intuition: think of $V(s)$ as the *baseline quality* of being in the state, 
and $A(s,a)$ as the *action-specific adjustment*.

Note: We will revisit advantage more formally in Block 04 for policy gradients. 
Here it's just a useful decomposition for Q-networks

---

### What does advantage mean here?

In this dueling architecture, advantage is not a new RL algorithm concept. 
It's just a useful way to represent Q-values.

The meaning is the same as the formal definition you will see again in Block 04:

$$
A(s,a) = Q(s,a) - V(s)
$$

Interpretation:
- $V(s)$ = how good is this state on average?
- $A(s,a)$ = is action $a$ better or worse than the average action in this state?

So you can think of dueling as:
> learn the overall value of the state ($V$), then learn small per-action adjustments ($A$).

**Important:** In Dueling DQN, $A(s,a)$ is an internal network head used to compute $Q(s,a)$.\
In policy gradients (Block 04), advantage becomes a training signal for the policy.
Same intuition (better/worse than average), but used in a different way.

---

So, going back to $V$ and $A$, after outputting we combine them:

$$
Q(s,a) = V(s) + \Big(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a^{\prime}}A(s,a^{\prime})\Big)
$$

### Why subtract the mean advantage?

The network is trying to represent the same Q-values using two pieces:

- a state-wide baseline $V(s)$
- action-specific adjustments $A(s,a)$

But without a constraint, this decomposition is not unique.

Without normalization, $V$ and $A$ are not identifiable.

#### The shift trick

Suppose you have some pair $(V, A)$ that produces correct Q-values:

$$
Q(s,a) = V(s) + A(s,a)
$$

Now pick any constant $c$ and define:

- $V^{\prime}(s) = V(s) + c$
- $A^{\prime}(s,a) = A(s,a) - c$ for every action $a$

Then:

$$
V^{\prime}(s) + A^{\prime}(s,a) = \big(V(s)+c\big) + \big(A(s,a)-c\big) = V(s) + A(s,a)
$$

So **the Q-values are identical**, even though $V$ and $A$ changed.

That means the network could hide value in $V$ or in $A$ arbitrarily,
which makes learning unstable (nothing pins the roles down).

#### The fix: force advantages to be centered

A simple and standard constraint is: advantages should sum to zero across actions.

So we define:

$$
Q(s,a) = V(s) + \Big(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a^{\prime}} A(s,a^{\prime})\Big)
$$

This subtracts the mean advantage, which guarantees:

$$
\frac{1}{|\mathcal{A}|}\sum_a \Big(A(s,a) - \text{mean}_{a} A(s,a)\Big) = 0
$$

Now $V(s)$ becomes the baseline level for the state, and $A(s,a)$ becomes deviations from that baseline.

#### Why this gives a clean interpretation

Average $Q(s,a)$ over actions:

$$
\frac{1}{|\mathcal{A}|}\sum_a Q(s,a)
= \frac{1}{|\mathcal{A}|}\sum_a \left[V(s) + (A(s,a) - \text{mean}_{a} A)\right]
= V(s)
$$

So you can literally read it as:

- $V(s)$ = average value of being in state $s$
- $A(s,a)$ = how much action $a$ is above/below that average

Other normalizations exist, but mean-subtraction is the common default.

### Intuition: why dueling helps

In many states, the choice of action barely matters.
Example: you are far from any meaningful decision.

A normal Q-network must learn Q-values for every action anyway.

A dueling network can learn:
- "this state is good/bad" via $V(s)$
- only small action differences via $A(s,a)$

This can speed learning and improve generalization in some environments.

### Mini example

Suppose $|\mathcal{A}|=3$ and the network outputs:
- $V(s)=5$
- $A(s,\cdot)=[2,0,-1]$

Mean advantage is $\bar{A} = (2+0-1)/3 = 1/3$.

So:
- $Q(s,a_0)=5 + (2-1/3)=6.67$
- $Q(s,a_1)=5 + (0-1/3)=4.67$
- $Q(s,a_2)=5 + (-1-1/3)=3.67$

Here $V(s)$ sets the overall state quality, and $A(s,a)$ ranks actions above/below that level.

---

## 13) Double + Dueling

These ideas target different parts of the system:

- Double DQN: changes **how targets are computed** (selection vs evaluation)
- Dueling DQN: changes **how $Q_\theta$ is represented** (value head + advantage head)

So you can combine them:
- use a dueling architecture for the online and target networks
- compute targets with the Double-DQN rule

That's what the repo calls `double_dueling`.

---

### Mini example (showing they don’t interfere)

Assume the next state is $s^{\prime}$ and there are 3 actions.

**Online network (dueling) produces Q-values** for $s'$:
- internally it computes $V_\theta(s^{\prime})$ and $A_\theta(s^{\prime},\cdot)$
- but after combining them you just get a normal Q-vector:

$$
Q_\theta(s^{\prime},\cdot) = [1.2,\ 1.1,\ 0.0]
$$

So the selected action is:

$$
a^{\ast} = \arg\max_a Q_\theta(s^{\prime},a) = 0
$$

**Target network (also dueling) produces Q-values** for the same $s^{\prime}$:

$$
Q_{\theta^-}(s^{\prime},\cdot) = [0.4,\ 0.9,\ 0.2]
$$

Now compare the bootstrap term:

- DQN would use $\max_a Q_{\theta^-}(s^{\prime},a) = 0.9$
- Double DQN uses $Q_{\theta^-}(s^{\prime},a^{\ast}) = Q_{\theta^-}(s^{\prime},0) = 0.4$

So the combined method uses the Double-DQN target:

$$
y = r + (1-done)\,\gamma\,Q_{\theta^-}(s^{\prime}, \arg\max_a Q_\theta(s^{\prime},a))
$$

Note: the dueling details are inside how each network produces its Q-values,
while Double DQN only changes which Q-value is used in the target.

---

## 14) How to read the plots from the scripts

Each run saves:

1) **Return vs episode**
- the main learning curve
- should generally trend upward (but LunarLander is noisy)

2) **TD loss vs update**
- this is not a perfect "health metric", but it's useful
- if it explodes or becomes NaN, something is wrong
- if it slowly decreases or stays bounded, that's usually fine

3) **Epsilon vs env step**
- sanity check: is exploration decaying as intended?

A very common pattern:
- returns improve first
- loss can stay noisy (that's normal)
- epsilon decays smoothly

---

## 15) Practical notes (things that bite you in DQN)

- **Warmup matters**: don't train before the replay buffer has enough variety.
- **Terminal handling**: if `done=True`, don't bootstrap.
- **Exploration schedule**: if epsilon decays too fast, learning can stall.
- **Loss choice**: Huber loss is usually more stable than plain MSE.
- **Target update interval**: too frequent -> unstable; too rare -> slow learning.
- **One env, many seeds**: single curves can be misleading; averaging is better.

---

## 16) Bridge to the next blocks

DQN is value-based and works best for discrete actions.

Next you will see:
- policy gradients / actor-critic (continuous actions)
- PPO (practical stable actor-critic)
- SAC/TD3 (off-policy continuous control)
