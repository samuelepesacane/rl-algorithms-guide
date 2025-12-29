# Multi-Armed Bandits ($\varepsilon$-Greedy and UCB)

Bandits are the smallest RL problems that still force you to deal with a real challenge:
**you don’t know what’s best until you try it**.

They’re also a nice training ground because you can run lots of experiments quickly and build intuition before moving to full MDPs (Markov Decision Processes), where states and transitions enter the picture.

---

## 1) Problem setup (what a bandit really is)

A **k-armed bandit** is a repeated decision problem:

- You have a fixed set of **actions** (arms)  
  $$a \in \{0, 1, \dots, k-1\}.$$

- There is no real "state" that changes over time (or you can think there’s a single state repeated forever).  
  So nothing like "next state" or "episode" matters here.

- Each arm $a$ has an unknown **reward distribution**.  
  If you choose arm $a$, you observe a random reward:
  $$R \sim \mathcal{D}_a.$$

- The key quantity is the arm’s **true mean reward** (unknown to you):
  $$q_*(a) = \mathbb{E}[R \mid A=a].$$

### What you control vs what you don’t

At each time step $t$:

1. You choose an action $A_t$.
2. The environment draws a reward $R_t$ from that action’s distribution.
3. You update your beliefs (your estimates) and repeat.

Your goal is to maximize reward over time. Common objectives are:

- maximize cumulative reward:  
  $$\sum_{t=1}^{T} R_t$$

- or equivalently maximize the expected reward per step.

### The "regret" view (why exploration matters)

A useful way to think about bandits is **regret**: how much reward you lose because you didn’t always play the best arm.

Let $a^* = \arg\max_a q_*(a)$ be the optimal arm (unknown to you).  
Then expected regret over $T$ steps is:

$$
\text{Regret}(T) =
\sum_{t=1}^{T} \left(q_*(a^*) - q_*(A_t)\right).
$$

If you explore too little, you might commit to the wrong arm and regret becomes large.  
If you explore too much, you keep trying suboptimal arms and regret is also large.

That trade-off is the whole game.

### A concrete example (Gaussian bandit)

In the code for this repo we often use a simple setup:

- each arm has a true mean $q_*(a)$ sampled once at the beginning
- rewards are noisy observations around that mean

For example:
$$
R_t \mid (A_t=a) \sim \mathcal{N}(q_*(a), \sigma^2).
$$

So even if an arm is "good", you can still get unlucky rewards early.  
That’s why pure greedy strategies can fail.

Bandits can be seen as MDPs with:
- One state,
- Only the action choice matters,
- No state transition structure.

---

## 2) Action-Value estimates (how we learn which arm is good)

Since we don’t know $q_*(a)$, we maintain an estimate $Q_t(a)$ that we update over time.

We also keep a count:

- $N_t(a)$ = how many times we have selected arm $a$ up to time $t$

### Sample-average estimate

The most basic estimate of the mean reward of an arm is the sample average:

If you played arm $a$ exactly $N_t(a)$ times and observed rewards
$R_1^{(a)}, R_2^{(a)}, \dots, R_{N_t(a)}^{(a)}$ for that arm, then:

$$
Q_t(a) = \frac{1}{N_t(a)} \sum_{i=1}^{N_t(a)} R_i^{(a)}.
$$

This makes sense because it’s just the empirical mean of what you observed.

### Incremental update (same result, easier to implement)

We don’t want to store all rewards forever.  
Instead, we update the mean incrementally.

Suppose at time $t$ we selected arm $a$ and observed reward $R_t$.  
Let the old estimate be $Q_{t-1}(a)$ and old count be $N_{t-1}(a)$.

Update the count:
$$
N_t(a) = N_{t-1}(a) + 1.
$$

Then the sample-average update is:

$$
Q_t(a) = Q_{t-1}(a) + \frac{1}{N_t(a)} \left(R_t - Q_{t-1}(a)\right).
$$

This is the exact same as recomputing the mean from scratch, just written in a smarter form.

### Why the step size shrinks over time

Notice the learning rate is:

$$
\alpha_t = \frac{1}{N_t(a)}.
$$

- Early on, $N_t(a)$ is small → updates are big → estimates adapt quickly.
- Later, $N_t(a)$ is large → updates are small → estimates stabilize.

This is good in stationary problems (means don’t change).  
For non-stationary bandits, you often use a constant step size instead (we’ll mention that later).

### A practical detail: tie-breaking

When you compute $\arg\max_a Q(a)$, you might get ties early (many $Q(a)$ equal).
Indeed, when an algorithm says "pick the greedy action"
$$
a_t = \arg\max_a Q(a),
$$
it assumes there is a single best action. But in code, ties happen a lot, especially at the beginning:
* At the very start, many estimates are identical (e.g., all $Q(a)$)
* Or two arms may end up with the same estimate after a few updates (especially with discrete rewards or small samples)

So $\arg\max$ is not uniquely defined. There are multiple actions that maximize $Q(a)$. The algorithm still works, but your implementation must decide what to return.

Common tie-breaking choices:

* **First max (deterministic)**
  `np.argmax(Q)` returns the first index that reaches the maximum.
  Example: if `Q = [0.0, 0.0, 0.0]`, it will always return action `0`.
  This can bias early behavior because you keep picking the same arm until exploration kicks in.

* **Random among the max (stochastic)**
  Find all actions with the max value and choose randomly among them.
  Example: if all are tied, you pick uniformly at random, which tends to be fairer early on.

Different implementations break ties differently.  
This can slightly change early behavior but not the big picture. Indeed:
* In the first few steps, those tie decisions can affect which arms get sampled first
* Over many steps (and averaged over many runs), the effect usually washes out, but for a single run you might see noticeably different trajectories

---

## 3) $\varepsilon$-greedy exploration

**Goal:** balance exploitation and exploration.
- **Exploitation**: choose the arm with the highest current estimate $\left( Q(a)\right)$,
- **Exploration**: try other arms to avoid getting stuck with a suboptimal choice.

At each step:

- with probability $\varepsilon$: explore $\rightarrow$ choose a random arm
- with probability $1-\varepsilon$: exploit $\rightarrow$ choose the greedy arm

Formally:

$$
A_t =
\begin{cases}
\text{random arm} & \text{with prob. } \varepsilon \\
\arg\max_a Q_t(a) & \text{with prob. } 1-\varepsilon
\end{cases}
$$

Common choices:
- constant $\varepsilon$ (e.g. 0.1)
- decaying $\varepsilon$ which starts large (explore more early) and decreases over time (exploit more later).

After observing $R_t$, we update $Q_t(a)$ using the incremental mean from the previous section.

### Decaying $\varepsilon$

Sometimes you want a lot of exploration at the beginning, then mostly exploitation later.

A simple linear schedule is:

$$
\varepsilon_t = \max\left(\varepsilon_{\text{end}},\ \varepsilon_{\text{start}} +
\min\left(1, \frac{t}{T}\right)\left(\varepsilon_{\text{end}}-\varepsilon_{\text{start}}\right)\right)
$$

Here, $T$ is the number of steps over which you want $\varepsilon$ to decay.

- early on: $\varepsilon_t \approx \varepsilon_{\text{start}}$ (almost random)
- later: $\varepsilon_t \to \varepsilon_{\text{end}}$ (mostly greedy)

This method lets the agent explore a lot at the beginning, while (as the number of steps grows) $\varepsilon$ becomes smaller, so the agent exploits more.

**Pros:** easy and often works well.  
**Cons:** if you decay too fast, you can still get stuck like fixed greedy. If you decay too slowly, you keep exploring and sacrifice reward.

---

## 4) Upper Confidence Bound (UCB)

$\varepsilon$-greedy explores randomly. It does not care whether an arm is:
- clearly bad (tried many times, low reward), or
- uncertain (tried only a few times, could still be good)

UCB methods explore more intelligently by adding an uncertainty bonus.

Prefer arms that:
  - Have high estimated value $\left( Q(a)\right)$,
  - Or have **high uncertainty** (i.e., they have been tried few times).

A common UCB1-style rule (UCB1 is the name of a classic algorithm, from the multi-armed bandit literature, that uses an upper confidence bound built from a particular theoretical bound):

$$
A_t = \arg\max_a \left[
Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}}
\right]
$$

Where:
- $Q_t(a)$ is the current estimate (exploitation term)
- $\sqrt{\frac{\ln t}{N_t(a)}}$ is the exploration bonus
- $c > 0$ controls how aggressive exploration is
- $t$ starts from 1 to avoid $\ln(0)$

### Intuition

- The first term $\left( Q_t(a)\right)$ favors arms that look good so far.
- The second term is a confidence bonus.
- If $N_t(a)$ is small, the bonus is large $\rightarrow$ try that arm more.
- If $N_t(a)$ is large, the bonus shrinks $\rightarrow$ stop wasting time on it.
- As $t$ grows, $\ln t$ increases slowly $\rightarrow$ the algorithm still forces occasional exploration.

### Practical detail in this repo

To avoid division by zero in a clean way, we do:
- pull each arm once at the beginning
- then apply the UCB formula

---

## 5) Common pitfalls and pros/cons

This section is here because bandits can look "weird" the first time you run them.  
Most confusion comes from randomness plus early bad luck.

### Pitfall 1: Pure greedy can get stuck (and look "confidently wrong")

If you set $\varepsilon = 0$, the agent picks whatever arm looks best early.

Problem: early rewards are noisy.  
If a mediocre arm gets a lucky high reward early, greedy can commit to it and never recover.

**Pro (greedy):**
- very simple
- high exploitation

**Con (greedy):**
- no mechanism to correct early mistakes

---

### Pitfall 2: $\varepsilon$ too small vs too large

With $\varepsilon$-greedy, the exploration rate matters a lot.

- If $\varepsilon$ is too small: you explore too rarely $\rightarrow$ slow recovery from early bad choices.
- If $\varepsilon$ is too large: you keep exploring forever $\rightarrow$ reward stays lower than it could be.

**Pro ($\varepsilon$-greedy):**
- extremely simple baseline
- robust enough to work "okay" in many settings

**Con ($\varepsilon$-greedy):**
- exploration is random and wasteful (tries arms you already know are bad)

A common beginner experiment:
- compare $\varepsilon = 0$, $0.01$, $0.1$, $0.5$ and look at reward + % optimal action.

---

### Pitfall 3: Optimistic initialization changes the behavior a lot

If you initialize $Q(a)$ to a high value (higher than typical rewards),
even a greedy agent will "explore" because every arm looks good at the start.
After trying an arm once or twice, its estimate drops, so the agent moves on.

This can be a nice trick, but it’s also easy to misunderstand.

**Pro:**
- encourages exploration without randomness

**Con:**
- depends on knowing a reasonable reward scale
- can behave strangely if rewards are unbounded or highly noisy

---

### Pitfall 4: UCB can look aggressive early (that’s the point)

UCB often forces exploration early by pulling under-sampled arms.

To beginners this can look like "why is it trying arms that seem bad?"

The reason is that UCB is being optimistic:
until it has enough evidence, it treats uncertainty as potential upside.

**Pro (UCB):**
- explores more intelligently than random exploration
- tends to lock onto good arms efficiently

**Con (UCB):**
- needs careful handling at the start (we solve this by pulling each arm once)
- the exploration strength depends on $c$ and can be sensitive in practice

---

### Pitfall 5: Stationary vs non-stationary bandits

Everything above assumes the reward distribution for each arm is fixed (stationary).

If the true values change over time (non-stationary):
- sample-average updates become slow to adapt (old data never "expires")
- you usually want a constant step size update:
  $$Q(a) \leftarrow Q(a) + \alpha (r - Q(a))$$
  with a fixed $\alpha$ (e.g. 0.1)

This repo starts with stationary bandits because they’re easier to reason about.

---

### Pitfall 6: Don’t over-read single runs

Bandit curves can vary a lot between random seeds, especially early on.

That’s why most plots average over many runs.
If you only run once, you might draw the wrong conclusions.

---

## 6) What you should get from this block

After this section, you should be able to:

1. Explain the exploration–exploitation trade-off in plain words.
2. Implement:
   - $\varepsilon$-greedy with sample-average updates
   - UCB action selection
3. Reproduce classic plots:
   - average reward over time
   - % optimal action over time

This is the foundation for what happens next with MDPs (value iteration, Q-learning, policy gradients, ...).
