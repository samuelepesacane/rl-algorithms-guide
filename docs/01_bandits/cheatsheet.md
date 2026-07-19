# 01 - Bandits Cheatsheet

A one-page refresher for when you already learned this block and just want the
key ideas back fast. For the full story (regret, derivations, pitfalls) read
`docs/01_bandits/theory.md`.

The whole block is about one tension: **exploration vs exploitation**. You do
not know which arm is best until you try it, but every try you spend on a bad
arm is reward you gave up. Bandits are the smallest setting where this matters:
one state (or none), no transitions, just pick an action and get a reward.

---

## The objects you keep reusing

- **Action-value estimate** $Q_t(a)$: your current guess of arm $a$'s mean reward.
- **Count** $N_t(a)$: how many times you have pulled arm $a$ so far.
- **Incremental mean update** (same as recomputing the average, but cheap):

$$
Q_t(a) = Q_{t-1}(a) + \frac{1}{N_t(a)}\big(R_t - Q_{t-1}(a)\big)
$$

The step size $1/N_t(a)$ shrinks over time, so estimates adapt fast early and
settle later. For **non-stationary** arms, swap it for a constant $\alpha$ (e.g.
$0.1$) so old data eventually expires.

---

## At a glance

| Method | How it explores | Key knob | Wins when |
|---|---|---|---|
| Greedy ($\varepsilon=0$) | not at all | (none) | never really; baseline only |
| $\varepsilon$-greedy | random, fixed rate | $\varepsilon$ | simple, robust default |
| UCB | targets uncertain arms | $c$ | you want efficient, directed exploration |
| Optimistic init | high initial $Q$ drives early trying | $Q_0$ | rewards have a known scale |

---

## $\varepsilon$-greedy

**Problem it solves:** pure greedy commits to whatever looked good early and
never recovers from an unlucky start.

**Rule:**

$$
A_t =
\begin{cases}
\text{random arm} & \text{prob. } \varepsilon \\
\arg\max_a Q_t(a) & \text{prob. } 1-\varepsilon
\end{cases}
$$

**Defining idea:** explore a fixed fraction of the time, blindly.

**Key knobs (typical):** $\varepsilon \approx 0.1$. Optionally decay it from
$\approx 1.0$ down to $\approx 0.05$ so you explore early, exploit later.

**Watch out for:** too small $\varepsilon$ recovers slowly from early mistakes;
too large keeps wasting pulls on arms you already know are bad. Exploration is
undirected: it re-tries known-bad arms as often as uncertain ones.

---

## UCB (Upper Confidence Bound)

**Problem it solves:** $\varepsilon$-greedy explores at random; it does not
prefer arms it is genuinely uncertain about.

**Rule (UCB1-style):**

$$
A_t = \arg\max_a \left[ Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

**Defining idea:** be optimistic under uncertainty. The bonus is large for
rarely-pulled arms and shrinks as $N_t(a)$ grows, so exploration is directed at
arms that might still be good rather than at random ones.

**Key knobs (typical):** $c \approx 2$. Pull each arm once at the start to avoid
dividing by $N_t(a)=0$.

**Watch out for:** UCB looks aggressive early on purpose (it pulls under-sampled
arms). Sensitive to $c$ and to the start-up handling.

---

## Two tricks worth remembering

- **Optimistic initialisation:** set $Q_0(a)$ above any plausible reward. Even a
  greedy agent then "explores", because every untried arm looks great until it
  is tried. Needs a known reward scale.
- **Decaying $\varepsilon$:** a linear schedule from $\varepsilon_{\text{start}}$
  to $\varepsilon_{\text{end}}$ over $T$ steps. Decay too fast and you behave
  like greedy; too slow and you keep sacrificing reward.

---

## Mental traps

- Pure greedy can be "confidently wrong" after one lucky early reward.
- Never read a single run: bandit curves swing a lot by seed, so average over
  many runs (this is why the standard plots are over many runs).
- Stationary vs non-stationary changes the right step size ($1/N$ vs constant).

---

## Where to look in the repo

- Code: `src/rl_algorithms_guide/bandits/{bandit_envs,epsilon_greedy,ucb}.py`
- Run: `examples/01_bandits/train_compare.py`
- Full theory: `docs/01_bandits/theory.md`

**Next block:** `docs/02_tabular/` adds states and transitions (MDPs), which is
where value functions and Q-learning enter.
