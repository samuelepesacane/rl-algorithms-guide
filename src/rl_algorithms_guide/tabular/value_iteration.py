from __future__ import annotations

import numpy as np
from rl_algorithms_guide.tabular.gridworld import GridworldMDP


def value_iteration(
    mdp: GridworldMDP,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iters: int = 100_000,
    deltas: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration for a finite MDP.

    Repeatedly applies the Bellman optimality backup:

        V(s) <- max_a sum_{s'} P(s'|s,a) [r + gamma * V(s')]

    this function follows the same steps described in theory.md:
        - loop many times (iterations)
        - for each state, compute best action value and update V(s)
        - stop when changes are tiny
        - then extract a greedy policy from the final V


    :param mdp: MDP that exposes mdp.n_states, mdp.n_actions, and mdp.transitions(s,a).
        :type mdp: GridworldMDP
    :param gamma: Discount factor in [0, 1).
        :type gamma: float
    :param theta: Convergence threshold (stop when max value change < theta).
        :type theta: float
    :param max_iters: Maximum number of iterations.
        :type max_iters: int
    :param deltas: Optional list to store the per-iteration convergence metric delta_k = max_s |V_{k+1}(s) - V_k(s)|.
        If provided, this function appends one value per iteration.
        :type deltas: list[float] | None

    :return: (V, policy)
        - V: optimal state values (shape: (n_states,))
        - policy: greedy policy w.r.t. V (shape: (n_states,), ints in [0, n_actions-1])
        :rtype: tuple[np.ndarray, np.ndarray]
    """
    nS = mdp.n_states  # number of states
    nA = mdp.n_actions  # number of actions

    # we initialize the state values with a zero-array (in theory.md we said that we can start with V_0 = 0)
    V = np.zeros(nS, dtype=np.float64)

    for _ in range(max_iters):
        # delta tracks the maximum change in any state value during an iteration
        # so delta = max_s |V_new(s) - V_old(s)|
        delta = 0.0

        for s in range(nS):
            if mdp.is_terminal(s):
                # Terminal values are fixed by convention (absorbing, no future reward)
                # their value is 0 for remaining return in this Gridworld setup
                # So we can skip them
                continue

            v_old = V[s]  # keep the old value to measure change

            # compute action-values using the model
            # we are using q(s,a)=\sum_{s'} P(s'|s,a)\big(R(s,a,s') + \gamma V(s')\big)
            # This code version works for deterministic and stochastic MDPs
            # Even though our gridworld is deterministic, 'transitions()' returns a list, so the DP code is general
            # deterministic -> list length 1;  stochastic -> list length > 1
            q_values = np.zeros(nA, dtype=np.float64)  # we initialize the state-action values with a zero-array (as V)
            for a in range(nA):
                q = 0.0
                for tr in mdp.transitions(s, a):
                    bootstrap = 0.0 if tr.done else gamma * V[tr.next_state]  # no bootstrapping past terminal
                    q += tr.prob * (tr.reward + bootstrap)
                q_values[a] = q

            # we apply the optimality backup: V(s)=\max_a q(s,a)
            # this is the max over actions part that turns prediction into control
            V[s] = float(np.max(q_values))
            delta = max(delta, abs(v_old - V[s]))  # we find the max difference across all states

        if deltas is not None:
            deltas.append(float(delta))

        if delta < theta:  # the max change across states is small
            break

    # Extract greedy policy
    # remember that once you have the optimal V, (let's denote it with V_*), the optimal policy is
    # \pi_*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \big(R+\gamma V_*(s')\big)
    policy = np.zeros(nS, dtype=np.int64)
    for s in range(nS):
        if mdp.is_terminal(s):
            policy[s] = 0
            # small note on terminal policy, we set terminal action to 0, this is fine because it's unused
            # terminal states won't be stepped from in our DP training
            continue

        q_values = np.zeros(nA, dtype=np.float64)
        for a in range(nA):
            q = 0.0
            for tr in mdp.transitions(s, a):
                bootstrap = 0.0 if tr.done else gamma * V[tr.next_state]
                q += tr.prob * (tr.reward + bootstrap)
            q_values[a] = q

        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(a=q_values, b=max_q, rtol=1e-12, atol=1e-12))
        policy[s] = int(best_actions[0])  # or int(mdp.rng.choice(best_actions)) if you pass rng (for random tie-breaking)

    return V, policy
