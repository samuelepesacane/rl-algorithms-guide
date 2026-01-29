"""
This script contains two things:

1. Policy evaluation: given a fixed policy \pi, compute V^\pi using DP backups
2. Policy iteration: alternate evaluate -> improve until the policy stops changing
"""

from __future__ import annotations

import numpy as np
from rl_algorithms_guide.tabular.gridworld import GridworldMDP


def policy_evaluation(
    mdp: GridworldMDP,
    policy: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iters: int = 100_000,
    V_init: np.ndarray | None = None,
) -> np.ndarray:
    """
    Iterative policy evaluation for a finite MDP.

    For the value estimate V, we use a warm-start evaluation V_init.
    Standard evaluation starts from zero (V = 0); here we optionally warm-start (if V_init is provided, we warm-start from it; otherwise we start from zeros).
    We can speed things up by reusing the previous V across outer iterations (same result, fewer eval sweeps).

    In policy iteration we:
    1. evaluate current policy \pi_k -> get V^{\pi_k}
    2. improve to \pi_{k+1}
    3. evaluate \pi_{k+1} again

    But \pi_{k+1} is usually only a small change from \pi_k, so V^{\pi_{k+1}} is usually close to V^{\pi_k}.
    If we restart from zero each time, we throw away useful information and need more evaluation sweeps to converge again.
    Warm-start keeps that information, so it converges in fewer sweeps.

    :param mdp: MDP instance.
        :type mdp: GridworldMDP
    :param policy: Array of actions, one per state (shape: (n_states,)).
        :type policy: np.ndarray
    :param gamma: Discount factor.
        :type gamma: float
    :param theta: Convergence threshold.
        :type theta: float
    :param max_iters: Max evaluation iterations.
        :type max_iters: int
    :param V_init: previous value estimate used as the initial guess for the next evaluation.
        :type V_init: np.ndarray | None

    :return: Value function V^pi (shape: (n_states,)).
        :rtype: np.ndarray
    """
    nS = mdp.n_states  # number of states

    if V_init is None:
        # we initialize the state values with a zero-array (in theory.md we said that we can start with V_0 = 0)
        V = np.zeros(nS, dtype=np.float64)
    else:
        if V_init.shape != (nS,):
            raise ValueError(f"V_init must have shape ({nS},), got {V_init.shape}")
        # warm-start evaluation -> don't restart policy evaluation from V=0 every time you improve the policy
        # instead, reuse the previous value estimate as the initial guess for the next evaluation
        V = V_init.astype(np.float64, copy=True)

    if policy.shape != (nS,):
        raise ValueError(f"policy must have shape ({nS},), got {policy.shape}")
    if np.any((policy < 0) | (policy >= mdp.n_actions)):
        raise ValueError(f"policy must contain ints in [0, {mdp.n_actions - 1}]")

    for s in mdp.terminal_states:
        # in this Gridworld implementation, terminal states are absorbing with reward 0
        # so the remaining return from terminal is 0 by convention
        V[s] = 0.0  # just in case V_init is a little weird

    for _ in range(max_iters):
        delta = 0.0

        for s in range(nS):
            if mdp.is_terminal(s):
                # Terminal values are fixed by convention (absorbing, no future reward)
                # their value is 0 for remaining return in this Gridworld setup
                # So we can skip them
                continue

            v_old = V[s]
            a = int(policy[s])  # action

            v_new = 0.0
            for tr in mdp.transitions(s, a):
                bootstrap = 0.0 if tr.done else gamma * V[tr.next_state]  # no bootstrapping past terminal
                v_new += tr.prob * (tr.reward + bootstrap)

            V[s] = v_new
            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            break

    return V


def policy_iteration(
    mdp: GridworldMDP,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_eval_iters: int = 100_000,
    max_improve_iters: int = 1_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Policy Iteration (evaluation + improvement) for a finite MDP.

    This is the classic loop:
    1. Evaluate current policy \pi -> get V^\pi
    2. Improve policy greedily w.r.t. that value function
    3. Repeat until policy is stable (no changes)

    :param mdp: MDP instance.
        :type mdp: GridworldMDP
    :param gamma: Discount factor.
        :type gamma: float
    :param theta: Convergence threshold for evaluation.
        :type theta: float
    :param max_eval_iters: Max iterations inside policy evaluation.
        :type max_eval_iters: int
    :param max_improve_iters: Max outer policy improvement iterations.
        :type max_improve_iters: int

    :return: (V, policy)
        - V: value function for the final policy
        - policy: greedy policy (stable)
        :rtype: tuple[np.ndarray, np.ndarray]
    """
    nS = mdp.n_states  # number of states
    nA = mdp.n_actions  # number of actions

    policy = np.zeros(nS, dtype=np.int64)  # start with "always up" just to have something
    # this is just a starting point
    # policy iteration will improve it anyway

    V = np.zeros(nS, dtype=np.float64)  # put just in case we somehow give max_improve_iters < 1

    for _ in range(max_improve_iters):
        # get V with current policy
        V = policy_evaluation(
            mdp=mdp,
            policy=policy,
            gamma=gamma,
            theta=theta,
            max_iters=max_eval_iters,
            V_init=V,
        )

        policy_stable = True

        # for each state, this function computes
        # q_\pi(s,a)=\sum_{s'} P(s'|s,a)\Big(R(s,a,s') + \gamma V^\pi(s')\Big)
        # Then it sets
        # \pi(s) <- \arg\max_a q_\pi(s,a)
        for s in range(nS):
            if mdp.is_terminal(s):
                continue

            old_action = int(policy[s])

            q_values = np.zeros(nA, dtype=np.float64)
            for a in range(nA):
                q = 0.0
                for tr in mdp.transitions(s, a):
                    bootstrap = 0.0 if tr.done else gamma * V[tr.next_state]
                    q += tr.prob * (tr.reward + bootstrap)
                q_values[a] = q

            max_q = np.max(q_values)
            all_best_actions = np.flatnonzero(np.isclose(a=q_values, b=max_q, rtol=1e-12, atol=1e-12))  # safer way to compare floats
            best_action = int(all_best_actions[0])  # or int(mdp.rng.choice(all_best_actions)) if you pass rng (for random tie-breaking)
            policy[s] = best_action
            # otherwise you can just do this (it is the same as int(all_best_actions[0]))
            # best_action = int(np.argmax(q_values))
            # policy[s] = best_action

            if best_action != old_action:
                policy_stable = False
            # if no state changes its greedy action in an improvement sweep, then:
            # policy is stable
            # algorithm stops
            # this corresponds to reaching an optimal policy in this finite MDP setting

        if policy_stable:
            break

    return V, policy
