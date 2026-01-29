import numpy as np

from rl_algorithms_guide.tabular import GridworldMDP, value_iteration, policy_iteration


def _q_value_from_V(mdp: GridworldMDP, V: np.ndarray, s: int, a: int, gamma: float) -> float:
    """
    Compute the one-step lookahead Q(s,a) given a value function V.

    This mirrors the Bellman backup used in value iteration/policy extraction.

    :param mdp: MDP providing transitions(s, a).
        :type mdp: GridworldMDP
    :param V: State-value function array, shape (n_states,).
        :type V: np.ndarray
    :param s: State index.
        :type s: int
    :param a: Action index.
        :type a: int
    :param gamma: Discount factor.
        :type gamma: float

    :return: One-step lookahead action value.
        :rtype: float
    """
    q = 0.0
    for tr in mdp.transitions(s, a):
        bootstrap = 0.0 if tr.done else gamma * float(V[tr.next_state])
        q += tr.prob * (tr.reward + bootstrap)
    return float(q)


def test_value_iteration_policy_is_greedy_wrt_V():
    """
    Value iteration should return a policy that is greedy w.r.t. the returned V.
    """
    mdp = GridworldMDP(height=4, width=4, terminal_states=((0, 0), (3, 3)), step_reward=-1.0)

    gamma = 0.99
    V, policy = value_iteration(mdp=mdp, gamma=gamma, theta=1e-8, max_iters=10_000)

    # Terminal values should stay at 0 in this setup
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            assert np.isclose(V[s], 0.0)


    # For each non-terminal state, the chosen action must be among the greedy actions
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            continue

        q_values = np.array(
            [_q_value_from_V(mdp=mdp, V=V, s=s, a=a, gamma=gamma) for a in range(mdp.n_actions)],
            dtype=np.float64,
        )
        best_q = float(np.max(q_values))
        greedy_actions = set(np.flatnonzero(np.isclose(a=q_values, b=best_q, atol=1e-10)).tolist())

        assert int(policy[s]) in greedy_actions


def test_policy_iteration_policy_is_greedy_wrt_its_V():
    """
    Policy iteration should return a stable greedy policy w.r.t. its evaluated V.
    """
    mdp = GridworldMDP(height=4, width=4, terminal_states=((0, 0), (3, 3)), step_reward=-1.0)

    gamma = 0.99
    V, policy = policy_iteration(
        mdp=mdp,
        gamma=gamma,
        theta=1e-8,
        max_eval_iters=10_000,
        max_improve_iters=1_000,
    )

    # Same greedy check as above, but using V returned by policy iteration
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            continue

        q_values = np.array(
            [_q_value_from_V(mdp=mdp, V=V, s=s, a=a, gamma=gamma) for a in range(mdp.n_actions)],
            dtype=np.float64,
        )
        best_q = float(np.max(q_values))
        greedy_actions = set(np.flatnonzero(np.isclose(a=q_values, b=best_q, atol=1e-10)).tolist())

        assert int(policy[s]) in greedy_actions
