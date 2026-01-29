from __future__ import annotations

import numpy as np


class ExpectedSARSAgent:
    """
    Tabular Expected SARSA agent with ε-greedy behaviour.
    This file implements Expected SARSA, which is:
        - Tabular: stores Q[s,a] in a table
        - On-policy: it evaluates the same behaviour policy, \mu (ε-greedy as SARSA), it uses to generate data
        - TD control
        - Different target than SARSA

    Instead of using the sampled next action,
    update target uses expected next value under ε-greedy (behaviour policy):

        E[Q(s',a')] = sum_a' \mu(a'|s') * Q(s',a')

    So:
    - same policy being evaluated -> on-policy (\mu = \pi)
    - lower variance than SARSA
    - still reflects exploration (unlike Q-learning)

    :param n_states: Number of discrete states.
        :type n_states: int
    :param n_actions: Number of discrete actions.
        :type n_actions: int
    :param alpha: Learning rate.
        :type alpha: float
    :param gamma: Discount factor.
        :type gamma: float
    :param epsilon: Exploration probability for ε-greedy.
        :type epsilon: float
    :param seed: RNG seed for action selection.
        :type seed: int | None
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        self.rng = np.random.default_rng(seed)
        # create the Q-table initialized to zero with shape (n_states, n_actions)
        # starting from all zeros is a standard practice for tabular RL
        self.Q = np.zeros(shape=(self.n_states, self.n_actions), dtype=np.float64)

    def select_action(self, state: int) -> int:
        """
        Select an action using ε-greedy.

        This function implements ε-greedy:
            1. With probability epsilon: explore -> random action
            2. Else: exploit -> choose an action with the maximum Q-value at that state

        :param state: Current state index.
            :type state: int

        :return: Action index.
            :rtype: int
        """
        # Exploration
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(low=0, high=self.n_actions))

        # Exploitation
        q = self.Q[state]  # all Q-values of that state (one value for each action)
        max_q = np.max(q)  # max Q-value at that state (to find the best actions)
        best_actions = np.flatnonzero(np.isclose(a=q, b=max_q))
        return int(self.rng.choice(best_actions))  # tie-breaking -> Early in learning, many actions have identical Q-values (often all 0)
        # If you always take the first argmax, you accidentally bias the policy -> random tie-breaking avoids that

    def _expected_next_value(self, next_state: int) -> float:
        """
        Compute E_a'~\mu[ Q(next_state, a') ] under ε-greedy.
        Remember: on-policy -> \pi = \mu

        Expected SARSA needs:

        E_{a'~\mu(.|s')} [Q(s',a')] = \sum_a' \mu(a'|s') * Q(s',a')

        So we need the probabilities \mu(a'|s') of taking each action at next_state under ε-greedy.
        ε-greedy means:
        - With probability ε: choose uniformly random among all actions -> each action gets epsilon/number_of_actions

        - With probability 1-ε: choose a greedy action
          If there are ties, split that mass evenly among the k greedy actions -> each greedy action gets (1-ε)/k (the exploitation mass)

        I really advise to check the section about Expected Sarsa in theory.md.

        :param next_state: Next state index.
            :type next_state: int

        :return: Expected action-value at next_state.
            :rtype: float
        """
        q = self.Q[next_state]  # get q-value of the next state
        max_q = np.max(q)
        best_actions = np.flatnonzero(np.isclose(a=q, b=max_q))  # find greedy actions

        if best_actions.size == 0:
            raise RuntimeError("No greedy actions found. Q may contain NaNs.")

        # for each action we assign a probability of epsilon/number_of_actions -> assume exploration happens, and it's uniform across all actions
        mu = np.full(shape=self.n_actions, fill_value=self.epsilon / self.n_actions, dtype=np.float64) # array of actions
        # for each of the greedy actions we add to the probability the value (1 - epsilon)/number_of_greedy_actions -> add the probability of choosing greedily
        # if there are ties, share it equally among all greedy actions
        mu[best_actions] += (1.0 - self.epsilon) / len(best_actions)

        return float(np.dot(mu, q))  # dot product between the probability vector and the Q-values

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Apply Expected SARSA update.

        Update:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * \sum_{a'} \mu(a' | s') Q(s',a') - Q(s,a)]

        :param state: Current state.
            :type state: int
        :param action: Action taken.
            :type action: int
        :param reward: Observed reward.
            :type reward: float
        :param next_state: Next state.
            :type next_state: int
        :param done: Whether the episode ended after this transition.
            :type done: bool

        :return: None
            :rtype: None
        """
        target = reward
        if not done:
            target += self.gamma * self._expected_next_value(next_state)

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
