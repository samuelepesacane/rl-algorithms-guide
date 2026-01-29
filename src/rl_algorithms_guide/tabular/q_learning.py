from __future__ import annotations

import numpy as np


class QLearningAgent:
    """
    Tabular Q-learning agent with ε-greedy exploration.

    This file implements tabular Q-learning:

        - Tabular: Q[s,a] stored in a table.
        - Off-policy TD control: you may behave ε-greedily to explore, but the update target assumes greedy behaviour at the next state.

    Update:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Important, see the max in 'update'. This is the difference between q-learning and SARSA.
    Even though the selected action 'a' is ε-greedy (exploratory), the update uses max_a' Q(s',a') (greedy target policy).
    So:
        - behaviour policy \mu: ε-greedy (used to collect data)
        - target policy \pi: greedy (used inside the update)

    That mismatch \pi != \mu is exactly what off-policy means.

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

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Apply the Q-learning update.

        Update:
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

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
            target += self.gamma * float(np.max(self.Q[next_state]))  # this is where Q-learning differs from SARSA
            # np.max(self.Q[next_state]) -> it comes from target policy \pi that is different from behaviour policy \mu

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
