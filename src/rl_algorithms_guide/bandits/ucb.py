from __future__ import annotations
import numpy as np


class UCBAgent:
    """
    UCB (Upper Confidence Bound) agent for k-armed bandits (UCB1-style).

    This implementation uses a friendly initialization:
    - pull each arm once first (so N[a] >= 1 for all a)
    - then apply the UCB formula

    UCB score:
        score(a) = Q[a] + c * sqrt( ln(t) / N[a] )

    :param k: Number of arms.
        :type k: int
    :param c: Exploration strength (larger c => more exploration).
        :type c: float
    :param initial_value: Initial Q values for all arms.
        :type initial_value: float
    """

    def __init__(self, k: int, c: float = 2.0, initial_value: float = 0.0):
        self.k = int(k)
        self.c = float(c)

        self.Q = np.full(self.k, float(initial_value), dtype=np.float64)
        self.N = np.zeros(self.k, dtype=np.int64)

        # Global time step (number of actions chosen so far)
        self.t = 0

    def select_action(self) -> int:
        """
        Choose an arm using UCB.

        The first k calls return arms 0...k-1 (each arm is tried once).
        After that, we compute UCB scores and pick the argmax.

        :return: Arm index in [0, k-1].
            :rtype: int
        """
        self.t += 1  # global time step (starts at 1 to avoid log(0))

        if self.t <= self.k:
            # Warm-up: try each arm once so N[a] is never zero in the UCB formula
            return self.t - 1

        # Exploration bonus is larger for arms with small N[a] (less tried/more uncertain)
        bonus = self.c * np.sqrt(np.log(self.t) / self.N)
        # UCB score = estimated value + uncertainty bonus
        scores = self.Q + bonus
        return int(np.argmax(scores))

    def update(self, action: int, reward: float) -> None:
        """
        Update Q[action] using incremental sample-average.

        :param action: Arm index selected.
            :type action: int
        :param reward: Observed reward.
            :type reward: float

        :return: None.
        :rtype: None
        """
        self.N[action] += 1
        alpha = 1.0 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])
