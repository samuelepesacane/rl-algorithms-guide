from __future__ import annotations
import numpy as np


class EpsilonGreedyAgent:
    """
    ε-greedy agent for k-armed bandits using sample-average value estimates.

    What it stores:
    - Q[a]: estimated value (mean reward) of arm a
    - N[a]: how many times arm a has been selected

    Action selection:
    - with probability ε: choose a random arm
    - with probability 1-ε: choose an arm with the highest Q

    Update (incremental mean):
        N[a] <- N[a] + 1
        Q[a] <- Q[a] + (1 / N[a]) * (r - Q[a])

    :param k: Number of arms.
        :type k: int
    :param epsilon: Exploration probability (fixed for the whole run).
        :type epsilon: float
    :param initial_value: Initial Q value for all arms (can be optimistic).
        :type initial_value: float
    :param seed: Seed for agent RNG (controls exploration randomness and tie-breaking).
        :type seed: int | None
    """

    def __init__(
        self,
        k: int,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        seed: int | None = None,
    ):
        self.k = int(k)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

        self.Q = np.full(self.k, float(initial_value), dtype=np.float64)
        self.N = np.zeros(self.k, dtype=np.int64)

    def select_action(self) -> int:
        """
        Choose an arm index using fixed ε-greedy.

        :return: Arm index in [0, k-1].
            :rtype: int
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(low=0, high=self.k))

        # Random tie-breaking avoids a subtle "arm 0 bias" when values are equal
        max_q = np.max(self.Q)
        best_actions = np.flatnonzero(self.Q == max_q)
        return int(self.rng.choice(best_actions))

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


class DecayingEpsilonGreedyAgent:
    """
    ε-greedy agent with a decaying exploration rate.

    This is useful when you want:
    - heavy exploration early
    - mostly exploitation later

    Two schedules are supported:
    - "linear": ε goes from epsilon_start to epsilon_end over decay_steps steps
    - "exp": ε decays exponentially toward epsilon_end

    :param k: Number of arms.
        :type k: int
    :param epsilon_start: Initial ε at the beginning (e.g., 1.0).
        :type epsilon_start: float
    :param epsilon_end: Minimum/final ε after decay (e.g., 0.05 or 0.1).
        :type epsilon_end: float
    :param decay_steps: Number of steps over which ε decays (linear) or time constant (exp).
        :type decay_steps: int
    :param schedule: Decay schedule, either "linear" or "exp".
        :type schedule: str
    :param initial_value: Initial Q value for all arms.
        :type initial_value: float
    :param seed: Seed for agent RNG.
        :type seed: int | None
    """

    def __init__(
        self,
        k: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        decay_steps: int = 500,
        schedule: str = "linear",
        initial_value: float = 0.0,
        seed: int | None = None,
    ):
        self.k = int(k)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.decay_steps = int(decay_steps)
        self.schedule = str(schedule).lower()

        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be >= 1")

        if self.schedule not in {"linear", "exp"}:
            raise ValueError('schedule must be either "linear" or "exp"')

        self.rng = np.random.default_rng(seed)

        self.Q = np.full(self.k, float(initial_value), dtype=np.float64)
        self.N = np.zeros(self.k, dtype=np.int64)

        # Time step for epsilon scheduling (number of actions selected so far).
        self.t = 0

    def current_epsilon(self) -> float:
        """
        Compute ε at the current time step (based on self.t).

        :return: Current exploration probability ε_t.
            :rtype: float
        """
        if self.schedule == "linear":
            frac = min(1.0, self.t / self.decay_steps)
            eps = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
            return float(max(self.epsilon_end, eps))

        # Exponential decay toward epsilon_end
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.t / self.decay_steps
        )
        return float(max(self.epsilon_end, eps))

    def select_action(self) -> int:
        """
        Choose an arm using decaying ε-greedy.

        :return: Arm index in [0, k-1].
            :rtype: int
        """
        eps = self.current_epsilon()
        self.t += 1

        if self.rng.random() < eps:
            return int(self.rng.integers(low=0, high=self.k))

        max_q = np.max(self.Q)
        best_actions = np.flatnonzero(self.Q == max_q)
        return int(self.rng.choice(best_actions))

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
