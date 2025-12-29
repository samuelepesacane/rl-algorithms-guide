from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BanditInfo:
    """
    Lightweight container with "ground-truth" bandit information.

    :param q_star: True mean reward for each arm.
        :type q_star: np.ndarray
    :param best_action: Index of the arm with the highest true mean.
        :type best_action: int
    """
    q_star: np.ndarray
    best_action: int


class GaussianBandit:
    """
    Stationary k-armed Gaussian bandit.

    Each arm 'a' returns a noisy reward sampled from:
        R | A=a ~ Normal(q_star[a], std^2)

    :param k: Number of arms.
        :type k: int
    :param std: Standard deviation of reward noise.
        :type std: float
    :param mean_range: Range (low, high) used to sample q_star uniformly if q_star is not provided.
        :type mean_range: tuple[float, float]
    :param q_star: Optional vector of true means (shape (k,)). If provided, means are not sampled.
        :type q_star: np.ndarray | None
    :param seed: Seed for the RNG (controls reward noise and q_star sampling if needed).
        :type seed: int | None
    """

    def __init__(
        self,
        k: int = 10,
        std: float = 1.0,
        mean_range: tuple[float, float] = (-1.0, 1.0),
        q_star: np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.k = int(k)
        self.std = float(std)
        self.mean_range = mean_range
        self.rng = np.random.default_rng(seed)

        if q_star is None:
            low, high = mean_range
            self.q_star = self.rng.uniform(low, high, size=self.k).astype(np.float64)
        else:
            q_star = np.asarray(q_star, dtype=np.float64)
            if q_star.shape != (self.k,):
                raise ValueError(f"q_star must have shape ({self.k},), got {q_star.shape}")
            self.q_star = q_star

        self.best_action = int(np.argmax(self.q_star))

    def reset(self) -> BanditInfo:
        """
        Reset the bandit.

        For stationary bandits there is no internal state to reset, but returning
        the true means and the best arm is handy for evaluation.

        :return: BanditInfo containing q_star and best_action.
            :rtype: BanditInfo
        """
        return BanditInfo(q_star=self.q_star.copy(), best_action=self.best_action)

    def step(self, action: int) -> float:
        """
        Pull an arm and sample a reward.

        :param action: Arm index in [0, k-1].
            :type action: int

        :return: Sampled reward from the selected arm.
            :rtype: float
        """
        if not (0 <= action < self.k):
            raise ValueError(f"action must be in [0, {self.k - 1}], got {action}")

        mean = self.q_star[action]
        reward = self.rng.normal(loc=mean, scale=self.std)
        return float(reward)
