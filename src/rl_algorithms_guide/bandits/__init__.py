"""
Bandit algorithms (stateless RL).

This subpackage contains:
- a simple Gaussian k-armed bandit environment
- ε-greedy agent
- UCB agent
"""

from .bandit_envs import GaussianBandit
from .epsilon_greedy import EpsilonGreedyAgent, DecayingEpsilonGreedyAgent
from .ucb import UCBAgent

__all__ = [
    "GaussianBandit",
    "EpsilonGreedyAgent",
    "DecayingEpsilonGreedyAgent",
    "UCBAgent",
]
