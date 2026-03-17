"""
REINFORCE (Monte Carlo policy gradient) for discrete action spaces.

This module implements REINFORCE as described in docs/04_pg_ac/theory.md
(Sections 10-10.3).

REINFORCE is the simplest policy gradient algorithm:
- run one full episode with the current policy
- compute reward-to-go G_t for each step (Monte Carlo return)
- update: for each step t, push up the log-prob of the taken action if G_t is high,
  push it down if G_t is low

This is unbiased (we use real returns, not estimates), but high variance
because a single lucky or unlucky episode can dominate the gradient.

Two optional variance-reduction techniques are included:

1) Running-mean baseline (Section 8): subtract a scalar EMA of past episode
   returns from G_t. This centers the signal without biasing the gradient.

2) Advantage normalization (Section 13): standardize the (baseline-subtracted)
   returns within each episode to zero mean and unit std. Slightly biased
   but often stabilizes learning in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from rl_algorithms_guide.common.utils import (
    RunningMeanBaseline,
    compute_discounted_returns,
    normalize_1d_torch
)


class CategoricalPolicy(nn.Module):
    """
    A simple MLP categorical policy for discrete action spaces.

    Same architecture as in ac.py and a2c.py. We keep it
    self-contained here so this module can be imported independently.

    Given state s, outputs logits for each discrete action.
    The policy is:
      pi_theta(a | s) = Categorical(logits=network(s))

    We output raw logits (not softmax probabilities) because PyTorch's
    Categorical distribution applies softmax internally, which is more stable.

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param n_actions: Number of discrete actions.
        :type n_actions: int
    :param hidden_sizes: Hidden layer sizes.
        :type hidden_sizes: tuple[int, ...]
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = int(obs_dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features=in_dim, out_features=int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        # Final layer: one logit per action, no activation
        layers.append(nn.Linear(in_features=in_dim, out_features=int(n_actions)))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param obs: Observations, shape (batch, obs_dim).
            :type obs: torch.Tensor

        :return: Logits for each action, shape (batch, n_actions).
            :rtype: torch.Tensor
        """
        return self.net(obs)


@dataclass
class ReinforceConfig:
    """
    Configuration for REINFORCE.

    :param gamma: Discount factor.
        :type gamma: float
    :param lr: Policy network learning rate.
        :type lr: float
    :param hidden_sizes: Policy network hidden layer sizes.
        :type hidden_sizes: tuple[int, ...]
    :param normalize_advantages: Whether to normalize the (baseline-subtracted) returns
        within each episode to zero mean and unit std. Slightly biased but often stabilizes
        training when returns vary a lot across episodes.
        :type normalize_advantages: bool
    :param use_running_baseline: Whether to subtract a running-mean EMA baseline from G_t.
        The baseline is a scalar that does not depend on the action, so subtracting it
        does not bias the gradient (see theory.md Section 8.2).
        :type use_running_baseline: bool
    :param baseline_momentum: EMA momentum for the running-mean baseline. Higher -> slower updates.
        :type baseline_momentum: float
    :param max_grad_norm: Gradient clipping threshold (L2 norm). Set to 0 to disable.
        :type max_grad_norm: float
    """
    gamma: float = 0.99
    lr: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)
    normalize_advantages: bool = True
    use_running_baseline: bool = True
    baseline_momentum: float = 0.9
    max_grad_norm: float = 10.0


class ReinforceAgent:
    """
    REINFORCE agent (Monte Carlo policy gradient) for discrete action spaces.

    The training loop this agent expects is:
      1) collect a full episode with act()
      2) call update_from_episode() with the collected buffers
      3) repeat

    The update uses the loss:
      L = - mean_t [ log pi(a_t | s_t) * (G_t - baseline) ]

    where G_t is the discounted reward-to-go and baseline is an optional
    running-mean scalar that reduces variance without biasing the gradient.

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param n_actions: Number of discrete actions.
        :type n_actions: int
    :param cfg: REINFORCE configuration.
        :type cfg: ReinforceConfig
    :param seed: Random seed for weight initialization.
        :type seed: int
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: ReinforceConfig, seed: int) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")

        torch.manual_seed(int(seed))

        self.policy = CategoricalPolicy(
            obs_dim=int(obs_dim),
            n_actions=int(n_actions),
            hidden_sizes=tuple(cfg.hidden_sizes)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(cfg.lr))

        # Optional EMA baseline: tracks the running average of episode returns
        # We set it to None when disabled so the update code stays clean
        self.baseline: Optional[RunningMeanBaseline] = None
        if bool(cfg.use_running_baseline):
            self.baseline = RunningMeanBaseline(momentum=float(cfg.baseline_momentum))

    @torch.no_grad()
    def act(self, *, obs: np.ndarray) -> tuple[int, float]:
        """
        Sample an action from the current policy and return its log-probability.

        We use no_grad here because act() is pure inference: gradients are
        not needed until update_from_episode() is called.

        :param obs: Current observation, shape (obs_dim,).
            :type obs: np.ndarray

        :return: (action, log_prob) where log_prob is a plain Python float.
            :rtype: tuple[int, float]
        """
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        logits = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        return int(action_t.item()), float(log_prob_t.item())

    def update_from_episode(
        self,
        *,
        obs_list: list[np.ndarray],
        action_list: list[int],
        reward_list: list[float]
    ) -> dict[str, float]:
        """
        Perform one REINFORCE gradient update from a completed episode.

        Steps:
          1) Compute Monte Carlo reward-to-go G_t for each step.
          2) Subtract the running-mean baseline (if enabled) to center the signal.
          3) Optionally normalize the centered returns (zero mean, unit std).
          4) Compute log-probs of the taken actions under the current policy.
          5) Compute loss: L = -mean( log_prob * weight ) and backprop.
          6) Update the baseline with the episode return (after using it).

        Note on step 6: we update the baseline AFTER using it for the gradient.
        This is not an error, the baseline used for the update is from the
        previous episode, which is fine because baseline updates don't affect
        the unbiasedness of the gradient.

        :param obs_list: Observations collected during the episode.
            :type obs_list: list[np.ndarray]
        :param action_list: Actions taken at each step.
            :type action_list: list[int]
        :param reward_list: Rewards received at each step.
            :type reward_list: list[float]

        :return: Dict with "policy_loss" and "episode_return".
            :rtype: dict[str, float]
        """
        if not (len(obs_list) == len(action_list) == len(reward_list)):
            raise ValueError("Episode buffers must all have the same length.")
        if len(reward_list) == 0:
            return {"policy_loss": 0.0, "episode_return": 0.0}

        # 1) Monte Carlo reward-to-go: G_t = r_{t+1} + gamma * r_{t+2} + ...
        # computed by backward recursion in utils.py
        returns_np = compute_discounted_returns(rewards=reward_list, gamma=float(self.cfg.gamma))
        episode_return = float(np.sum(np.asarray(reward_list, dtype=np.float64)))

        # 2) Build tensors
        obs_t = torch.as_tensor(np.asarray(obs_list, dtype=np.float32), device=self.device)
        actions_t = torch.as_tensor(np.asarray(action_list, dtype=np.int64), device=self.device)
        returns_t = torch.as_tensor(returns_np, device=self.device, dtype=torch.float32)

        # 3) Subtract baseline to center the learning signal
        # The baseline is a scalar that does not depend on the action, so it
        # does not bias the gradient (see theory.md Section 8.2)
        if self.baseline is not None:
            b = float(self.baseline.value())
            returns_t = returns_t - b

        # 4) Optional normalization: standardize to zero mean / unit std within the episode
        # This makes the update step size consistent across episodes with different return scales
        if bool(self.cfg.normalize_advantages) and returns_t.numel() > 1:
            returns_t = normalize_1d_torch(x=returns_t)

        # 5) Log-prob of the taken actions under the current policy
        # This forward pass needs gradients, so it must NOT be inside no_grad
        logits = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs_t = dist.log_prob(actions_t)  # shape (T,)

        # 6) Policy gradient loss (negative because we minimize, but want to maximize return)
        #    L = - E[ log pi(a|s) * weight ]
        #    Positive weight -> loss goes down when log-prob goes up -> action becomes more likely
        policy_loss = -(log_probs_t * returns_t).mean()

        # 7) Gradient step with optional clipping
        self.optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        if float(self.cfg.max_grad_norm) > 0.0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=float(self.cfg.max_grad_norm))
        self.optimizer.step()

        # 8) Update baseline AFTER the gradient step (does not affect this update's bias)
        if self.baseline is not None:
            self.baseline.update(episode_return=episode_return)

        return {
            "policy_loss": float(policy_loss.item()),
            "episode_return": float(episode_return)
        }
