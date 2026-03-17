"""
Single-step Actor-Critic for discrete action spaces.

This module implements the minimal on-policy Actor-Critic described in
docs/04_pg_ac/theory.md (Sections 11-11.4).

The idea is simple:
- the actor is the same policy as in REINFORCE
- instead of waiting for the episode to end and using Monte Carlo returns,
  we add a critic: a value network V(s) that gives us a cheap, online
  advantage estimate after every single step

One step of interaction produces:
  delta_t = r + gamma * V(s') * (1 - done) - V(s)

delta_t is the TD error, and it doubles as our advantage estimate:
  - if delta_t > 0, the step went better than the critic expected -> reinforce the action
  - if delta_t < 0, the step went worse than expected             -> discourage the action

This is lower-variance than Monte Carlo returns (we don't wait for the
whole future), but it introduces bias if V is inaccurate early in training.
That bias/variance tradeoff is usually worth it in practice.

The actor and critic are trained with separate optimizers, which gives
independent control over their learning rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class CategoricalActor(nn.Module):
    """
    MLP categorical actor: maps observations to action logits.

    Same architecture as in a2c.py and reinforce.py. We keep it
    self-contained here so this module can be imported independently.

    Given state s, the policy is:
      pi_theta(a | s) = Categorical(logits=network(s))

    We output raw logits (not probabilities) because PyTorch's
    Categorical distribution handles the softmax internally, which is
    more numerically stable than computing softmax ourselves.

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param n_actions: Number of discrete actions.
        :type n_actions: int
    :param hidden_sizes: Hidden layer sizes for the MLP.
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
        # Final layer outputs one logit per action (no activation)
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


class ValueCritic(nn.Module):
    """
    MLP value function V(s): maps observations to a scalar state value.

    The critic predicts:
      V(s) ≈ expected discounted return if we follow the current policy from s

    This is used to compute the TD error (our advantage estimate):
      delta_t = r + gamma * V(s') - V(s)

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param hidden_sizes: Hidden layer sizes for the MLP.
        :type hidden_sizes: tuple[int, ...]
    """

    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = int(obs_dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features=in_dim, out_features=int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        # Output is a single scalar V(s), so the final layer has 1 output
        layers.append(nn.Linear(in_features=in_dim, out_features=1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param obs: Observations, shape (batch, obs_dim).
            :type obs: torch.Tensor

        :return: State values, shape (batch,).
            :rtype: torch.Tensor
        """
        v = self.net(obs)
        # squeeze(-1) removes the trailing size-1 dimension: (batch, 1) -> (batch,)
        return v.squeeze(-1)


@dataclass
class ActorCriticConfig:
    """
    Configuration for the single-step Actor-Critic agent.

    The update rule is:
      delta_t = r + gamma * V(s') * (1 - done) - V(s)   [TD error = advantage estimate]
      L_actor  = - log pi(a|s) * stop_grad(delta_t)      [weighted log-likelihood]
      L_critic = delta_t^2                               [TD regression]

    Entropy bonus (optional):
      L_total_actor = L_actor - entropy_coef * H[pi(.|s)]
    Adding entropy to the actor loss discourages the policy from collapsing
    to a single deterministic action too early, which helps exploration.

    :param gamma: Discount factor.
        :type gamma: float
    :param lr_actor: Learning rate for the actor optimizer.
        :type lr_actor: float
    :param lr_critic: Learning rate for the critic optimizer.
        :type lr_critic: float
    :param hidden_sizes: Hidden layer sizes shared by actor and critic.
        :type hidden_sizes: tuple[int, ...]
    :param value_coef: Scaling factor for the critic loss.
        :type value_coef: float
    :param entropy_coef: Weight for the entropy bonus. Set to 0 to disable.
        :type entropy_coef: float
    :param max_grad_norm: Gradient clipping threshold (L2 norm). Set to 0 to disable.
        :type max_grad_norm: float
    """
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 10.0


class ActorCriticAgent:
    """
    Single-step on-policy Actor-Critic for discrete action spaces.

    This is the simplest Actor-Critic: one agent, one environment,
    one gradient update per environment step.

    The actor and critic share the same hidden-size configuration
    but have separate network weights and separate optimizers.
    This lets you tune their learning rates independently, which
    is useful because the critic often needs to learn faster than the actor.

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param n_actions: Number of discrete actions.
        :type n_actions: int
    :param cfg: Agent configuration.
        :type cfg: ActorCriticConfig
    :param seed: Random seed for weight initialization.
        :type seed: int
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: ActorCriticConfig, seed: int) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")

        torch.manual_seed(int(seed))

        self.actor = CategoricalActor(
            obs_dim=int(obs_dim),
            n_actions=int(n_actions),
            hidden_sizes=tuple(cfg.hidden_sizes)
        ).to(self.device)

        self.critic = ValueCritic(
            obs_dim=int(obs_dim),
            hidden_sizes=tuple(cfg.hidden_sizes)
        ).to(self.device)

        # Separate optimizers give independent lr control for actor and critic
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.lr_actor))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.lr_critic))

    @torch.no_grad()
    def act(self, *, obs: np.ndarray) -> tuple[int, float]:
        """
        Sample an action from the current policy and return its log-probability.

        We use no_grad here because act() is pure inference: we don't need
        gradients until the update step.

        :param obs: Current observation, shape (obs_dim,).
            :type obs: np.ndarray

        :return: (action, log_prob) where log_prob is a plain Python float.
            :rtype: tuple[int, float]
        """
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        logits = self.actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        return int(action_t.item()), float(log_prob_t.item())

    def update_step(
        self,
        *,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> dict[str, float]:
        """
        Perform one Actor-Critic update from a single (s, a, r, s', done) transition.

        The update has three parts:

        1) Compute the TD error (our advantage estimate):
             delta_t = r + gamma * V(s') * (1 - done) - V(s)
           The (1 - done) mask prevents bootstrapping past terminal states.
           We use stop_grad on delta_t when computing the actor loss, because
           the actor should not try to optimize through the critic's prediction.

        2) Actor loss (weighted log-likelihood):
             L_actor = - log pi(a|s) * stop_grad(delta_t)
           If delta_t > 0, this step went better than expected -> push action up.
           If delta_t < 0, this step went worse than expected  -> push action down.

        3) Critic loss (TD regression):
             L_critic = delta_t^2
           Train V(s) to predict the TD target r + gamma * V(s').

        :param obs: Current observation.
            :type obs: np.ndarray
        :param action: Action taken.
            :type action: int
        :param reward: Reward received.
            :type reward: float
        :param next_obs: Next observation.
            :type next_obs: np.ndarray
        :param done: Whether the episode ended after this step.
            :type done: bool

        :return: Dict with "policy_loss", "value_loss", "td_error", "entropy".
            :rtype: dict[str, float]
        """
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        no = torch.as_tensor(np.asarray(next_obs, dtype=np.float32), device=self.device).unsqueeze(0)

        r = torch.as_tensor([float(reward)], device=self.device, dtype=torch.float32)
        # done mask: 1.0 if episode ended, 0.0 otherwise
        # multiplying (1 - d) by V(s') zeroes out the bootstrap term at terminal states
        d = torch.as_tensor([1.0 if bool(done) else 0.0], device=self.device, dtype=torch.float32)

        # Critic forward pass: V(s)
        v = self.critic(o)  # shape (1,)

        # V(s') is only used as a bootstrap target, so we don't need gradients through it
        with torch.no_grad():
            v_next = self.critic(no)  # shape (1,)

        # TD target: what the critic should have predicted for V(s)
        target = r + float(self.cfg.gamma) * (1.0 - d) * v_next

        # TD error = advantage estimate: positive means "better than expected"
        td_error = target - v  # shape (1,)

        # Actor: recompute log-prob of the taken action under the current policy
        # (we need gradients here, so this cannot be inside no_grad)
        logits = self.actor(o)
        dist = torch.distributions.Categorical(logits=logits)
        a_t = torch.as_tensor([int(action)], device=self.device, dtype=torch.int64)
        log_prob = dist.log_prob(a_t)  # shape (1,)
        entropy = dist.entropy()       # shape (1,), higher = more exploratory policy

        # Actor loss: negative because we want to maximize, but optimizers minimize
        # td_error.detach() is critical: the actor should not update to make the critic's
        # prediction look better, only to make good actions more likely
        policy_loss = -(log_prob * td_error.detach()).mean()

        # Critic loss: minimize squared TD error (train V toward the bootstrap target)
        value_loss = (td_error ** 2).mean()

        # Optional entropy bonus: subtract it from actor loss to discourage early collapse
        entropy_bonus = entropy.mean()
        total_actor_loss = policy_loss - float(self.cfg.entropy_coef) * entropy_bonus
        total_critic_loss = float(self.cfg.value_coef) * value_loss

        # Update actor
        self.opt_actor.zero_grad(set_to_none=True)
        total_actor_loss.backward()
        if float(self.cfg.max_grad_norm) > 0.0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=float(self.cfg.max_grad_norm))
        self.opt_actor.step()

        # Update critic (separate backward pass, separate optimizer)
        self.opt_critic.zero_grad(set_to_none=True)
        total_critic_loss.backward()
        if float(self.cfg.max_grad_norm) > 0.0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=float(self.cfg.max_grad_norm))
        self.opt_critic.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "td_error": float(td_error.mean().item()),
            "entropy": float(entropy_bonus.item())
        }
