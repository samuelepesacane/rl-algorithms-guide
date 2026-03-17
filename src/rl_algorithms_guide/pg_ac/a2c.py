"""
A2C (Advantage Actor-Critic) for discrete action spaces.

This module implements A2C as described in docs/04_pg_ac/theory.md
(Sections 16-16.7).

A2C is a natural extension of single-step Actor-Critic (actor_critic.py):
the algorithm is identical, but instead of one environment, we run N
environments in parallel and average the gradients across all of them.

Why does this help?
- Consecutive transitions from one environment are highly correlated
  (s_t and s_{t+1} look very similar).
- Running N independent environments gives us a batch of transitions from
  N different trajectory fragments, which are much less correlated.
- Averaging gradients over a less-correlated batch reduces variance.

Rather than spawning real OS processes, we use Gymnasium's SyncVectorEnv,
which runs N environments in the same Python process and stacks their
observations, rewards, and dones into batched arrays.

Structure of one A2C update:
  1) All N environments run for `rollout_steps` steps in sync.
  2) For each step t and worker i, compute the n-step return:
       G_t^(i) = r_{t+1}^(i) + gamma*r_{t+2}^(i) + ... + gamma^n * V(s_n^(i))
  3) Advantage estimate: A_t^(i) = G_t^(i) - V(s_t^(i))
  4) Average actor and critic losses across all N*n transitions.
  5) One synchronised gradient update.
  6) All workers continue from where they left off.

If you set num_envs=1 and rollout_steps=1, this collapses to single-step
Actor-Critic, which is a useful sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class CategoricalActor(nn.Module):
    """
    MLP categorical actor: maps observations to action logits.

    Same architecture as in actor_critic.py and reinforce.py.
    We keep it self-contained here so this module can be imported independently.

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
        # Output one logit per action (no activation; Categorical handles softmax)
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

    Used to compute the advantage estimate for each step:
      A_t = G_t^(n) - V(s_t)

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param hidden_sizes: Hidden layer sizes.
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
        # Scalar output: one value per state
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
        return self.net(obs).squeeze(-1)


@dataclass
class A2CConfig:
    """
    Configuration for the A2C agent.

    Losses:
      L_actor  = - (1 / N*n) sum_{i,t} log pi(a_t^i | s_t^i) * stop_grad(A_t^i)
      L_critic = (1 / N*n) sum_{i,t} (V(s_t^i) - G_t^(n,i))^2
      L_entropy = - (1 / N*n) sum_{i,t} H[pi(.|s_t^i)]  [optional]

      L_total = L_actor + value_coef * L_critic - entropy_coef * L_entropy

    :param gamma: Discount factor.
        :type gamma: float
    :param lr: Shared learning rate for both actor and critic.
        :type lr: float
    :param hidden_sizes: Hidden layer sizes shared by actor and critic.
        :type hidden_sizes: tuple[int, ...]
    :param value_coef: Scaling factor for the critic loss term.
        :type value_coef: float
    :param entropy_coef: Weight for the entropy bonus. Set to 0 to disable.
        Positive entropy_coef encourages the policy to stay exploratory.
        :type entropy_coef: float
    :param max_grad_norm: Gradient clipping threshold (L2 norm). Set to 0 to disable.
        :type max_grad_norm: float
    :param normalize_advantages: Whether to normalize advantages to zero mean / unit std
        within each rollout batch. Slightly biased but often stabilizes training.
        :type normalize_advantages: bool
    """
    gamma: float = 0.99
    lr: float = 7e-4
    hidden_sizes: tuple[int, ...] = (64, 64)
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = False


class A2CAgent:
    """
    A2C agent for discrete action spaces.

    This agent operates on batched observations from N environments.
    The actor and critic share a single optimizer, which is the standard
    A2C setup (one combined loss, one backward pass per update).

    :param obs_dim: Observation dimension per environment.
        :type obs_dim: int
    :param n_actions: Number of discrete actions.
        :type n_actions: int
    :param cfg: A2C configuration.
        :type cfg: A2CConfig
    :param seed: Random seed for weight initialization.
        :type seed: int
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: A2CConfig, seed: int) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")

        torch.manual_seed(int(seed))

        self.actor = CategoricalActor(
            obs_dim=int(obs_dim),
            n_actions=int(n_actions),
            hidden_sizes=tuple(cfg.hidden_sizes),
        ).to(self.device)

        self.critic = ValueCritic(
            obs_dim=int(obs_dim),
            hidden_sizes=tuple(cfg.hidden_sizes),
        ).to(self.device)

        # A single optimizer covers both actor and critic parameters.
        # This means one backward pass computes all gradients at once,
        # which is more efficient and the standard practice in A2C.
        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=float(cfg.lr))

    @torch.no_grad()
    def act_batch(self, *, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample actions for a batch of observations (one per environment).

        :param obs: Batched observations, shape (N, obs_dim).
            :type obs: np.ndarray

        :return: (actions, log_probs) both shape (N,).
            :rtype: tuple[np.ndarray, np.ndarray]
        """
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        logits = self.actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions_t = dist.sample()
        log_probs_t = dist.log_prob(actions_t)
        return actions_t.cpu().numpy(), log_probs_t.cpu().numpy()

    @torch.no_grad()
    def value_batch(self, *, obs: np.ndarray) -> np.ndarray:
        """
        Compute V(s) for a batch of observations.

        :param obs: Batched observations, shape (N, obs_dim).
            :type obs: np.ndarray

        :return: Values, shape (N,).
            :rtype: np.ndarray
        """
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.critic(obs_t).cpu().numpy()

    def update(
        self,
        *,
        obs_buf: np.ndarray,
        act_buf: np.ndarray,
        rew_buf: np.ndarray,
        done_buf: np.ndarray,
        last_obs: np.ndarray,
        last_dones: np.ndarray
    ) -> dict[str, float]:
        """
        Perform one A2C update from a rollout buffer.

        The buffer contains transitions from N environments over n steps.
        Layout: obs_buf[t, i] is the observation at step t in environment i.

        Steps:
          1) Bootstrap from the last observation to get n-step returns G_t^(i).
          2) Compute advantages: A_t^(i) = G_t^(i) - V(s_t^(i)).
          3) Compute actor loss, critic loss, and entropy bonus.
          4) One gradient update on the combined loss.

        :param obs_buf: Observations, shape (n, N, obs_dim).
            :type obs_buf: np.ndarray
        :param act_buf: Actions taken, shape (n, N).
            :type act_buf: np.ndarray
        :param rew_buf: Rewards received, shape (n, N).
            :type rew_buf: np.ndarray
        :param done_buf: Done flags (1.0 if episode ended), shape (n, N).
            :type done_buf: np.ndarray
        :param last_obs: Observations after the last step, shape (N, obs_dim).
            :type last_obs: np.ndarray
        :param last_dones: Done flags after the last step, shape (N,).
            :type last_dones: np.ndarray

        :return: Dict with "policy_loss", "value_loss", "entropy", "total_loss".
            :rtype: dict[str, float]
        """
        n_steps, n_envs = rew_buf.shape
        gamma = float(self.cfg.gamma)

        # 1) Bootstrap: compute V(s_last) for each environment
        # If the last step ended an episode, there is no future value to bootstrap from
        with torch.no_grad():
            last_obs_t = torch.as_tensor(np.asarray(last_obs, dtype=np.float32), device=self.device)
            last_values = self.critic(last_obs_t).cpu().numpy()  # shape (N,)

        # 2) Compute n-step returns by backward recursion
        # G_T = V(s_last) * (1 - done_last)   [bootstrap or 0 if terminal]
        # G_t = r_{t+1} + gamma * G_{t+1} * (1 - done_t)
        returns = np.zeros_like(rew_buf, dtype=np.float32)  # shape (n, N)
        G = last_values * (1.0 - last_dones.astype(np.float32))  # shape (N,)

        for t in reversed(range(n_steps)):
            G = rew_buf[t] + gamma * G * (1.0 - done_buf[t])
            returns[t] = G

        # 3) Flatten everything: (n, N, ...) -> (n*N, ...)
        # We treat all N*n transitions as a single flat batch for the loss computation
        obs_flat = obs_buf.reshape(n_steps * n_envs, -1).astype(np.float32)
        act_flat = act_buf.reshape(n_steps * n_envs).astype(np.int64)
        ret_flat = returns.reshape(n_steps * n_envs).astype(np.float32)

        obs_t = torch.as_tensor(obs_flat, device=self.device)
        act_t = torch.as_tensor(act_flat, device=self.device)
        ret_t = torch.as_tensor(ret_flat, device=self.device)

        # 4) Critic: predict V(s) for all transitions
        values_t = self.critic(obs_t)  # shape (n*N,)

        # 5) Advantage: A_t = G_t - V(s_t)
        # We detach returns because they are targets computed outside the graph
        advantages = ret_t.detach() - values_t.detach()

        # Optional normalization: center and scale advantages within the batch
        # This keeps the actor update step size consistent across rollouts
        if bool(self.cfg.normalize_advantages) and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 6) Actor: log-prob of the taken actions under the current policy
        logits = self.actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(act_t)  # shape (n*N,)
        entropy = dist.entropy()           # shape (n*N,), higher = more exploratory

        # 7) Actor loss: weighted negative log-likelihood
        # advantages.detach() prevents the actor from trying to adjust the critic's prediction
        policy_loss = -(log_probs * advantages.detach()).mean()

        # 8) Critic loss: mean squared error against the n-step returns
        value_loss = ((values_t - ret_t.detach()) ** 2).mean()

        # 9) Combined loss (one backward pass for everything)
        # We subtract the entropy term because we want to MAXIMISE entropy (keep policy spread out),
        # but the optimizer minimises the loss
        entropy_mean = entropy.mean()
        total_loss = (
            policy_loss
            + float(self.cfg.value_coef) * value_loss
            - float(self.cfg.entropy_coef) * entropy_mean
        )

        # 10) Gradient update
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if float(self.cfg.max_grad_norm) > 0.0:
            # Clip across both actor and critic parameters together
            all_params = list(self.actor.parameters()) + list(self.critic.parameters())
            nn.utils.clip_grad_norm_(all_params, max_norm=float(self.cfg.max_grad_norm))
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_mean.item()),
            "total_loss": float(total_loss.item())
        }
