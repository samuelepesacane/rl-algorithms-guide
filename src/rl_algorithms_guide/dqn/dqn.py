from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DQNConfig:
    """
    Configuration for DQN-style agents.

    :param gamma: Discount factor.
        :type gamma: float
    :param lr: Learning rate for Adam.
        :type lr: float
    :param batch_size: Minibatch size sampled from the replay buffer.
        :type batch_size: int
    :param buffer_size: Maximum replay buffer capacity.
        :type buffer_size: int
    :param learning_starts: Number of environment steps before starting gradient updates.
        :type learning_starts: int
    :param train_freq: Perform one gradient update every this many environment steps.
        :type train_freq: int
    :param target_update_interval: Copy online network -> target network every this many env steps.
        :type target_update_interval: int
    :param max_grad_norm: Gradient clipping (L2 norm). Use 0 to disable.
        :type max_grad_norm: float
    :param double_dqn: If True, use Double DQN target (decouple argmax selection and evaluation).
        :type double_dqn: bool
    :param dueling: If True, use Dueling network architecture.
        :type dueling: bool
    :param hidden_sizes: Hidden layer sizes for the MLP.
        :type hidden_sizes: Tuple[int, ...]
    """

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_freq: int = 1
    target_update_interval: int = 1_000
    max_grad_norm: float = 10.0
    double_dqn: bool = False
    dueling: bool = False
    hidden_sizes: tuple[int, ...] = (128, 128)


class ReplayBuffer:
    """
    A simple replay buffer for off-policy RL.
    Stores transitions (s, a, r, s', done).

    We store transitions in fixed-size numpy arrays for speed and simplicity.
    Later we sample random minibatches to break correlation between consecutive steps.

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param capacity: Maximum number of transitions.
        :type capacity: int
    :param seed: Random seed for sampling.
        :type seed: int
    """

    def __init__(self, obs_dim: int, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self._rng = np.random.default_rng(int(seed))
        self._pos = 0  # position in which we need to store the transition
        self._size = 0  # current size of buffer (how many transitions are in the buffer)

        self.obs = np.zeros(shape=(self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros(shape=(self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(shape=(self.capacity,), dtype=np.int64)
        self.rewards = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.dones = np.zeros(shape=(self.capacity,), dtype=np.float32)

    def __len__(self) -> int:
        """
        Current number of stored transitions.

        :return: Buffer size.
            :rtype: int
        """
        return self._size

    def add(
        self,
        *,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add one transition to the buffer.
        We overwrite old data once capacity is full (ring buffer).
        That gives a moving window of experience, which is usually what you want in DQN.

        :param obs: Observation at time t, shape (obs_dim,).
            :type obs: np.ndarray
        :param action: Action taken at time t.
            :type action: int
        :param reward: Reward observed after taking the action.
            :type reward: float
        :param next_obs: Observation at time t+1, shape (obs_dim,).
            :type next_obs: np.ndarray
        :param done: Whether the episode ended at t+1.
            :type done: bool

        :return: None.
            :rtype: None
        """
        idx = self._pos

        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if bool(done) else 0.0

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random minibatch.

        Random sampling is the whole point of replay.
        It decorrelates data and makes SGD behave more like supervised learning.

        :param batch_size: Number of transitions to sample.
            :type batch_size: int
        :param device: Torch device where tensors will be created.
            :type device: torch.device

        :return: (obs, actions, rewards, next_obs, dones) tensors.
            :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        if self._size < int(batch_size):
            raise ValueError("Not enough samples in replay buffer to draw a batch.")

        # Sample indices uniformly from the filled portion of the buffer
        idxs = self._rng.integers(low=0, high=self._size, size=int(batch_size), endpoint=False)

        # Convert to torch tensors on the desired device
        obs = torch.as_tensor(self.obs[idxs], device=device)
        actions = torch.as_tensor(self.actions[idxs], device=device)
        rewards = torch.as_tensor(self.rewards[idxs], device=device)
        next_obs = torch.as_tensor(self.next_obs[idxs], device=device)
        dones = torch.as_tensor(self.dones[idxs], device=device)

        return obs, actions, rewards, next_obs, dones


class MLPQNetwork(nn.Module):
    """
    A plain MLP that outputs Q-values for all actions.

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param n_actions: Number of discrete actions.
        :type n_actions: int
    :param hidden_sizes: Hidden layer sizes.
        :type hidden_sizes: Tuple[int, ...]
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()

        layers: list[nn.Module] = []  # list of layers
        in_dim = int(obs_dim)  # input dimension

        # Build a simple MLP trunk: Linear + ReLU repeated (deeper network)
        # deeper networks can represent more complex value surfaces than a single layer
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)

        # Final linear layer maps to one Q-value per action
        layers.append(nn.Linear(in_dim, int(n_actions)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Batch of observations, shape (batch, obs_dim).
            :type x: torch.Tensor

        :return: Q-values for all actions, shape (batch, n_actions).
            :rtype: torch.Tensor
        """
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling network: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    We learn:
      - V(s): state value (how good is the state overall)
      - A(s,a): advantage (how much action a differs from the average action in s)
    Then combine them.

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param n_actions: Number of discrete actions.
        :type n_actions: int
    :param hidden_sizes: Hidden layer sizes for the shared trunk.
        :type hidden_sizes: Tuple[int, ...]
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()

        trunk: list[nn.Module] = []
        in_dim = int(obs_dim)

        # Shared feature extractor for both heads
        # V and A should reuse the same state representation
        for h in hidden_sizes:
            trunk.append(nn.Linear(in_dim, int(h)))
            trunk.append(nn.ReLU())
            in_dim = int(h)

        self.trunk = nn.Sequential(*trunk)
        # Two heads: scalar value and per-action advantages
        self.value_head = nn.Linear(in_dim, 1)  # V
        self.adv_head = nn.Linear(in_dim, int(n_actions))  # A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Batch of observations, shape (batch, obs_dim).
            :type x: torch.Tensor

        :return: Q-values for all actions, shape (batch, n_actions).
            :rtype: torch.Tensor
        """
        z = self.trunk(x)
        # V(s): one value per state in the batch
        v = self.value_head(z)  # (batch, 1)
        # A(s,a): one value per (state, action)
        a = self.adv_head(z)    # (batch, n_actions)
        # Center advantages so they sum/average to 0 across actions
        # this pins down the decomposition and makes V(s) the average Q-value in the state
        a_centered = a - a.mean(dim=1, keepdim=True)
        return v + a_centered


class DQNAgent:
    """
    DQN-style agent (supports Double DQN and Dueling networks).

    :param obs_dim: Observation dimension.
        :type obs_dim: int
    :param n_actions: Number of discrete actions.
        :type n_actions: int
    :param cfg: DQN configuration.
        :type cfg: DQNConfig
    :param seed: Random seed.
        :type seed: int
    :param device: Torch device. If None, picks CUDA if available else CPU.
        :type device: torch.device | None
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: DQNConfig,
        seed: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.cfg = cfg

        # Default to GPU if available, otherwise CPU
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Seed torch + numpy for reproducible networks and exploration
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        # Choose network architecture
        # Remember: Dueling changes representation, Double DQN changes the target computation
        if cfg.dueling:
            self.q_net: nn.Module = DuelingQNetwork(obs_dim=self.obs_dim,
                                                    n_actions=self.n_actions,
                                                    hidden_sizes=cfg.hidden_sizes
                                                    )
            self.target_net: nn.Module = DuelingQNetwork(obs_dim=self.obs_dim,
                                                         n_actions=self.n_actions,
                                                         hidden_sizes=cfg.hidden_sizes
                                                         )
        else:
            self.q_net = MLPQNetwork(obs_dim=self.obs_dim,
                                     n_actions=self.n_actions,
                                     hidden_sizes=cfg.hidden_sizes
                                     )
            self.target_net = MLPQNetwork(obs_dim=self.obs_dim,
                                          n_actions=self.n_actions,
                                          hidden_sizes=cfg.hidden_sizes
                                          )

        self.q_net.to(self.device)
        self.target_net.to(self.device)

        # Target network starts as an exact copy
        # targets should be computed from a slowly-moving network, not the one we update every step
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # makes it clear we never intend to train it directly

        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=float(cfg.lr))
        # Replay buffer is seeded separately so sampling is reproducible too
        self.replay = ReplayBuffer(obs_dim=self.obs_dim, capacity=int(cfg.buffer_size), seed=int(seed) + 123)
        self._updates = 0

    @property
    def num_updates(self) -> int:
        """
        Number of gradient updates performed so far.

        :return: Update count.
            :rtype: int
        """
        return self._updates

    def select_action(self, *, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        """
        Epsilon-greedy action selection.
        Early training needs exploration to populate replay with diverse transitions.
        Later training benefits from exploiting the learned Q-function.

        :param obs: Current observation, shape (obs_dim,).
            :type obs: np.ndarray
        :param epsilon: Exploration probability in [0,1].
            :type epsilon: float
        :param rng: NumPy RNG used for exploration decisions.
            :type rng: np.random.Generator

        :return: Action index in [0, n_actions-1].
            :rtype: int
        """
        eps = float(np.clip(epsilon, 0.0, 1.0))

        # Exploration branch: take a random discrete action
        if rng.random() < eps:
            return int(rng.integers(low=0, high=self.n_actions))

        # Exploitation branch: pick argmax_a Q(s,a)
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
            q = self.q_net(x)
            return int(torch.argmax(q, dim=1).item())

    def store(
        self,
        *,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store one transition in replay.

        :param obs: Observation at time t.
            :type obs: np.ndarray
        :param action: Action at time t.
            :type action: int
        :param reward: Reward at time t+1.
            :type reward: float
        :param next_obs: Observation at time t+1.
            :type next_obs: np.ndarray
        :param done: Episode ended at t+1.
            :type done: bool

        :return: None.
            :rtype: None
        """
        self.replay.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def train_step(self) -> Optional[float]:
        """
        Perform one gradient update using a minibatch from replay.

        The core supervised-style regression is:
          Q_theta(s,a)  <-  r + gamma * bootstrap(next_state)

        where bootstrap() depends on DQN vs Double DQN.

        - DQN target: r + gamma * max_a Q_target(s',a)
        - Double DQN target: r + gamma * Q_target(s', argmax_a Q_online(s',a))

        :return: The scalar loss value, or None if not enough data.
            :rtype: float | None
        """
        if len(self.replay) < self.cfg.batch_size:
            return None

        # Sample a shuffled minibatch
        # Replay makes training more stable by breaking temporal correlations
        obs, actions, rewards, next_obs, dones = self.replay.sample(
            batch_size=self.cfg.batch_size, device=self.device
        )

        # Compute Q_theta(s, ·) for the batch
        # then pick the values for the actions that were actually taken
        # the network outputs all actions, but each transition only supervises Q(s,a_taken)
        q_all = self.q_net(obs)  # (B, A) returns all Q-values for all actions for each state in the batch
        q_sa = q_all.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)  # (B,) each replay transition contains only the action that was actually taken -> actions[i]
        # gather is just pick, for each row, the column indicated by an index

        with torch.no_grad():
            # Compute bootstrap term from next state
            # the target should be treated as a fixed label (no backprop through target computation)
            if self.cfg.double_dqn:
                # Double DQN:
                # select best action using ONLINE net (reduces max overestimation bias)
                # evaluate that action using TARGET net (keeps targets stable)
                next_actions = torch.argmax(self.q_net(next_obs), dim=1)  # (B,)
                next_q = self.target_net(next_obs).gather(dim=1, index=next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Vanilla DQN:
                # evaluate best next action by taking max over TARGET net outputs
                next_q = torch.max(self.target_net(next_obs), dim=1).values  # (B,)

            # Terminal masking: if done=1 target should be just reward (no bootstrap)
            target = rewards + (1.0 - dones) * float(self.cfg.gamma) * next_q

        # Huber loss is the standard robust TD regression loss in DQN
        loss = F.smooth_l1_loss(q_sa, target)

        # Standard optimizer step: clear grads, backprop, clip (this is optional), update
        self.optim.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping prevents rare huge TD errors from exploding updates
        if float(self.cfg.max_grad_norm) > 0.0:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=float(self.cfg.max_grad_norm))

        self.optim.step()
        self._updates += 1

        return float(loss.item())

    def maybe_update_target(self, *, env_step: int) -> bool:
        """
        Update target network if it's time.
        Copy online network -> target network periodically.
        Targets must move slowly, otherwise you chase a moving label and training becomes unstable.

        Note:
        - We already copy online -> target once at initialization.
        - Here we update again every `target_update_interval` environment steps.
        - We don't update at env_step=0 to avoid a redundant copy.

        :param env_step: Current environment step (0-based).
            :type env_step: int

        :return: True if target network was updated.
            :rtype: bool
        """
        interval = int(self.cfg.target_update_interval)
        if interval <= 0:
            return False

        # Avoid a redundant copy at step 0 (we copied at init already)
        if int(env_step) <= 0 or (int(env_step) % interval) != 0:
            return False

        self.target_net.load_state_dict(self.q_net.state_dict())
        return True


def linear_epsilon(
    *,
    step: int,
    start_e: float,
    end_e: float,
    duration: int,
) -> float:
    """
    Linear epsilon schedule.
    Start with lots of exploration to fill replay,
    then gradually exploit more as Q improves.

    :param step: Current environment step.
        :type step: int
    :param start_e: Initial epsilon.
        :type start_e: float
    :param end_e: Final epsilon.
        :type end_e: float
    :param duration: Number of steps over which to anneal.
        :type duration: int

    :return: Epsilon value for this step.
        :rtype: float
    """
    d = max(int(duration), 1)
    t = min(max(int(step), 0), d)
    frac = t / d
    return float(start_e + frac * (end_e - start_e))
