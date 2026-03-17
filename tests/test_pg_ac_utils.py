import numpy as np
import torch

from rl_algorithms_guide.pg_ac.reinforce import CategoricalPolicy, ReinforceAgent, ReinforceConfig
from rl_algorithms_guide.common.utils import compute_discounted_returns, normalize_1d_torch


def test_compute_discounted_returns_known_values() -> None:
    """
    Check discounted returns on a tiny hand-computable example.
    """
    rewards = [1.0, 1.0, 1.0]
    gamma = 0.9
    rets = compute_discounted_returns(rewards=rewards, gamma=gamma)

    # G0 = 1 + 0.9 + 0.81 = 2.71
    # G1 = 1 + 0.9 = 1.9
    # G2 = 1
    assert rets.shape == (3,)
    assert np.allclose(rets, np.asarray([2.71, 1.9, 1.0], dtype=np.float32), atol=1e-5)


def test_normalize_1d_torch_shape_and_finite() -> None:
    """
    Normalization should keep shape and not produce NaNs/infs.
    """
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    y = normalize_1d_torch(x=x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all().item() is True


def test_categorical_policy_logprob_finite() -> None:
    """
    Sampling/log-prob from categorical policy should be finite.
    """
    policy = CategoricalPolicy(obs_dim=4, n_actions=2, hidden_sizes=(16,))
    obs = torch.zeros((1, 4), dtype=torch.float32)
    logits = policy(obs)
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    lp = dist.log_prob(a)

    assert lp.shape == (1,)
    assert torch.isfinite(lp).all().item() is True


def test_reinforce_update_no_nans() -> None:
    """
    One REINFORCE update on a fake episode should not produce NaNs.
    """
    cfg = ReinforceConfig(gamma=0.99, lr=1e-3, hidden_sizes=(32,), normalize_advantages=True, use_running_baseline=True)
    agent = ReinforceAgent(obs_dim=4, n_actions=2, cfg=cfg, seed=0)

    obs_list = [np.zeros(4, dtype=np.float32) for _ in range(5)]
    action_list = [0, 1, 0, 1, 0]
    reward_list = [1.0, 1.0, 1.0, 1.0, 1.0]

    metrics = agent.update_from_episode(obs_list=obs_list, action_list=action_list, reward_list=reward_list)

    assert "policy_loss" in metrics
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["episode_return"])
