from __future__ import annotations

import torch
from rl_algorithms_guide.dqn.dqn import DuelingQNetwork


def test_dueling_centering_mean_advantage_is_zero() -> None:
    """
    Dueling DQN uses: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    Therefore, the centered advantage has zero mean across actions for each sample:
        mean_a (A - mean(A)) = 0
    """
    torch.manual_seed(0)
    net = DuelingQNetwork(obs_dim=4, n_actions=3, hidden_sizes=(16,))

    x = torch.randn(5, 4)
    z = net.trunk(x)
    a = net.adv_head(z)
    a_centered = a - a.mean(dim=1, keepdim=True)

    mean_adv = a_centered.mean(dim=1)
    torch.testing.assert_close(mean_adv, torch.zeros_like(mean_adv), atol=1e-6, rtol=0)  # returns None when it passes


def test_dueling_mean_q_equals_v() -> None:
    """
    Averaging Q over actions cancels the centered advantage:
        mean_a Q(s,a) = V(s)
    """
    torch.manual_seed(0)
    net = DuelingQNetwork(obs_dim=4, n_actions=3, hidden_sizes=(16,))

    x = torch.randn(5, 4)
    z = net.trunk(x)
    v = net.value_head(z)  # (B, 1)
    a = net.adv_head(z)  # (B, A)
    q_forward = net(x)
    q_manual = v + (a - a.mean(dim=1, keepdim=True))

    assert torch.allclose(q_forward, q_manual, atol=1e-6)

    q_mean = q_manual.mean(dim=1, keepdim=True)  # (B, 1)
    assert torch.allclose(q_mean, v, atol=1e-6)


def test_double_dqn_target_uses_online_argmax_and_target_value() -> None:
    """
    Double DQN target bootstrap:
    a* = argmax_a Q_online(s')
    next_q = Q_target(s', a*)
    """
    # Batch size 2, 3 actions
    q_online_next = torch.tensor(data=[[1.0, 5.0, 2.0], [3.0, 1.0, 0.0]], dtype=torch.float32)
    q_target_next = torch.tensor(data=[[10.0, 20.0, 30.0], [7.0, 8.0, 9.0]], dtype=torch.float32)

    a_star = torch.argmax(q_online_next, dim=1)  # [1, 0]
    picked = q_target_next.gather(dim=1, index=a_star.unsqueeze(1)).squeeze(1)

    assert picked.tolist() == [20.0, 7.0]

def test_td_target_masks_terminal_transitions() -> None:
    """
    Terminal handling in TD targets.
    If done=1, we must not bootstrap, so the target reduces to the immediate reward:
        y = r + (1 - done) * gamma * next_q
    """
    gamma = 0.99
    rewards = torch.tensor([1.0, 1.0])
    dones = torch.tensor([1.0, 0.0])  # first is terminal
    next_q = torch.tensor([100.0, 100.0])

    target = rewards + (1.0 - dones) * gamma * next_q
    assert target.tolist() == [1.0, 1.0 + 0.99 * 100.0]
