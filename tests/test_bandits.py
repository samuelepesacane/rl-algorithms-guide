import numpy as np
from rl_algorithms_guide.bandits import GaussianBandit, EpsilonGreedyAgent, DecayingEpsilonGreedyAgent, UCBAgent


def test_gaussian_bandit_best_action_matches_argmax() -> None:
    """
    The bandit's reported best_action should match argmax(q_star).

    Test Goal: make sure the environment correctly identifies which arm is the best.
    Why this matters: our "% optimal action" plot depends on best_action being correct.
    """
    q_star = np.array([-1.0, 0.2, 0.1, 0.9], dtype=np.float64)
    env = GaussianBandit(k=4, q_star=q_star, std=0.0, seed=123)

    info = env.reset()
    assert info.best_action == int(np.argmax(q_star))
    assert np.allclose(info.q_star, q_star)


def test_gaussian_bandit_step_is_deterministic_when_std_zero() -> None:
    """
    With std=0, rewards should equal the arm mean exactly.

    Test Goal: verify that reward sampling matches the environment definition.
    Why this matters: it confirms step() is implemented correctly and gives you a deterministic mode for debugging.
    """
    q_star = np.array([0.0, 1.5], dtype=np.float64)
    env = GaussianBandit(k=2, q_star=q_star, std=0.0, seed=0)

    r0 = env.step(0)
    r1 = env.step(1)

    assert r0 == 0.0
    assert r1 == 1.5


def test_epsilon_greedy_incremental_mean_update() -> None:
    """
    After multiple updates on the same action, Q[action] should equal the sample mean.

    Test Goal: confirm the incremental mean update is correct.
    Why this matters: both ε-greedy and UCB rely on the same "sample-average" update logic. If this is wrong, everything else is wrong too.
    """
    agent = EpsilonGreedyAgent(k=2, epsilon=0.0, seed=0)

    rewards = [1.0, 2.0, 3.0]
    for r in rewards:
        agent.update(action=1, reward=r)

    assert agent.N[1] == 3
    assert agent.Q[1] == np.mean(rewards)

def test_decaying_epsilon_decreases() -> None:
    """
    Decaying ε-greedy should start at epsilon_start, decrease over time,
    and never go below epsilon_end.

    Test Goal: validate the schedule behavior of the decaying ε-greedy variant.
    Why this matters: it ensures your "decaying exploration" works as intended and doesn't do something surprising like increasing again or going below the minimum.
    """
    agent = DecayingEpsilonGreedyAgent(
        k=2,
        epsilon_start=1.0,
        epsilon_end=0.1,
        decay_steps=10,
        schedule="linear",
        seed=0,
    )

    eps_values = [agent.current_epsilon()]

    # We call select_action() just to advance the internal time step
    for _ in range(25):
        _ = agent.select_action()
        eps_values.append(agent.current_epsilon())

    # Should be non-increasing (allow tiny numerical wiggles)
    for i in range(len(eps_values) - 1):
        assert eps_values[i] >= eps_values[i + 1] - 1e-12

    # Should clamp at epsilon_end
    assert eps_values[-1] == 0.1


def test_ucb_pulls_each_arm_once_first() -> None:
    """
    UCBAgent should select arms in order 0...k-1 for the first k calls to select_action().

    Test Goal: confirm the "warm start" behavior used for UCB.
    Why this matters: it prevents division-by-zero and keeps the implementation deterministic at the start.
    """
    k = 5
    agent = UCBAgent(k=k, c=2.0)

    actions = [agent.select_action() for _ in range(k)]
    assert actions == list(range(k))


def test_ucb_prefers_high_value_when_counts_equal() -> None:
    """
    If all counts are equal, UCB reduces to picking the arm with the highest Q (bonus is the same).

    Test Goal: check the logic of the UCB score when uncertainty is the same.
    Why this matters: it verifies the UCB selection step is correctly combining Q and the bonus (and that the argmax logic is correct).
    """
    agent = UCBAgent(k=4, c=2.0)

    # Move time forward past the "pull each arm once" phase.
    for _ in range(4):
        _ = agent.select_action()

    # Pretend each arm has been tried once.
    agent.N[:] = 1
    agent.Q[:] = 0.0
    agent.Q[0] = 10.0

    a = agent.select_action()
    assert a == 0
