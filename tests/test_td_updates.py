import numpy as np

from rl_algorithms_guide.tabular import QLearningAgent, SARSAgent


def test_q_learning_one_step_update_non_terminal() -> None:
    """
    Q-learning update (non-terminal):

        Q <- Q + alpha * (r + gamma * max Q(s',.) - Q)

    This test checks the numerical update exactly.
    """
    agent = QLearningAgent(
        n_states=2,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        seed=0,
    )

    # Set a known Q table
    agent.Q[:] = 0.0
    agent.Q[1, 0] = 2.0
    agent.Q[1, 1] = 1.0  # max at next_state=1 is 2.0

    state = 0
    action = 1
    reward = 1.0
    next_state = 1
    done = False

    # target = 1 + 0.9 * 2 = 2.8
    # old Q = 0
    # new Q = 0 + 0.5 * (2.8 - 0) = 1.4
    agent.update(state, action, reward, next_state, done)
    assert np.isclose(a=agent.Q[state, action], b=1.4)


def test_q_learning_one_step_update_terminal() -> None:
    """
    Q-learning update (terminal): no bootstrap from next state.

        target = r
    """
    agent = QLearningAgent(
        n_states=2,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        seed=0,
    )

    agent.Q[:] = 0.0

    state = 0
    action = 0
    reward = 1.0
    next_state = 1
    done = True

    # target = 1
    # new Q = 0 + 0.5 * (1 - 0) = 0.5
    agent.update(state, action, reward, next_state, done)
    assert np.isclose(a=agent.Q[state, action], b=0.5)


def test_sarsa_one_step_update_non_terminal() -> None:
    """
    SARSA update (non-terminal):

        Q <- Q + alpha * (r + gamma * Q(s',a') - Q)

    This test checks the numerical update exactly.
    """
    agent = SARSAgent(
        n_states=2,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        seed=0,
    )

    agent.Q[:] = 0.0
    agent.Q[1, 1] = 2.0  # Q(next_state=1, next_action=1)

    state = 0
    action = 0
    reward = 1.0
    next_state = 1
    next_action = 1
    done = False

    # target = 1 + 0.9 * 2 = 2.8
    # old Q = 0
    # new Q = 0 + 0.5 * (2.8 - 0) = 1.4
    agent.update(state, action, reward, next_state, next_action, done)
    assert np.isclose(a=agent.Q[state, action], b=1.4)


def test_sarsa_one_step_update_terminal() -> None:
    """
    SARSA update (terminal): no bootstrap from next state.

        target = r
    """
    agent = SARSAgent(
        n_states=2,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        seed=0,
    )

    agent.Q[:] = 0.0

    state = 0
    action = 1
    reward = 1.0
    next_state = 1
    next_action = 0  # irrelevant when done=True
    done = True

    # target = 1
    # new Q = 0 + 0.5 * (1 - 0) = 0.5
    agent.update(state, action, reward, next_state, next_action, done)
    assert np.isclose(a=agent.Q[state, action], b=0.5)
