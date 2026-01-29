from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings
import numpy as np


@dataclass(frozen=True)  # frozen=True makes the dataclass immutable -> you cannot reassign its attributes after creation
class Transition:
    """
    One transition tuple for a tabular MDP.

    :param prob: Transition probability.
        :type prob: float
    :param next_state: Next state index.
        :type next_state: int
    :param reward: Reward received.
        :type reward: float
    :param done: Whether the transition ends the episode.
        :type done: bool
    """
    prob: float
    next_state: int
    reward: float
    done: bool


class GridworldMDP:
    """
    A small deterministic Gridworld MDP.

    This is mainly used for planning algorithms (value iteration/policy iteration).
    It keeps everything explicit and easy to print as a grid.

    States are integers in [0, n_states-1], mapped from (row, col).

    Actions:
    - 0: up
    - 1: right
    - 2: down
    - 3: left

    Default behaviour:
    - stepping into walls keeps you in the same state
    - non-terminal steps give `step_reward` (default -1)
    - terminal states are absorbing with reward 0 and done=True

    :param height: Number of rows.
        :type height: int
    :param width: Number of columns.
        :type width: int
    :param terminal_states: Terminal states as (row, col) positions.
        :type terminal_states: Iterable[tuple[int, int]]
    :param step_reward: Reward for non-terminal transitions.
        :type step_reward: float
    :param seed: RNG seed (only used if you sample start states for simulation).
        :type seed: int | None
    """

    def __init__(
        self,
        height: int = 4,
        width: int = 4,
        terminal_states: Iterable[tuple[int, int]] = ((0, 0), (3, 3)),
        step_reward: float = -1.0,
        seed: int | None = None,
    ):
        self.height = int(height)
        self.width = int(width)
        self.n_states = self.height * self.width  # number of states
        self.n_actions = 4  # number of actions

        self.step_reward = float(step_reward)
        self.rng = np.random.default_rng(seed)

        self.terminal_states = {self.pos_to_state(row=r, col=c) for (r, c) in terminal_states}

        # For optional simulation (not needed for DP, but it can be useful if you want to run simulations)
        self._state: int | None = None
        self._needs_reset = True
        self._done = False

    def pos_to_state(self, row: int, col: int) -> int:
        """
        Convert grid position (row, col) to a state index.

        :param row: Row index.
            :type row: int
        :param col: Column index.
            :type col: int

        :return: State index.
            :rtype: int
        """
        return row * self.width + col

    def state_to_pos(self, state: int) -> tuple[int, int]:
        """
        Convert a state index to grid position (row, col).

        :param state: State index.
            :type state: int

        :return: (row, col) position.
            :rtype: tuple[int, int]
        """
        row = state // self.width
        col = state % self.width
        return int(row), int(col)

    def is_terminal(self, state: int) -> bool:
        """
        Check whether a state is terminal.

        :param state: State index.
            :type state: int

        :return: True if terminal.
            :rtype: bool
        """
        return state in self.terminal_states

    def transitions(self, state: int, action: int) -> list[Transition]:
        """
        Return the transition list for (state, action).

        This Gridworld is deterministic, so this list always has exactly one element.

        :param state: Current state index.
            :type state: int
        :param action: Action index in {0,1,2,3}.
            :type action: int

        :return: List of Transition objects.
            :rtype: list[Transition]
        """
        if self.is_terminal(state):
            return [Transition(prob=1.0, next_state=state, reward=0.0, done=True)]

        row, col = self.state_to_pos(state)

        if action == 0:  # up
            row2, col2 = max(0, row - 1), col
        elif action == 1:  # right
            row2, col2 = row, min(self.width - 1, col + 1)
        elif action == 2:  # down
            row2, col2 = min(self.height - 1, row + 1), col
        elif action == 3:  # left
            row2, col2 = row, max(0, col - 1)
        else:
            raise ValueError(f"Invalid action {action}. Must be in {{0,1,2,3}}.")

        next_state = self.pos_to_state(row2, col2)
        done = self.is_terminal(next_state)
        reward = 0.0 if done else self.step_reward

        return [Transition(prob=1.0, next_state=next_state, reward=float(reward), done=bool(done))]

    # Optional simulation methods (handy for possible simulations using gridworld)

    def reset(self, start_state: int | None = None) -> int:
        """
        Reset the internal state for simulation.

        :param start_state: Optional start state. If None, use a random non-terminal state.
            :type start_state: int | None

        :return: Start state index.
            :rtype: int
        """
        if start_state is not None:
            start_state = int(start_state)

            if not (0 <= start_state < self.n_states):
                raise ValueError(f"start_state={start_state} out of bounds [0, {self.n_states - 1}]")

            if self.is_terminal(start_state):
                # If you pick a terminal state as starting state, the simulation will pick another state to start
                # using candidates list. You can deactivate this function, changing this part of the script
                warnings.warn(message=f"The state {start_state} is terminal. Another state will be sampled randomly.",
                              category=RuntimeWarning)
            else:
                self._state = int(start_state)
                self._needs_reset = False
                self._done = False  # new episode
                return self._state

        candidates = [s for s in range(self.n_states) if not self.is_terminal(s)]
        if not candidates:
            raise ValueError("All states are terminal. Cannot sample a non-terminal start state.")  # The gridworld is a black hole

        self._state = int(self.rng.choice(candidates))
        self._needs_reset = False
        self._done = False
        return self._state

    def step(self, action: int) -> tuple[int, float, bool]:
        """
        Step the internal simulator by one action.

        :param action: Action index.
            :type action: int

        :return: (next_state, reward, done)
            :rtype: tuple[int, float, bool]
        """
        if self._needs_reset:
            raise RuntimeError("You must call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before calling step() again.")

        tr = self.transitions(self._state, action)[0]
        self._state = tr.next_state
        self._done = tr.done
        return tr.next_state, tr.reward, tr.done
