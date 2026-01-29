from __future__ import annotations

from typing import Iterable

import gymnasium as gym


def make_first_available(env_ids: Iterable[str]) -> gym.Env:
    """
    Try environment IDs in order and return the first one that works.

    This is useful because toy-text env IDs can differ slightly across versions.

    :param env_ids: Candidate Gymnasium environment IDs.
        :type env_ids: Iterable[str]

    :return: A created Gymnasium environment.
        :rtype: gym.Env
    """
    last_err: Exception | None = None
    for env_id in env_ids:
        try:
            return gym.make(env_id)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"None of the env IDs worked: {list(env_ids)}") from last_err
