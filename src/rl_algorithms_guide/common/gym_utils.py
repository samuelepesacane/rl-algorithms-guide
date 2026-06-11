from __future__ import annotations

from typing import Iterable

import gymnasium as gym


def extract_env_name(env_id: str) -> str:
    """
    Turn a Gymnasium environment id into a filesystem-safe name for plots.

    Training scripts expose a `--env-id` flag with a default (e.g. LunarLander-v3),
    but it is always possible to choose a custom environment (so a custom name).
    With this function, the name is extracted and possible symbols changed:
    the namespace separator "/" (as in "ALE/Breakout-v5") and the version hyphen are
    both folded to underscores so the result is one clean token: "ale_breakout_v5".

    :param env_id: A Gymnasium environment id.
        :type env_id: str

    :return: A lowercase, underscore-separated name.
        :rtype: str
    """
    return env_id.lower().replace("/", "_").replace("-", "_")


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
