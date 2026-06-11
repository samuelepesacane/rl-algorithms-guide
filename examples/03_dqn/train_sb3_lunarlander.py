"""
Train SB3 DQN on LunarLander-v3 across seeds and save an evaluation-curve plot.

This is the harder-environment "calibrated baseline" for Block 03, the counterpart
to train_sb3_cartpole.py on LunarLander-v3.
The minimal implementation in `dqn.py` (Vanilla, Double, and Dueling DQN)
is written for clarity and therefore prioritizes readability over extensive tuning. Stable-Baselines3 (SB3),
on the other hand, provides a production-quality DQN implementation. Running it on the same environment helps quantify
how much performance the repository's own implementations may be leaving on the table.
Additionally, SB3's DQN is a vanilla DQN implementation and does not include Double DQN or Dueling DQN variants.
As a result, it serves as a useful vanilla-DQN reference baseline for this block.
The goal of this script is also to demonstrate how to use SB3 in a simple DQN training workflow.
This allows you to both build your own DQN implementation from scratch and understand how to train the same algorithm
using a well-established library. Comparing the two approaches can help develop a stronger intuition for the algorithm
while also providing a practical template for running SB3 on your own environments.

What this script does:
  1) For each seed, train an SB3 DQN model on LunarLander-v3 for
     ``--total-steps`` env steps.
  2) Use SB3's ``EvalCallback`` to evaluate the policy every ``--eval-freq``
     env steps and record the mean return.
  3) Average the evaluation curves across seeds and plot mean +/- std.

The DQN hyperparameters below follow the RL Baselines3 Zoo tuning for
LunarLander-v3. 

LunarLander-v3 needs the Box2D physics backend, which the ``box2d`` extra pulls
in (swig + gymnasium[box2d]). v3 is the current stable LunarLander in gymnasium
1.x (the older v2 was removed), so every block 03 and 05 LunarLander script uses v3.

Plot saved (in --plot-dir):
  sb3_dqn_lunarlander_eval_returns_mean_std.png   -- SB3 DQN eval return

Requirements:
  pip install -e ".[sb3,box2d]"

Quickstart:
  python examples/03_dqn/train_sb3_lunarlander.py
  python examples/03_dqn/train_sb3_lunarlander.py --total-steps 300000 --n-seeds 5
"""

from __future__ import annotations

import argparse
import os
import tempfile
import gymnasium as gym
import numpy as np

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required for this script. "
        "Install it with: pip install -e '.[sb3,box2d]'"
    ) from exc

from rl_algorithms_guide.common.plotting import save_lines_with_bands
from rl_algorithms_guide.common.seeding import seed_everything
from rl_algorithms_guide.common.gym_utils import extract_env_name


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args.

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Stable-Baselines3 DQN baseline on LunarLander-v3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--env-id", type=str, default="LunarLander-v3",
                        help="Gymnasium environment id (must be a discrete-action env for DQN).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed. Seeds [seed, ..., seed+n_seeds-1] are used.")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of independent SB3 runs.")
    parser.add_argument("--total-steps", type=int, default=200_000,
                        help="Total env steps per SB3 training run.")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Env steps between policy evaluations.")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                        help="Episodes per evaluation point.")

    parser.add_argument("--plot-dir", type=str, default="assets/plots",
                        help="Directory where plot PNGs are saved.")
    parser.add_argument("--smooth-window", type=int, default=3,
                        help="Moving-average window for smoothing the curve.")
    parser.add_argument("--band-k", type=float, default=1.0,
                        help="Uncertainty band width: plots mean +/- band_k * std.")

    return parser.parse_args()


def _train_sb3_run(
    *,
    env_id: str,
    seed: int,
    total_steps: int,
    eval_freq: int,
    n_eval_episodes: int
) -> np.ndarray:
    """
    Train one SB3 DQN model on the given env and return its evaluation curve.

    A separate single-env wrapper is used for evaluation so that the
    epsilon-greedy exploration used during training does not pollute the
    eval distribution.

    :param env_id: Gymnasium environment id to train on.
        :type env_id: str
    :param seed: Random seed for this run.
        :type seed: int
    :param total_steps: Total env steps to train.
        :type total_steps: int
    :param eval_freq: Eval frequency in env steps.
        :type eval_freq: int
    :param n_eval_episodes: Episodes per evaluation point.
        :type n_eval_episodes: int

    :return: 1D array of mean evaluation returns, one per eval point.
        :rtype: np.ndarray
    """
    train_env = gym.make(env_id)
    # SB3's EvalCallback expects a vectorised env, so the eval env is wrapped in
    # DummyVecEnv (a single-process vec env that runs the one env in-line).
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])

    # RL Baselines3 Zoo hyperparameters for LunarLander-v3.
    model = DQN(
        "MlpPolicy",                              # network type SB3 builds; MlpPolicy = fully connected net for vector obs
        train_env,                                # env used for training
        learning_rate=6.3e-4,                     # step size of the Adam optimiser
        batch_size=128,                           # transitions sampled from the replay buffer per gradient step
        buffer_size=50_000,                       # max transitions kept in the replay buffer
        learning_starts=0,                        # env steps of pure exploration before training begins (0 = start learning immediately)
        gamma=0.99,                               # discount factor for future rewards
        target_update_interval=250,               # gradient steps between hard copies of the online net into the target net
        train_freq=4,                             # env steps collected between training phases
        gradient_steps=-1,                        # gradient updates per training phase; -1 = as many as steps just collected (train_freq)
        exploration_fraction=0.12,                # fraction of total steps over which epsilon is annealed down
        exploration_final_eps=0.1,                # epsilon value held after annealing finishes
        policy_kwargs=dict(net_arch=[256, 256]),  # hidden-layer sizes of the MLP (two layers of 256 units)
        verbose=0,                                # 0 = silent; 1 or 2 print SB3 training logs
        seed=int(seed)                            # seeds SB3's RNGs so the run is reproducible
    )

    # EvalCallback only records evaluation results to memory when a log_path is
    # set; without it self.evaluations_results stays empty. We point it at a
    # throwaway temp dir so the in-memory curve is populated.
    with tempfile.TemporaryDirectory() as log_dir:
        eval_callback = EvalCallback(
            eval_env=eval_env,                      # separate env the policy is evaluated on
            log_path=log_dir,                       # dir where eval results are written (also enables the in-memory curve)
            eval_freq=int(eval_freq),               # env steps between evaluations
            n_eval_episodes=int(n_eval_episodes),   # episodes averaged per evaluation point
            deterministic=True,                     # act greedily during eval (no epsilon exploration)
            render=False,                           # do not open a render window during eval
            verbose=0                               # 0 = silent; 1 prints eval logs
        )

        # Run training; the callback fires every eval_freq steps and stashes each
        # evaluation's per-episode returns in eval_callback.evaluations_results.
        model.learn(total_timesteps=int(total_steps), callback=eval_callback)

        # Each entry is the list of episode returns at one eval point; we average
        # them into a single mean return per eval point.
        if eval_callback.evaluations_results:
            means = np.array(
                [float(np.mean(r)) for r in eval_callback.evaluations_results],
                dtype=np.float64
            )
        else:
            # No eval ran (e.g. total_steps < eval_freq); return an empty curve.
            means = np.array([], dtype=np.float64)

    # Release the env resources now that this run is finished.
    train_env.close()
    eval_env.close()
    return means


def main():
    """
    Train SB3 DQN across seeds and save the evaluation-curve plot.
    """
    args = parse_args()
    if int(args.total_steps) <= 0:
        raise ValueError("--total-steps must be > 0")

    os.makedirs(args.plot_dir, exist_ok=True)

    # One independent training run per seed; the curves are averaged at the end.
    seeds = [int(args.seed + i) for i in range(int(args.n_seeds))]

    curves: list[np.ndarray] = []
    for s in seeds:
        seed_everything(seed=s, use_torch=True)   # make this run reproducible
        curve = _train_sb3_run(
            env_id=str(args.env_id),
            seed=s,
            total_steps=int(args.total_steps),
            eval_freq=int(args.eval_freq),
            n_eval_episodes=int(args.n_eval_episodes)
        )
        curves.append(curve)
        if curve.size > 0:
            print(f"[seed={s}] SB3 DQN | eval_points={curve.size} | last_eval={curve[-1]:.1f}")

    # Align curves to the shortest length so the same x-axis applies to all.
    if not any(c.size > 0 for c in curves):
        print("No eval curves produced. Increase --total-steps or check SB3 installation.")
        return

    min_len = min(int(c.shape[0]) for c in curves if c.size > 0)
    mat = np.stack([c[:min_len] for c in curves if c.size >= min_len], axis=0)

    # Derive the plot name from the actual env id
    env_name = extract_env_name(args.env_id)
    save_lines_with_bands(
        ys_mean=[np.nanmean(mat, axis=0)],
        ys_std=[np.nanstd(mat, axis=0)],
        labels=[f"SB3 DQN (n={mat.shape[0]})"],
        title=f"SB3 DQN on {args.env_id} (eval return)",
        xlabel="Evaluation point",
        ylabel="Mean eval return",
        out_path=os.path.join(args.plot_dir, f"sb3_dqn_{env_name}_eval_returns_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )


if __name__ == "__main__":
    main()
