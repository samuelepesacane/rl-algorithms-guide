"""
Train REINFORCE on CartPole-v1 and save learning-curve plots.

This script is the runnable entry point for the REINFORCE algorithm
described in docs/04_pg_ac/theory.md (Sections 10-10.3).

REINFORCE is episode-based: we collect one full episode, compute
reward-to-go G_t for every step, then do one gradient update. The
script runs the same experiment over multiple random seeds and saves
mean ± std plots so you can see how stable the algorithm is, not just
how it did on one lucky run.

Plots saved (in --plot-dir):
  reinforce_cartpole_returns_mean_std.png   -- learning curve over episodes
  reinforce_cartpole_policy_loss_mean_std.png -- policy loss over episodes

Quickstart:
  python examples/04_pg_ac/train_reinforce_cartpole.py
  python examples/04_pg_ac/train_reinforce_cartpole.py --no-baseline --no-adv-norm   # raw REINFORCE
  python examples/04_pg_ac/train_reinforce_cartpole.py --n-seeds 1 --episodes 300    # single fast run
"""

from __future__ import annotations

import argparse
import os
import gymnasium as gym
import numpy as np

from rl_algorithms_guide.common.plotting import pad_curves_with_last_value, save_lines_with_bands
from rl_algorithms_guide.common.seeding import seed_everything
from rl_algorithms_guide.pg_ac.reinforce import ReinforceAgent, ReinforceConfig


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="REINFORCE on CartPole-v1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed. Seeds [seed, seed+1, ..., seed+n_seeds-1] are used.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of independent runs to average over for mean/std plots.",
    )
    parser.add_argument(
        "--episodes", type=int, default=600,
        help="Number of episodes per seed. REINFORCE updates once per episode.",
    )

    # Algorithm
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor used when computing reward-to-go G_t.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Policy network learning rate (Adam).",
    )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[64, 64],
        help="Hidden layer sizes for the policy MLP. Example: --hidden-sizes 128 128.",
    )
    parser.add_argument(
        "--no-adv-norm", action="store_true",
        help=(
            "Disable advantage normalization. "
            "By default, G_t values are standardized to zero mean / unit std within each "
            "episode, which keeps the update step size consistent across episodes."
        ),
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help=(
            "Disable the running-mean EMA baseline. "
            "By default, a scalar exponential moving average of past episode returns is "
            "subtracted from G_t to reduce variance without biasing the gradient."
        ),
    )
    parser.add_argument(
        "--baseline-momentum", type=float, default=0.9,
        help="EMA momentum for the running-mean baseline (higher -> slower updates).",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=10.0,
        help="L2 gradient clipping threshold. Set to 0 to disable.",
    )

    # Plotting
    parser.add_argument(
        "--plot-dir", type=str, default="assets/plots",
        help="Directory where plot PNGs are saved.",
    )
    parser.add_argument(
        "--smooth-window", type=int, default=10,
        help="Moving-average window for smoothing curves (1 = no smoothing).",
    )
    parser.add_argument(
        "--band-k", type=float, default=1.0,
        help="Uncertainty band width: plots mean ± band_k * std.",
    )

    return parser.parse_args()


def run_one_seed(
    *,
    seed: int,
    args: argparse.Namespace,
    cfg: ReinforceConfig
) -> tuple[list[float], list[float]]:
    """
    Run one full REINFORCE training session and return learning curves.

    Each episode follows the standard REINFORCE loop:
      1) Roll out the current policy until the episode ends.
      2) Call update_from_episode() with the full trajectory.
      3) Log the episode return and policy loss.

    :param seed: Random seed for this run.
        :type seed: int
    :param args: Parsed CLI args.
        :type args: argparse.Namespace
    :param cfg: REINFORCE configuration.
        :type cfg: ReinforceConfig

    :return: (episode_returns, policy_losses), one value per episode.
        :rtype: tuple[list[float], list[float]]
    """
    seed_everything(seed=seed, use_torch=True)

    env = gym.make("CartPole-v1")
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    agent = ReinforceAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg, seed=seed)

    episode_returns: list[float] = []
    policy_losses: list[float] = []

    for ep in range(int(args.episodes)):
        # Fresh episode with a deterministic seed so individual episodes are reproducible
        obs, _ = env.reset(seed=int(seed + ep))
        done = False

        # Buffers to store the trajectory
        # REINFORCE needs the full episode before updating
        obs_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        rew_buf: list[float] = []

        while not done:
            action, _logp = agent.act(obs=obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            obs_buf.append(np.asarray(obs, dtype=np.float32))
            act_buf.append(int(action))
            rew_buf.append(float(reward))

            obs = next_obs

        # One gradient update on the completed episode
        metrics = agent.update_from_episode(obs_list=obs_buf, action_list=act_buf, reward_list=rew_buf)

        episode_returns.append(float(metrics["episode_return"]))
        policy_losses.append(float(metrics["policy_loss"]))

    env.close()
    return episode_returns, policy_losses


def main():
    """
    Train REINFORCE across multiple seeds and save mean/std plots.
    """
    args = parse_args()
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be > 0")

    os.makedirs(args.plot_dir, exist_ok=True)

    cfg = ReinforceConfig(
        gamma=float(args.gamma),
        lr=float(args.lr),
        hidden_sizes=tuple(int(x) for x in args.hidden_sizes),
        normalize_advantages=not bool(args.no_adv_norm),
        use_running_baseline=not bool(args.no_baseline),
        baseline_momentum=float(args.baseline_momentum),
        max_grad_norm=float(args.max_grad_norm)
    )

    seeds = [int(args.seed + i) for i in range(int(args.n_seeds))]

    all_returns: list[np.ndarray] = []
    all_losses: list[np.ndarray] = []

    for s in seeds:
        rets, losses = run_one_seed(seed=s, args=args, cfg=cfg)

        all_returns.append(np.asarray(rets, dtype=np.float64))
        all_losses.append(np.asarray(losses, dtype=np.float64))

        if rets:
            print(
                f"[seed={s}] REINFORCE | episodes={len(rets)} | "
                f"last_return={rets[-1]:.1f} | mean_last_10={np.mean(rets[-10:]):.1f}"
            )

    prefix = "reinforce_cartpole"

    # Plot 1: episode return (learning curve)
    returns_mat = pad_curves_with_last_value(curves=all_returns)
    returns_mean = np.nanmean(returns_mat, axis=0)
    returns_std = np.nanstd(returns_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[returns_mean],
        ys_std=[returns_std],
        labels=[f"return (n={len(seeds)})"],
        title="Episode Return (REINFORCE) on CartPole-v1",
        xlabel="Episode",
        ylabel="Return",
        out_path=os.path.join(args.plot_dir, f"{prefix}_returns_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )

    # Plot 2: policy loss
    # Loss is noisy per episode but should trend downward as the policy improves
    loss_mat = pad_curves_with_last_value(curves=all_losses)
    loss_mean = np.nanmean(loss_mat, axis=0)
    loss_std = np.nanstd(loss_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[loss_mean],
        ys_std=[loss_std],
        labels=[f"policy loss (n={len(seeds)})"],
        title="Policy Loss (REINFORCE) on CartPole-v1",
        xlabel="Episode",
        ylabel="Loss",
        out_path=os.path.join(args.plot_dir, f"{prefix}_policy_loss_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )


if __name__ == "__main__":
    main()
