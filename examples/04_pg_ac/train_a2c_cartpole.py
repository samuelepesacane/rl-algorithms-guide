"""
Train A2C on CartPole-v1 and save learning-curve plots.

This script is the runnable entry point for the A2C algorithm described
in docs/04_pg_ac/theory.md (Sections 16-16.7).

A2C = Advantage Actor-Critic with synchronised parallel environments.
The idea is identical to single-step Actor-Critic, but instead of one
environment we run N environments in parallel using Gymnasium's
SyncVectorEnv. All environments step in lockstep so gradients can be
averaged over a batch of decorrelated transitions every rollout.

The training loop:
  1) All N environments run for `rollout-steps` steps, collecting
     observations, actions, rewards, and done flags into rollout buffers.
  2) Bootstrap from the last observation to compute n-step returns G_t.
  3) Advantage: A_t = G_t - V(s_t)
  4) One gradient update on the combined actor + critic loss.
  5) Repeat.

Setting --num-envs 1 and --rollout-steps 1 recovers single-step
Actor-Critic, which is a useful sanity check.

Plots saved (in --plot-dir):
  a2c_cartpole_returns_mean_std.png       -- episode returns (per rollout)
  a2c_cartpole_policy_loss_mean_std.png   -- actor loss per update
  a2c_cartpole_value_loss_mean_std.png    -- critic loss per update
  a2c_cartpole_entropy_mean_std.png       -- policy entropy per update

Quickstart:
  python examples/04_pg_ac/train_a2c_cartpole.py
  python examples/04_pg_ac/train_a2c_cartpole.py --num-envs 1 --rollout-steps 1   # = Actor-Critic
  python examples/04_pg_ac/train_a2c_cartpole.py --num-envs 8 --rollout-steps 5   # more parallelism
  python examples/04_pg_ac/train_a2c_cartpole.py --n-seeds 1 --total-steps 200000
"""

from __future__ import annotations

import argparse
import os
import gymnasium as gym
import numpy as np

from rl_algorithms_guide.common.plotting import pad_curves_with_last_value, save_lines_with_bands
from rl_algorithms_guide.common.seeding import seed_everything
from rl_algorithms_guide.pg_ac.a2c import A2CAgent, A2CConfig


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args.

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A2C (parallel Actor-Critic) on CartPole-v1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed. Seeds [seed, ..., seed+n_seeds-1] are used.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of independent runs to average for mean/std plots.",
    )
    parser.add_argument(
        "--total-steps", type=int, default=200_000,
        help=(
            "Total environment steps per seed (summed across all parallel environments). "
            "One update happens every rollout_steps * num_envs steps."
        ),
    )

    # Parallelism
    parser.add_argument(
        "--num-envs", type=int, default=4,
        help=(
            "Number of parallel environments (N). "
            "All environments step in sync inside a single Python process. "
            "More environments -> less correlated batch -> lower variance gradients. "
            "Set to 1 to approximate single-step Actor-Critic."
        ),
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=5,
        help=(
            "Number of steps each environment contributes per update (n). "
            "Each update uses a batch of num_envs * rollout_steps transitions. "
            "Longer rollouts give a better return estimate but delay the update."
        ),
    )

    # Algorithm
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor used in the n-step return.",
    )
    parser.add_argument(
        "--lr", type=float, default=7e-4,
        help=(
            "Learning rate for the shared Adam optimizer covering actor and critic. "
            "A single optimizer is the standard A2C setup."
        ),
    )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[64, 64],
        help="Hidden layer sizes for both actor and critic MLPs.",
    )
    parser.add_argument(
        "--value-coef", type=float, default=0.5,
        help="Scaling factor for the critic loss in the combined loss.",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help=(
            "Weight for the entropy bonus. "
            "A small positive value (e.g. 0.01) discourages the policy from collapsing "
            "to a single action. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5,
        help=(
            "L2 gradient clipping threshold applied jointly to actor and critic parameters. "
            "A tighter value (e.g. 0.5) is typical for A2C because a large batch can "
            "produce large gradients. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--no-adv-norm", action="store_true",
        help=(
            "Disable advantage normalization within each rollout batch. "
            "By default advantages are standardized to zero mean / unit std, which keeps "
            "the actor update step size stable across rollouts."
        ),
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
    cfg: A2CConfig
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Run one full A2C training session and return learning curves.

    The outer loop collects rollout_steps steps from num_envs environments,
    then calls agent.update() with the full rollout buffer.

    Episode returns are tracked by monitoring the done flags from
    SyncVectorEnv. When environment i signals done, its cumulative reward
    is logged and its counter is reset.

    :param seed: Random seed for this run.
        :type seed: int
    :param args: Parsed CLI args.
        :type args: argparse.Namespace
    :param cfg: A2C configuration.
        :type cfg: A2CConfig

    :return: (episode_returns, policy_losses, value_losses, entropies).
        episode_returns: one value per completed episode (across all envs).
        policy_losses, value_losses, entropies: one value per gradient update.
        :rtype: tuple[list[float], list[float], list[float], list[float]]
    """
    seed_everything(seed=seed, use_torch=True)

    num_envs = int(args.num_envs)
    rollout_steps = int(args.rollout_steps)
    total_steps = int(args.total_steps)

    # SyncVectorEnv runs N environments in the same process.
    # Each call to envs.step(actions) steps all N environments simultaneously
    # and returns batched (obs, reward, terminated, truncated, info) arrays.
    envs = gym.make_vec(
        "CartPole-v1",
        num_envs=num_envs,
        vectorization_mode="sync",
    )

    obs_dim = int(envs.single_observation_space.shape[0])
    n_actions = int(envs.single_action_space.n)

    agent = A2CAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg, seed=seed)

    episode_returns: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []

    # obs shape: (N, obs_dim)
    obs, _ = envs.reset(seed=int(seed))

    # Per-environment running episode return trackers
    ep_returns = np.zeros(num_envs, dtype=np.float64)

    # done flags from the previous step (used to mask bootstrapping at terminal states)
    dones = np.zeros(num_envs, dtype=np.float32)

    steps_done = 0

    while steps_done < total_steps:
        # Collect a rollout of rollout_steps steps from all N environments
        obs_buf = np.zeros(shape=(rollout_steps, num_envs, obs_dim), dtype=np.float32)
        act_buf = np.zeros(shape=(rollout_steps, num_envs), dtype=np.int64)
        rew_buf = np.zeros(shape=(rollout_steps, num_envs), dtype=np.float32)
        done_buf = np.zeros(shape=(rollout_steps, num_envs), dtype=np.float32)

        for t in range(rollout_steps):
            actions, _logps = agent.act_batch(obs=obs)

            next_obs, rewards, terminated, truncated, _ = envs.step(actions)
            step_dones = np.logical_or(terminated, truncated).astype(np.float32)

            obs_buf[t] = obs
            act_buf[t] = actions
            rew_buf[t] = rewards
            done_buf[t] = step_dones

            # Track episode returns: accumulate until done, then log and reset
            ep_returns += rewards
            for i in range(num_envs):
                if step_dones[i]:
                    episode_returns.append(float(ep_returns[i]))
                    ep_returns[i] = 0.0

            obs = next_obs
            dones = step_dones  # keep the last step's done flags so agent.update() can mask the bootstrap correctly

        steps_done += rollout_steps * num_envs

        # One gradient update on the collected rollout
        metrics = agent.update(
            obs_buf=obs_buf,
            act_buf=act_buf,
            rew_buf=rew_buf,
            done_buf=done_buf,
            last_obs=obs,
            last_dones=dones
        )

        policy_losses.append(float(metrics["policy_loss"]))
        value_losses.append(float(metrics["value_loss"]))
        entropies.append(float(metrics["entropy"]))

    envs.close()
    return episode_returns, policy_losses, value_losses, entropies


def main():
    """
    Train A2C across multiple seeds and save mean/std plots.
    """
    args = parse_args()
    if int(args.total_steps) <= 0:
        raise ValueError("--total-steps must be > 0")
    if int(args.num_envs) <= 0:
        raise ValueError("--num-envs must be > 0")
    if int(args.rollout_steps) <= 0:
        raise ValueError("--rollout-steps must be > 0")

    os.makedirs(args.plot_dir, exist_ok=True)

    cfg = A2CConfig(
        gamma=float(args.gamma),
        lr=float(args.lr),
        hidden_sizes=tuple(int(x) for x in args.hidden_sizes),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        max_grad_norm=float(args.max_grad_norm),
        normalize_advantages=not bool(args.no_adv_norm)
    )

    seeds = [int(args.seed + i) for i in range(int(args.n_seeds))]

    all_returns: list[np.ndarray] = []
    all_pi_losses: list[np.ndarray] = []
    all_v_losses: list[np.ndarray] = []
    all_entropies: list[np.ndarray] = []

    for s in seeds:
        rets, pi_l, v_l, ents = run_one_seed(seed=s, args=args, cfg=cfg)

        all_returns.append(np.asarray(rets, dtype=np.float64))
        all_pi_losses.append(np.asarray(pi_l, dtype=np.float64))
        all_v_losses.append(np.asarray(v_l, dtype=np.float64))
        all_entropies.append(np.asarray(ents, dtype=np.float64))

        if rets:
            print(
                f"[seed={s}] A2C | episodes={len(rets)} | "
                f"last_return={rets[-1]:.1f} | mean_last_10={np.mean(rets[-10:]):.1f}"
            )

    prefix = "a2c_cartpole"

    # Plot 1: episode return (learning curve)
    returns_mat = pad_curves_with_last_value(curves=all_returns)
    returns_mean = np.nanmean(returns_mat, axis=0)
    returns_std = np.nanstd(returns_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[returns_mean],
        ys_std=[returns_std],
        labels=[f"return (n={len(seeds)})"],
        title="Episode Return (A2C) on CartPole-v1",
        xlabel="Episode",
        ylabel="Return",
        out_path=os.path.join(args.plot_dir, f"{prefix}_returns_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )

    # Plot 2: actor (policy) loss per update
    pi_mat = pad_curves_with_last_value(curves=all_pi_losses)
    pi_mean = np.nanmean(pi_mat, axis=0)
    pi_std = np.nanstd(pi_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[pi_mean],
        ys_std=[pi_std],
        labels=[f"policy loss (n={len(seeds)})"],
        title="Policy Loss (A2C) on CartPole-v1",
        xlabel="Update",
        ylabel="Loss",
        out_path=os.path.join(args.plot_dir, f"{prefix}_policy_loss_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )

    # Plot 3: critic (value) loss per update
    v_mat = pad_curves_with_last_value(curves=all_v_losses)
    v_mean = np.nanmean(v_mat, axis=0)
    v_std = np.nanstd(v_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[v_mean],
        ys_std=[v_std],
        labels=[f"value loss (n={len(seeds)})"],
        title="Value Loss (A2C) on CartPole-v1",
        xlabel="Update",
        ylabel="Loss",
        out_path=os.path.join(args.plot_dir, f"{prefix}_value_loss_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )

    # Plot 4: policy entropy per update
    # Entropy should start high (random policy) and gradually decrease as
    # the policy converges, but not collapse to zero if entropy_coef > 0
    ent_mat = pad_curves_with_last_value(curves=all_entropies)
    ent_mean = np.nanmean(ent_mat, axis=0)
    ent_std = np.nanstd(ent_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[ent_mean],
        ys_std=[ent_std],
        labels=[f"entropy (n={len(seeds)})"],
        title="Policy Entropy (A2C) on CartPole-v1",
        xlabel="Update",
        ylabel="Entropy",
        out_path=os.path.join(args.plot_dir, f"{prefix}_entropy_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )


if __name__ == "__main__":
    main()
