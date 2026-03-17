"""
Train single-step Actor-Critic on CartPole-v1 and save learning-curve plots.

This script is the runnable entry point for the Actor-Critic algorithm
described in docs/04_pg_ac/theory.md (Sections 11-11.4).

Unlike REINFORCE, Actor-Critic updates after every single step rather than
waiting for the episode to end. The critic's TD error is used as an online
advantage estimate, which reduces variance at the cost of some bias.

The training loop:
  1) Take one step in the environment.
  2) Compute TD error: delta = r + gamma * V(s') * (1-done) - V(s)
  3) Update actor with: -log pi(a|s) * stop_grad(delta)
  4) Update critic with: delta^2
  5) Repeat.

Plots saved (in --plot-dir):
  actor_critic_cartpole_returns_mean_std.png      -- episode returns
  actor_critic_cartpole_policy_loss_mean_std.png  -- actor loss per step
  actor_critic_cartpole_value_loss_mean_std.png   -- critic loss per step

Quickstart:
  python examples/04_pg_ac/train_actor_critic_cartpole.py
  python examples/04_pg_ac/train_actor_critic_cartpole.py --entropy-coef 0.01   # add exploration bonus
  python examples/04_pg_ac/train_actor_critic_cartpole.py --n-seeds 1 --total-steps 60000
"""

from __future__ import annotations

import argparse
import os
import gymnasium as gym
import numpy as np

from rl_algorithms_guide.common.plotting import pad_curves_with_last_value, save_lines_with_bands
from rl_algorithms_guide.common.seeding import seed_everything
from rl_algorithms_guide.pg_ac.actor_critic import ActorCriticAgent, ActorCriticConfig


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args.

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Single-step Actor-Critic (TD advantage) on CartPole-v1.",
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
        "--total-steps", type=int, default=120_000,
        help=(
            "Total environment steps per seed. "
            "Actor-Critic updates every step, so this equals the number of gradient updates."
        ),
    )

    # Algorithm
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor used in the TD target: r + gamma * V(s') * (1-done).",
    )
    parser.add_argument(
        "--lr-actor", type=float, default=3e-4,
        help="Learning rate for the actor (Adam). Often set lower than the critic.",
    )
    parser.add_argument(
        "--lr-critic", type=float, default=1e-3,
        help=(
            "Learning rate for the critic (Adam). "
            "A higher critic lr lets V(s) track the true value faster, giving the actor "
            "a more accurate advantage signal sooner."
        ),
    )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[64, 64],
        help="Hidden layer sizes for both actor and critic MLPs.",
    )
    parser.add_argument(
        "--value-coef", type=float, default=0.5,
        help="Scaling factor for the critic loss. Balances actor vs critic update magnitude.",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.0,
        help=(
            "Weight for the entropy bonus added to the actor loss. "
            "Set > 0 (e.g. 0.01) to discourage the policy from collapsing to a single "
            "action too early. Set to 0 for the standard Actor-Critic with no extra exploration."
        ),
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
    cfg: ActorCriticConfig
) -> tuple[list[float], list[float], list[float]]:
    """
    Run one full Actor-Critic training session and return learning curves.

    Returns are tracked per episode; losses are tracked per step (one update per step).

    :param seed: Random seed for this run.
        :type seed: int
    :param args: Parsed CLI args.
        :type args: argparse.Namespace
    :param cfg: Actor-Critic configuration.
        :type cfg: ActorCriticConfig

    :return: (episode_returns, policy_losses, value_losses).
        :rtype: tuple[list[float], list[float], list[float]]
    """
    seed_everything(seed=seed, use_torch=True)

    env = gym.make("CartPole-v1")
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    agent = ActorCriticAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg, seed=seed)

    episode_returns: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []

    obs, _ = env.reset(seed=int(seed))
    ep_return = 0.0
    episode_idx = 0

    for step in range(1, int(args.total_steps) + 1):
        action, _logp = agent.act(obs=obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        # One update per step: the agent sees the (s, a, r, s', done) tuple immediately
        metrics = agent.update_step(
            obs=obs,
            action=int(action),
            reward=float(reward),
            next_obs=next_obs,
            done=done
        )

        ep_return += float(reward)
        policy_losses.append(float(metrics["policy_loss"]))
        value_losses.append(float(metrics["value_loss"]))

        if done:
            episode_returns.append(float(ep_return))
            episode_idx += 1
            # Increment seed per episode so different episodes have different stochasticity
            obs, _ = env.reset(seed=int(seed + episode_idx))
            ep_return = 0.0
        else:
            obs = next_obs

    env.close()
    return episode_returns, policy_losses, value_losses


def main():
    """
    Train Actor-Critic across multiple seeds and save mean/std plots.
    """
    args = parse_args()
    if int(args.total_steps) <= 0:
        raise ValueError("--total-steps must be > 0")

    os.makedirs(args.plot_dir, exist_ok=True)

    cfg = ActorCriticConfig(
        gamma=float(args.gamma),
        lr_actor=float(args.lr_actor),
        lr_critic=float(args.lr_critic),
        hidden_sizes=tuple(int(x) for x in args.hidden_sizes),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        max_grad_norm=float(args.max_grad_norm)
    )

    seeds = [int(args.seed + i) for i in range(int(args.n_seeds))]

    all_returns: list[np.ndarray] = []
    all_pi_losses: list[np.ndarray] = []
    all_v_losses: list[np.ndarray] = []

    for s in seeds:
        rets, pi_l, v_l = run_one_seed(seed=s, args=args, cfg=cfg)

        all_returns.append(np.asarray(rets, dtype=np.float64))
        all_pi_losses.append(np.asarray(pi_l, dtype=np.float64))
        all_v_losses.append(np.asarray(v_l, dtype=np.float64))

        if rets:
            print(
                f"[seed={s}] Actor-Critic | episodes={len(rets)} | "
                f"last_return={rets[-1]:.1f} | mean_last_10={np.mean(rets[-10:]):.1f}"
            )

    prefix = "actor_critic_cartpole"

    # Plot 1: episode return (learning curve)
    returns_mat = pad_curves_with_last_value(curves=all_returns)
    returns_mean = np.nanmean(returns_mat, axis=0)
    returns_std = np.nanstd(returns_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[returns_mean],
        ys_std=[returns_std],
        labels=[f"return (n={len(seeds)})"],
        title="Episode Return (Actor-Critic) on CartPole-v1",
        xlabel="Episode",
        ylabel="Return",
        out_path=os.path.join(args.plot_dir, f"{prefix}_returns_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )

    # Plot 2: actor (policy) loss per step
    pi_mat = pad_curves_with_last_value(curves=all_pi_losses)
    pi_mean = np.nanmean(pi_mat, axis=0)
    pi_std = np.nanstd(pi_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[pi_mean],
        ys_std=[pi_std],
        labels=[f"policy loss (n={len(seeds)})"],
        title="Policy Loss (Actor-Critic) on CartPole-v1",
        xlabel="Update step",
        ylabel="Loss",
        out_path=os.path.join(args.plot_dir, f"{prefix}_policy_loss_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )

    # Plot 3: critic (value) loss per step
    # Should decrease as V(s) becomes a better predictor of return
    v_mat = pad_curves_with_last_value(curves=all_v_losses)
    v_mean = np.nanmean(v_mat, axis=0)
    v_std = np.nanstd(v_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[v_mean],
        ys_std=[v_std],
        labels=[f"value loss (n={len(seeds)})"],
        title="Value Loss (Actor-Critic) on CartPole-v1",
        xlabel="Update step",
        ylabel="Loss",
        out_path=os.path.join(args.plot_dir, f"{prefix}_value_loss_mean_std.png"),
        smooth_window=int(args.smooth_window),
        band_k=float(args.band_k)
    )


if __name__ == "__main__":
    main()
