"""Train DQN-style agents on CartPole-v1.

Before using CartPole-v1, a good practice is to study its documentation on gymnasium.
This is valid for all envs you try to use in a project. You should know the env before using it.

This script is meant as the fast sanity check for Block 03.

Algorithms supported (via --algo):
- dqn
- double
- dueling
- double_dueling

It follows the theory in docs/03_dqn/theory.md:
- Experience replay
- Target network (frozen copy for stable TD targets)
- Epsilon-greedy exploration with linear annealing
- TD targets with Huber loss
- Optional Double DQN target
- Optional Dueling architecture

Notes on `done`:
Gymnasium returns two flags: `terminated` and `truncated`.
- `terminated=True` means the task ended (e.g., failure/success terminal state)
- `truncated=True` usually means a time-limit cutoff

In this educational repo we treat (terminated OR truncated) as done, i.e. we do not
bootstrap across episode boundaries. This is simple and stable, but slightly biased
for pure time-limit truncation.
"""

from __future__ import annotations

import argparse
import os
import gymnasium as gym
import numpy as np

from rl_algorithms_guide.common.plotting import save_lines, save_lines_with_bands
from rl_algorithms_guide.common.seeding import seed_everything
from rl_algorithms_guide.dqn.dqn import DQNAgent, DQNConfig, linear_epsilon


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args.

    We expose the main knobs of DQN as command-line flags so you can:
    - reproduce runs (seed)
    - compare variants (algo)
    - tweak core stability hyperparameters without editing code

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Train DQN variants on CartPole-v1.")
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=["dqn", "double", "dueling", "double_dueling"],
        help=("Which DQN variant to run. "
              "'dqn' = vanilla DQN, "
              "'double' = Double DQN targets, "
              "'dueling' = dueling Q-network, "
              "'double_dueling' = both."
              ),
    )
    parser.add_argument("--total-steps",
                        type=int,
                        default=50_000,
                        help="Total environment interaction steps to run (training happens during these steps)."
                        )

    # Multi-seed support
    # Either pass an explicit list:   --seeds 0 1 2 3 4
    # Or pass a count:               --seed 0 --n-seeds 5
    # We will run each seed as an independent training run and then
    # plot mean \pm std across seeds using save_lines_with_bands()
    parser.add_argument("--seed",
                        type=int,
                        default=73,
                        help="Random seed for reproducibility. Use different seeds to check stability."
                        )
    parser.add_argument("--seeds",
                        type=int,
                        nargs="+",
                        default=None,
                        help="Optional explicit list of seeds. Example: --seeds 0 1 2 3 4. If set, overrides --seed/--n-seeds."
    )
    parser.add_argument("--n-seeds",
                        type=int,
                        default=1,
                        help="If --seeds is not provided, run seeds [seed, seed+1, ..., seed+n_seeds-1]."
    )
    parser.add_argument("--band-k",
                        type=float,
                        default=1.0,
                        help="Uncertainty band width in plots: mean ± k*std (1.0 = 1 std)."
    )

    # Epsilon schedule (exploration)
    # start highly exploratory to fill replay, then gradually exploit learned Q-values
    parser.add_argument("--eps-start",
                        type=float,
                        default=1.0,
                        help="Initial epsilon for epsilon-greedy exploration (1.0 = fully random at the start)."
                        )
    parser.add_argument("--eps-end",
                        type=float,
                        default=0.05,
                        help="Final epsilon after annealing (keeps a small amount of exploration)."
                        )
    parser.add_argument("--eps-decay-steps",
                        type=int,
                        default=20_000,
                        help="Number of env steps used to linearly anneal epsilon from eps-start to eps-end."
                        )

    # DQN hyperparameters
    parser.add_argument("--gamma",
                        type=float,
                        default=0.99,
                        help="Discount factor gamma in the TD target: y = r + (1-done) * gamma * bootstrap."
                        )
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="Learning rate for Adam optimizer on the Q-network parameters."
                        )
    parser.add_argument("--batch-size",
                        type=int,
                        default=64,
                        help="Minibatch size sampled from the replay buffer for each gradient update."
                        )
    parser.add_argument("--buffer-size",
                        type=int,
                        default=50_000,
                        help="Replay buffer capacity (number of transitions stored). Older transitions are overwritten."
                        )
    parser.add_argument("--learning-starts",
                        type=int,
                        default=1_000,
                        help="Number of env steps to collect before starting gradient updates (replay warmup)."
                        )
    parser.add_argument("--train-freq",
                        type=int,
                        default=1,
                        help="Train every N env steps after warmup (1 = update every step)."
                        )
    parser.add_argument("--target-update-interval",
                        type=int,
                        default=500,
                        help="Copy online network weights to target network every N env steps."
                        )
    parser.add_argument("--max-grad-norm",
                        type=float,
                        default=10.0,
                        help="Gradient clipping threshold (L2 norm). Set to 0 to disable clipping."
                        )

    # Plots
    parser.add_argument("--plot-dir",
                        type=str,
                        default="assets/plots",
                        help="Where to save plots."
                        )
    return parser.parse_args()


def pad_curves_with_last_value(curves: list[np.ndarray]) -> np.ndarray:
    """
    Pad variable-length curves to the same length by repeating the last value.

    When running multiple seeds, each run can complete a different number of episodes
    within the same total env steps. We want a common x-axis (episode index), so we
    pad shorter runs by holding their last observed value.

    This gives clean mean/std plots without dropping data from longer runs.

    :param curves: List of 1D arrays, one per run/seed (e.g., episode returns or losses).
        Each array can have a different length.
        :type curves: list[np.ndarray]

    :return: A 2D array of shape (n_curves, max_len), where shorter curves are padded
        by repeating their last value up to max_len. Empty curves are filled with NaNs.
        :rtype: np.ndarray
    """
    if len(curves) == 0:
        raise ValueError("pad_curves_with_last_value() received an empty list of curves.")

    max_len = max(c.shape[0] for c in curves)
    out = np.zeros(shape=(len(curves), max_len), dtype=np.float64)

    for i, c in enumerate(curves):
        if c.shape[0] == 0:
            # Extremely rare for CartPole
            # Fill with NaNs so mean/std can still be computed safely if desired
            out[i, :] = np.nan
            continue
        out[i, : c.shape[0]] = c
        out[i, c.shape[0] :] = c[-1]  # repeat last value
    return out


def run_one_seed(
    *,
    seed: int,
    args: argparse.Namespace,
    cfg: DQNConfig,
) -> tuple[list[float], list[float], list[float]]:
    """
    Run one full training session (one seed) and return learning curves.

    We keep this as a separate function so multi-seed experiments become:
    - loop over seeds
    - run training
    - aggregate curves (mean/std) and plot

    :param seed: Random seed for this run. Used to seed Python/NumPy/Torch, the environment
        reset, and the epsilon-greedy RNG so the whole training trajectory is reproducible.
        :type seed: int
    :param args: Parsed command-line arguments controlling the run (env steps, epsilon schedule,output directory, etc.).
        :type args: argparse.Namespace
    :param cfg: DQN agent configuration (network/training hyperparameters and variant flags).
        :type cfg: DQNConfig

    :return: Tuple of (episode_returns, losses, epsilons) collected during the run:
        - episode_returns: one scalar return per completed episode
        - losses: one scalar TD loss per gradient update (after warmup)
        - epsilons: epsilon value used at each environment step
        :rtype: tuple[list[float], list[float], list[float]]
    """
    # Seed everything for reproducibility: env stochasticity, torch init, numpy sampling
    seed_everything(seed=seed, use_torch=True)

    # Separate RNG for epsilon-greedy decisions (keeps randomness explicit and testable)
    rng = np.random.default_rng(seed)

    # Create the environment and read the dimensions needed to build a Q-network
    env = gym.make("CartPole-v1")
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    # The agent contains:
    # online Q-network
    # target Q-network
    # replay buffer
    # optimizer + train_step() that implements DQN/Double-DQN targets
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg, seed=seed)

    # Logging buffers for plots
    # We store full history so we can later save learning curves as images
    episode_returns: list[float] = []
    losses: list[float] = []
    epsilons: list[float] = []

    # Start the first episode
    # Env seeding makes the environment reproducible
    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    episode_idx = 0

    # Main interaction loop:
    # collect experience step-by-step
    # periodically train from replay
    for step in range(1, args.total_steps + 1):
        # 1) Compute the exploration rate for this step to fill replay with diverse transitions
        eps = linear_epsilon(
            step=step,
            start_e=args.eps_start,
            end_e=args.eps_end,
            duration=args.eps_decay_steps,
        )
        epsilons.append(float(eps))

        # 2) Choose an action using epsilon-greedy w.r.t the current online Q-network
        action = agent.select_action(obs=obs, epsilon=eps, rng=rng)

        # 3) Step the environment and build the done flag
        # we treat both terminated and truncated as episode boundaries for simplicity
        # so we never bootstrap across resets
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        # 4) Store the transition in replay -> train on random minibatches (decorrelated data).
        agent.store(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        ep_return += float(reward)

        # 5) Training step (after warmup) at the chosen frequency
        # we don't want to train before replay has some diversity
        if step >= cfg.learning_starts and (step % cfg.train_freq) == 0:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        # 6) Periodically refresh the target network -> makes TD labels stable
        agent.maybe_update_target(env_step=step)

        # 7) Handle end-of-episode bookkeeping
        # we log episode return, reset the env, and start accumulating a new episode
        if done:
            episode_returns.append(ep_return)
            episode_idx += 1
            obs, _ = env.reset(seed=seed + episode_idx)
            ep_return = 0.0
        else:
            # Continue episode
            obs = next_obs

    env.close()
    return episode_returns, losses, epsilons


def main():
    """
    Run a single DQN variant on CartPole and save plots.

    CartPole is intentionally easy and fast, so you can quickly verify:
    - the code runs
    - returns increase over time
    - epsilon decays as expected
    - TD loss stays finite (no NaNs/explosions)
    """
    args = parse_args()
    if args.total_steps <= 0:
        raise ValueError("--total-steps must be > 0")

    os.makedirs(args.plot_dir, exist_ok=True)

    # Decide which seeds to run
    # - if --seeds is passed, we use it directly
    # - otherwise, run a simple range starting from --seed
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds]
    else:
        seeds = [int(args.seed + i) for i in range(int(args.n_seeds))]

    # I will repeat this again:
    # Double DQN changes how targets are computed
    # Dueling changes how Q is represented
    double_dqn = args.algo in {"double", "double_dueling"}
    dueling = args.algo in {"dueling", "double_dueling"}

    # Build the agent config from CLI args -> experiments are fully described by the command
    cfg = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        max_grad_norm=args.max_grad_norm,
        double_dqn=double_dqn,
        dueling=dueling,
    )

    # Multi-seed run
    # For each seed we run a full independent training session and store the learning curves
    # Then we aggregate across seeds and plot mean \pm std using save_lines_with_bands()
    all_episode_returns: list[np.ndarray] = []
    all_losses: list[np.ndarray] = []
    all_epsilons: list[np.ndarray] = []

    for s in seeds:
        episode_returns, losses, epsilons = run_one_seed(seed=s, args=args, cfg=cfg)

        all_episode_returns.append(np.asarray(episode_returns, dtype=np.float64))
        all_losses.append(np.asarray(losses, dtype=np.float64))
        all_epsilons.append(np.asarray(epsilons, dtype=np.float64))

        # Print a tiny per-seed summary so you can compare runs quickly without opening plots
        if episode_returns:
            print(
                f"[seed={s}] Finished: {args.algo} | episodes={len(episode_returns)} | "
                f"last_return={episode_returns[-1]:.1f} | "
                f"mean_last_10={np.mean(episode_returns[-10:]):.1f}"
            )
        else:
            print(f"[seed={s}] Finished: {args.algo} | no episodes completed (increase total-steps)")

    # Plots
    prefix = f"{args.algo}_cartpole"

    # 1) Return curve (learning), aggregated across seeds
    returns_mat = pad_curves_with_last_value(curves=all_episode_returns)
    returns_mean = np.nanmean(returns_mat, axis=0)
    returns_std = np.nanstd(returns_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[returns_mean],
        ys_std=[returns_std],
        labels=[f"return (n={len(seeds)})"],
        title=f"Episode Return ({args.algo}) on CartPole-v1",
        xlabel="Episode",
        ylabel="Return",
        out_path=os.path.join(args.plot_dir, f"{prefix}_returns_mean_std.png"),
        smooth_window=10,
        band_k=float(args.band_k)
    )

    # 2) TD loss curve (training stability), aggregated across seeds
    # Loss can be noisy, so we smooth a bit more
    if any(len(x) > 0 for x in all_losses):
        losses_nonempty = [x for x in all_losses if x.size > 0]
        loss_mat = pad_curves_with_last_value(losses_nonempty)
        loss_mean = np.nanmean(loss_mat, axis=0)
        loss_std = np.nanstd(loss_mat, axis=0)

        save_lines_with_bands(
            ys_mean=[loss_mean],
            ys_std=[loss_std],
            labels=[f"huber loss (n={len(seeds)})"],
            title=f"TD Loss ({args.algo}) on CartPole-v1",
            xlabel="Update",
            ylabel="Huber loss",
            out_path=os.path.join(args.plot_dir, f"{prefix}_loss_mean_std.png"),
            smooth_window=50,
            band_k=float(args.band_k)
        )

    # 3) Epsilon schedule (exploration sanity check), aggregated across seeds
    # epsilon is step-based so lengths should match total_steps
    eps_mat = np.stack(all_epsilons, axis=0)  # (n_seeds, total_steps)
    eps_mean = np.mean(eps_mat, axis=0)
    eps_std = np.std(eps_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[eps_mean],
        ys_std=[eps_std],
        labels=[f"epsilon (n={len(seeds)})"],
        title=f"Epsilon schedule ({args.algo})",
        xlabel="Env step",
        ylabel="Epsilon",
        out_path=os.path.join(args.plot_dir, f"{prefix}_epsilon_mean_std.png"),
        smooth_window=1,
        band_k=float(args.band_k)
    )

    # Single-curve plots
    # If you run with n=1 seed, these are identical to mean/std
    # but if you run multiple seeds you might still want to inspect a single seed curve
    if len(seeds) == 1:
        episode_returns = all_episode_returns[0].tolist()
        losses = all_losses[0].tolist()
        epsilons = all_epsilons[0].tolist()

        # 1) Return curve (learning)
        save_lines(
            ys=[np.asarray(episode_returns, dtype=np.float64)],
            labels=["return"],
            title=f"Episode Return ({args.algo}) on CartPole-v1",
            xlabel="Episode",
            ylabel="Return",
            out_path=os.path.join(args.plot_dir, f"{prefix}_returns.png"),
            smooth_window=10,
        )

        # 2) TD loss curve (training stability)
        if len(losses) > 0:
            save_lines(
                ys=[np.asarray(losses, dtype=np.float64)],
                labels=["huber loss"],
                title=f"TD Loss ({args.algo}) on CartPole-v1",
                xlabel="Update",
                ylabel="Huber loss",
                out_path=os.path.join(args.plot_dir, f"{prefix}_loss.png"),
                smooth_window=50,
            )

        # 3) Epsilon schedule (exploration sanity check)
        save_lines(
            ys=[np.asarray(epsilons, dtype=np.float64)],
            labels=["epsilon"],
            title=f"Epsilon schedule ({args.algo})",
            xlabel="Env step",
            ylabel="Epsilon",
            out_path=os.path.join(args.plot_dir, f"{prefix}_epsilon.png"),
            smooth_window=1,
        )


if __name__ == "__main__":
    main()
