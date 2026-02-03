"""Train DQN-style agents on LunarLander-v2.

Before using LunarLander-v2, read the Gymnasium environment documentation.
This is valid for all envs you try to use in a project. You should know the env before using it.

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
import gymnasium as gym
import numpy as np
import os

from rl_algorithms_guide.common.seeding import seed_everything
from rl_algorithms_guide.common.plotting import save_lines, save_lines_with_bands
from rl_algorithms_guide.dqn.dqn import DQNAgent, DQNConfig, linear_epsilon


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    LunarLander is harder and noisier than CartPole, so this script exposes
    more practical defaults (bigger buffer, longer warmup, smaller LR).

    :return: Parsed args namespace.
        :rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser(description="Train DQN variants on LunarLander-v2 (discrete).")

    p.add_argument("--env-id",
                   type=str,
                   default="LunarLander-v2",
                   help="Gymnasium environment id")
    p.add_argument("--total-steps",
                   type=int,
                   default=500_000,
                   help="Total environment interaction steps to run (training happens during these steps)."
                   )
    # Algorithm variant: Double affects targets, Dueling affects network architecture
    p.add_argument("--algo",
                   type=str,
                   default="dqn",
                   choices=["dqn", "double", "dueling", "double_dueling"],
                   help=("Which DQN variant to run. "
                         "'dqn' = vanilla DQN, "
                         "'double' = Double DQN targets, "
                         "'dueling' = dueling Q-network, "
                         "'double_dueling' = both."
                         )
                   )
    # Multi-seed support
    # Either pass an explicit list:   --seeds 0 1 2 3 4
    # Or pass a count:               --seed 0 --n-seeds 5
    # We will run each seed as an independent training run and then
    # plot mean \pm std across seeds using save_lines_with_bands()
    p.add_argument("--seed",
                   type=int,
                   default=73,
                   help="Random seed for reproducibility. Use different seeds to check stability."
                   )
    p.add_argument("--seeds",
                   type=int,
                   nargs="+",
                   default=None,
                   help="Optional explicit list of seeds. Example: --seeds 0 1 2 3 4. If set, overrides --seed/--n-seeds."
                   )
    p.add_argument("--n-seeds",
                   type=int,
                   default=1,
                   help="If --seeds is not provided, run seeds [seed, seed+1, ..., seed+n_seeds-1]."
                   )

    # DQN hyperparams (reasonable CPU defaults; LunarLander needs more than CartPole)
    # Why these differ from CartPole:
    # smaller lr (more stable on a harder env)
    # larger batch/buffer (more diverse replay, smoother updates)
    # longer warmup (fill replay before learning)
    p.add_argument("--gamma",
                   type=float,
                   default=0.99,
                   help="Discount factor gamma in the TD target: y = r + (1-done) * gamma * bootstrap."
                   )
    p.add_argument("--lr",
                   type=float,
                   default=1e-4,
                   help="Learning rate for Adam optimizer on the Q-network parameters."
                   )
    p.add_argument("--batch-size",
                   type=int,
                   default=128,
                   help="Minibatch size sampled from the replay buffer for each gradient update."
                   )
    p.add_argument("--buffer-size",
                   type=int,
                   default=200_000,
                   help="Replay buffer capacity (number of transitions stored). Older transitions are overwritten."
                   )
    p.add_argument("--learning-starts",
                   type=int,
                   default=10_000,
                   help="Number of env steps to collect before starting gradient updates (replay warmup)."
                   )
    p.add_argument("--train-freq",
                   type=int,
                   default=1,
                   help="Train every N env steps after warmup (1 = update every step)."
                   )
    p.add_argument("--target-update-interval",
                   type=int,
                   default=2_000,
                   help="Copy online network weights to target network every N env steps."
                   )
    p.add_argument("--max-grad-norm",
                   type=float,
                   default=10.0,
                   help="Gradient clipping threshold (L2 norm). Set to 0 to disable clipping."
                   )

    # Epsilon schedule
    # LunarLander needs sustained exploration, so we decay more slowly than CartPole
    p.add_argument("--eps-start",
                   type=float,
                   default=1.0,
                   help="Initial epsilon for epsilon-greedy exploration (1.0 = fully random at the start)."
                   )
    p.add_argument("--eps-end",
                   type=float,
                   default=0.05,
                   help="Final epsilon after annealing (keeps a small amount of exploration)."
                   )
    p.add_argument("--eps-decay-steps",
                   type=int,
                   default=200_000,
                   help="Number of env steps used to linearly anneal epsilon from eps-start to eps-end."
                   )

    # Plots
    p.add_argument("--band-k",
                   type=float,
                   default=1.0,
                   help="Uncertainty band width in plots: mean ± k*std (1.0 = 1 std)."
                   )
    p.add_argument("--smooth",
                   type=int,
                   default=10,
                   help="Moving-average window for plots (1 = no smoothing)."
                   )
    p.add_argument("--outdir",
                   type=str,
                   default="assets/plots",
                   help="Where to save plots."
                   )

    return p.parse_args()


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
            # Extremely rare
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
    # Global seeding for reproducibility
    seed_everything(seed=seed, use_torch=True)

    # Separate RNG for epsilon-greedy action randomness (keeps exploration randomness explicit)
    rng = np.random.default_rng(seed + 12345)

    # Create the environment and wrap it to automatically record episode returns
    # LunarLander episodes are long and noisy, so using wrapper stats is convenient and robust
    # Chck wrappers on gym documentation
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # LunarLander observations are already 1D, but we keep this generic
    # obs_dim = product of shape (works for both vectors and flattened arrays)
    obs_dim = int(np.prod(env.observation_space.shape))
    # DQN only supports discrete actions because the network outputs one Q per action
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("This script expects a discrete action space.")
    n_actions = int(env.action_space.n)

    # Create the agent: Q-network + target network + replay buffer + optimizer
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg, seed=seed)

    # Logging buffers for plots:
    # ep_returns: one point per episode (learning curve)
    # losses: one point per gradient update (training stability)
    # epsilons: one point per env step (exploration schedule sanity check)
    ep_returns: list[float] = []
    losses: list[float] = []
    epsilons: list[float] = []

    # Reset environment to get the initial observation
    # Env seeding makes the environment reproducible
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    episode_idx = 0

    # Main loop: one iteration = one environment interaction step
    for step in range(args.total_steps):
        # 1) Compute epsilon for this step
        # start exploratory to populate replay
        # decay to exploit learned Q-values
        eps = linear_epsilon(step=step, start_e=args.eps_start, end_e=args.eps_end, duration=args.eps_decay_steps)
        epsilons.append(eps)

        # 2) Choose an action with epsilon-greedy policy
        action = agent.select_action(obs=obs, epsilon=eps, rng=rng)

        # 3) Step the environment and build done flag
        # As in CartPole, we treat (terminated OR truncated) as an episode boundary
        # and do not bootstrap across resets (simple and stable)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        # 4) Convert next_obs to a flat float32 vector for the replay buffer/network
        next_obs_arr = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        # 5) Store the transition in replay
        agent.store(obs=obs, action=action, reward=float(reward), next_obs=next_obs_arr, done=done)

        # Advance state for the next loop iteration
        obs = next_obs_arr

        # 6) One gradient update per step (training update after warmup), controlled by train_freq
        # training from random minibatches makes updates less correlated and more stable
        if step >= cfg.learning_starts and (step % cfg.train_freq == 0):
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        # 7) Periodically copy online -> target network for stable TD targets
        agent.maybe_update_target(env_step=step)

        # 8) Episode bookkeeping -> record return and reset when done
        # RecordEpisodeStatistics puts episode info in `info["episode"]` at terminal steps
        if done:
            if "episode" in info and "r" in info["episode"]:
                ep_returns.append(float(info["episode"]["r"]))
            episode_idx += 1
            obs, _ = env.reset(seed=seed + episode_idx)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    env.close()
    return ep_returns, losses, epsilons


def main():
    """
    Train a DQN variant on LunarLander-v2 and save learning curves.

    This follows the same DQN loop as the theory:
    1) interact with env using epsilon-greedy
    2) store transitions in replay
    3) after warmup, train from random minibatches with TD targets
    4) periodically update the target network
    """
    args = parse_args()
    if args.total_steps <= 0:
        raise ValueError("--total-steps must be > 0")

    os.makedirs(args.outdir, exist_ok=True)

    # Decide which seeds to run
    # - if --seeds is passed, we use it directly
    # - otherwise, run a simple range starting from --seed
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds]
    else:
        seeds = [int(args.seed + i) for i in range(int(args.n_seeds))]

    double = args.algo in {"double", "double_dueling"}
    dueling = args.algo in {"dueling", "double_dueling"}

    # Build the DQN config for this run
    # We use slightly larger hidden layers for LunarLander than CartPole
    cfg = DQNConfig(
        gamma=float(args.gamma),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        buffer_size=int(args.buffer_size),
        learning_starts=int(args.learning_starts),
        train_freq=int(args.train_freq),
        target_update_interval=int(args.target_update_interval),
        max_grad_norm=float(args.max_grad_norm),
        double_dqn=bool(double),
        dueling=bool(dueling),
        hidden_sizes=(256, 256)
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

        # Print a tiny per-seed summary
        if episode_returns:
            print(
                f"[seed={s}] Finished: {args.algo} | episodes={len(episode_returns)} | "
                f"last_return={episode_returns[-1]:.1f} | "
                f"mean_last_10={np.mean(episode_returns[-10:]):.1f}"
            )
        else:
            print(f"[seed={s}] Finished: {args.algo} | no episodes completed (increase total-steps)")

    # Save plots
    prefix = f"{args.algo}_lunarlander"

    # 1) Returns: variable episode count -> pad by repeating last value
    returns_mat = pad_curves_with_last_value(curves=all_episode_returns)
    returns_mean = np.nanmean(returns_mat, axis=0)
    returns_std = np.nanstd(returns_mat, axis=0)

    save_lines_with_bands(
        ys_mean=[returns_mean],
        ys_std=[returns_std],
        labels=[f"return (n={len(seeds)})"],
        title=f"Episode Return ({args.algo}) on LunarLander-v2",
        xlabel="Episode",
        ylabel="Return",
        out_path=os.path.join(args.outdir, f"{prefix}_returns_mean_std.png"),
        smooth_window=int(args.smooth),
        band_k=float(args.band_k)
    )

    # 2) Loss: some runs might have 0 losses if something went wrong early -> filter empties
    if any(x.size > 0 for x in all_losses):
        losses_nonempty = [x for x in all_losses if x.size > 0]
        loss_mat = pad_curves_with_last_value(losses_nonempty)
        loss_mean = np.nanmean(loss_mat, axis=0)
        loss_std = np.nanstd(loss_mat, axis=0)

        save_lines_with_bands(
            ys_mean=[loss_mean],
            ys_std=[loss_std],
            labels=[f"huber loss (n={len(seeds)})"],
            title=f"TD Loss ({args.algo}) on LunarLander-v2",
            xlabel="Update",
            ylabel="Huber loss",
            out_path=os.path.join(args.outdir, f"{prefix}_loss_mean_std.png"),
            smooth_window=50,
            band_k=float(args.band_k)
        )

    # 3) Epsilon: fixed length = total_steps -> stack directly
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
        out_path=os.path.join(args.outdir, f"{prefix}_epsilon_mean_std.png"),
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

        # 1) Episode returns: main learning curve (smoothed because LunarLander is noisy)
        save_lines(
            ys=[np.asarray(episode_returns, dtype=np.float64)],
            labels=[f"{args.algo}"],
            title=f"LunarLander-v2 returns ({args.algo})",
            xlabel="episode",
            ylabel="return",
            out_path=os.path.join(args.outdir, f"{prefix}_returns.png"),
            smooth_window=int(args.smooth),
        )

        # 2) TD loss: training stability diagnostic
        # Loss is naturally noisy
        # smoothing helps you see explosions/plateaus
        if len(losses) > 0:
            save_lines(
                ys=[np.asarray(losses, dtype=np.float64)],
                labels=[f"{args.algo}"],
                title=f"LunarLander-v2 TD loss ({args.algo})",
                xlabel="update",
                ylabel="Huber loss",
                out_path=os.path.join(args.outdir, f"{prefix}_loss.png"),
                smooth_window=max(1, int(args.smooth)),
            )

        # 3) Epsilon schedule: sanity check that exploration decays as intended
        save_lines(
            ys=[np.asarray(epsilons, dtype=np.float64)],
            labels=["epsilon"],
            title="Epsilon schedule",
            xlabel="env step",
            ylabel="epsilon",
            out_path=os.path.join(args.outdir, f"{prefix}_epsilon.png"),
            smooth_window=1,
        )


if __name__ == "__main__":
    main()
