"""
Compare ε-greedy vs UCB on a simple k-armed Gaussian bandit.

This script is meant to be run from the repo root, after installing the package:

    pip install -e .
    python examples/01_bandits/train_compare.py

It saves plots in:
    assets/plots/
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rl_algorithms_guide.common.seeding import seed_everything
from rl_algorithms_guide.common.plotting import save_lines
from rl_algorithms_guide.bandits.bandit_envs import GaussianBandit
from rl_algorithms_guide.bandits.epsilon_greedy import EpsilonGreedyAgent, DecayingEpsilonGreedyAgent
from rl_algorithms_guide.bandits.ucb import UCBAgent


@dataclass
class Curve:
    """
    Small container for averaged learning curves.

    :param avg_reward: Average reward at each step. Shape: (steps,).
        :type avg_reward: np.ndarray
    :param pct_optimal: Percentage of optimal-action selections at each step. Shape: (steps,).
        :type pct_optimal: np.ndarray
    """
    avg_reward: np.ndarray
    pct_optimal: np.ndarray


def run_agent_on_bandit(*, agent, bandit: GaussianBandit, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a single agent on a single bandit for a fixed number of steps.

    The agent is expected to expose:
    - select_action() -> int
    - update(action: int, reward: float) -> None

    :param agent: Agent instance (ε-greedy or UCB).
        :type agent: object
    :param bandit: Bandit environment to interact with.
        :type bandit: GaussianBandit
    :param steps: Number of interaction steps.
        :type steps: int

    :return: (rewards, optimal_picks)
        - rewards: reward at each step (shape: (steps,))
        - optimal_picks: 1.0 if optimal arm was chosen, else 0.0 (shape: (steps,))
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    info = bandit.reset()

    rewards = np.zeros(steps, dtype=np.float64)
    optimal_picks = np.zeros(steps, dtype=np.float64)

    for t in range(steps):
        action = agent.select_action()
        reward = bandit.step(action)
        agent.update(action, reward)

        rewards[t] = reward
        optimal_picks[t] = 1.0 if action == info.best_action else 0.0

    return rewards, optimal_picks


def run_experiment(
    *,
    k: int,
    steps: int,
    runs: int,
    eps_list: list[float],
    include_eps_decay: bool,
    eps_start: float,
    eps_end: float,
    eps_decay_steps: int,
    eps_schedule: str,
    ucb_c: float,
    bandit_std: float,
    seed: int,
) -> dict[str, Curve]:
    """
    Compare ε-greedy (multiple ε values), decaying-ε-greedy (optional), and UCB on a k-armed Gaussian bandit.

    What "runs" means here:
    - Each run samples a new set of true means q_star (a new bandit instance).
    - For that same q_star, we evaluate each method (ε-greedy variants, optional decay and UCB).
    - We average curves across all runs to get smoother, more reliable results.

    :param k: Number of arms.
        :type k: int
    :param steps: Steps per run.
        :type steps: int
    :param runs: Number of independent runs (averaged).
        :type runs: int
    :param eps_list: Epsilon values to compare for fixed ε-greedy.
        :type eps_list: list[float]
    :param include_eps_decay: Whether to include the decaying ε-greedy variant.
        :type include_eps_decay: bool
    :param eps_start: Starting ε for the decaying ε-greedy agent.
        :type eps_start: float
    :param eps_end: Final/minimum ε for the decaying ε-greedy agent.
        :type eps_end: float
    :param eps_decay_steps: Decay steps (linear) or time constant (exp) for the decaying ε schedule.
        :type eps_decay_steps: int
    :param eps_schedule: Decay schedule ("linear" or "exp").
        :type eps_schedule: str
    :param ucb_c: Exploration coefficient for UCB.
        :type ucb_c: float
    :param bandit_std: Reward noise standard deviation.
        :type bandit_std: float
    :param seed: Master seed (controls q_star sampling and internal RNGs).
        :type seed: int

    :return: Mapping {label -> Curve} with averaged curves for each method.
        :rtype: dict[str, Curve]

    An example of what this function returns:
    {
      "eps=0": Curve(avg_reward=[...], pct_optimal=[...]),
      "eps=0.01": Curve(...),
      "eps=0.1": Curve(...),
      "eps_decay(linear,1->0.1)": Curve(...),
      "ucb(c=2)": Curve(...),
    }
    """
    rng = np.random.default_rng(seed)

    labels: list[str] = [f"eps={eps:g}" for eps in eps_list]

    decay_label = f"eps_decay({eps_schedule},{eps_start:g}->{eps_end:g})"
    if include_eps_decay:
        labels.append(decay_label)

    labels.append(f"ucb(c={ucb_c:g})")

    reward_sums: dict[str, np.ndarray] = {lab: np.zeros(steps, dtype=np.float64) for lab in labels}
    optimal_sums: dict[str, np.ndarray] = {lab: np.zeros(steps, dtype=np.float64) for lab in labels}

    for _ in range(runs):
        # Sample a new bandit mean vector (q_star) for this run
        # Every method is evaluated on the same q_star within this run (fair comparison)
        q_star = rng.uniform(low=-1.0, high=1.0, size=k).astype(np.float64)

        # Evaluate fixed ε-greedy variants
        for eps in eps_list:
            label = f"eps={eps:g}"

            env = GaussianBandit(
                k=k,
                std=bandit_std,
                q_star=q_star,
                seed=int(rng.integers(low=0, high=1_000_000)),
            )
            agent = EpsilonGreedyAgent(
                k=k,
                epsilon=eps,
                seed=int(rng.integers(low=0, high=1_000_000)),
            )

            rewards, optimal = run_agent_on_bandit(agent=agent, bandit=env, steps=steps)
            reward_sums[label] += rewards
            optimal_sums[label] += optimal

        # Evaluate decay ε-greedy
        if include_eps_decay:
            env = GaussianBandit(
                k=k,
                std=bandit_std,
                q_star=q_star,
                seed=int(rng.integers(low=0, high=1_000_000)),
            )
            agent = DecayingEpsilonGreedyAgent(
                k=k,
                epsilon_start=eps_start,
                epsilon_end=eps_end,
                decay_steps=eps_decay_steps,
                schedule=eps_schedule,
                seed=int(rng.integers(low=0, high=1_000_000)),
            )

            rewards, optimal = run_agent_on_bandit(agent=agent, bandit=env, steps=steps)
            reward_sums[decay_label] += rewards
            optimal_sums[decay_label] += optimal

        # Evaluate UCB
        ucb_label = f"ucb(c={ucb_c:g})"
        env = GaussianBandit(
            k=k,
            std=bandit_std,
            q_star=q_star,
            seed=int(rng.integers(low=0, high=1_000_000)),
        )
        agent = UCBAgent(k=k, c=ucb_c)

        rewards, optimal = run_agent_on_bandit(agent=agent, bandit=env, steps=steps)
        reward_sums[ucb_label] += rewards
        optimal_sums[ucb_label] += optimal

    results: dict[str, Curve] = {}
    for label in labels:
        avg_reward = reward_sums[label] / runs
        pct_optimal = (optimal_sums[label] / runs) * 100.0
        results[label] = Curve(avg_reward=avg_reward, pct_optimal=pct_optimal)

    return results


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    :return: Parsed arguments namespace.
        :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Compare ε-greedy and UCB on k-armed Gaussian bandits."
    )
    parser.add_argument("--k", type=int, default=10, help="Number of arms.")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per run.")
    parser.add_argument("--runs", type=int, default=2000, help="Number of runs to average.")

    # Fixed ε-greedy
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.1],
        help="Epsilon values for fixed ε-greedy (space-separated).",
    )

    # Decaying ε-greedy
    parser.add_argument("--include-eps-decay", action="store_true", help="Include decaying ε-greedy in the comparison.")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Starting ε for decaying ε-greedy.")
    parser.add_argument("--eps-end", type=float, default=0.1, help="Final/min ε for decaying ε-greedy.")
    parser.add_argument("--eps-decay-steps", type=int, default=500, help="Decay steps for decaying ε-greedy.")
    parser.add_argument("--eps-schedule", type=str, default="linear", choices=["linear", "exp"], help="Decay schedule for ε (linear or exp).")

    # UCB
    parser.add_argument("--ucb-c", type=float, default=2.0, help="UCB exploration coefficient.")
    parser.add_argument("--std", type=float, default=1.0, help="Reward noise standard deviation.")

    # Seed and plotting
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window for plots (1 = no smoothing).",
    )
    return parser.parse_args()


def main():
    """
    Run the experiment and save plots to assets/plots/.

    :return: None.
        :rtype: None
    """
    args = parse_args()
    seed_everything(args.seed)

    results = run_experiment(
        k=args.k,
        steps=args.steps,
        runs=args.runs,
        eps_list=list(args.eps),
        include_eps_decay=args.include_eps_decay,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        eps_schedule=args.eps_schedule,
        ucb_c=args.ucb_c,
        bandit_std=args.std,
        seed=args.seed,
    )

    out_dir = Path("assets/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(results.keys())
    avg_rewards = [results[label].avg_reward for label in labels]
    pct_opts = [results[label].pct_optimal for label in labels]

    save_lines(
        ys=avg_rewards,
        labels=labels,
        title="Bandits: Average reward over time",
        xlabel="Step",
        ylabel="Average reward",
        out_path=out_dir / "bandits_avg_reward.png",
        smooth_window=args.smooth,
    )

    save_lines(
        ys=pct_opts,
        labels=labels,
        title="Bandits: % optimal action over time",
        xlabel="Step",
        ylabel="% optimal action",
        out_path=out_dir / "bandits_pct_optimal.png",
        smooth_window=args.smooth,
    )

    print("Saved plots in:")
    print(f"  {out_dir.resolve()}")
    print("  - bandits_avg_reward.png")
    print("  - bandits_pct_optimal.png")


if __name__ == "__main__":
    main()
