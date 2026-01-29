"""
Run value iteration and policy iteration on a tiny Gridworld MDP.

This script prints:
- the value function as a grid
- the greedy policy as arrows

It also saves plots to assets/plots/:
1) Value heatmap
2) Value heatmap + policy arrows
3) Convergence curve for value iteration (delta vs iteration)

Run from repo root:
    python examples/02_tabular/train_gridworld_dp.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from rl_algorithms_guide.tabular import GridworldMDP, value_iteration, policy_iteration
from rl_algorithms_guide.common.plotting import (
    positions_mask,
    values_to_grid,
    policy_to_grid,
    save_value_heatmap,
    save_value_heatmap_with_policy,
    save_convergence_curve,
)


ARROWS = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←",
}


def format_values(V: np.ndarray, mdp: GridworldMDP) -> str:
    """
    Format a value function as a grid string.
    So you get, for example, something like this as an output:

      T     -1.00  -1.99  -2.97
     -1.00  -1.99  -2.97  -1.99
     -1.99  -2.97  -1.99  -1.00
     -2.97  -1.99  -1.00    T

    :param V: Value array of shape (n_states,).
        :type V: np.ndarray
    :param mdp: Gridworld MDP.
        :type mdp: GridworldMDP

    :return: Multi-line string.
        :rtype: str
    """
    lines = []
    for r in range(mdp.height):
        row_vals = []
        for c in range(mdp.width):
            s = mdp.pos_to_state(r, c)
            if mdp.is_terminal(s):
                row_vals.append("  T   ")
            else:
                row_vals.append(f"{V[s]:6.2f}")
        lines.append(" ".join(row_vals))
    return "\n".join(lines)


def format_policy(policy: np.ndarray, mdp: GridworldMDP) -> str:
    """
    Format a deterministic policy as arrows on the grid.
    So you get, for example, something like this as an output:

    T ← ← ↓
    ↑ ← ← ↓
    ↑ ↑ → ↓
    ↑ → → T

    This is super useful for intuition, since you see the shortest route to a terminal.

    :param policy: Policy array of shape (n_states,).
        :type policy: np.ndarray
    :param mdp: Gridworld MDP.
        :type mdp: GridworldMDP

    :return: Multi-line string.
        :rtype: str
    """
    lines = []
    for r in range(mdp.height):
        row_syms = []
        for c in range(mdp.width):
            s = mdp.pos_to_state(r, c)
            if mdp.is_terminal(s):
                row_syms.append("T")
            else:
                row_syms.append(ARROWS[int(policy[s])])
        lines.append(" ".join(row_syms))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser(description="DP on Gridworld (Value Iteration/Policy Iteration).")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument("--theta", type=float, default=1e-8, help="Convergence threshold.")
    p.add_argument("--out_dir", type=str, default="assets/plots", help="Where to save plots.")
    return p.parse_args()


def main():
    """
    Main entry point.
    Run a tiny DP experiment on a transparent Gridworld MDP.

    What this script does:
    1) Builds a small deterministic 4x4 Gridworld MDP where ((we can change these settings) in Gridworld script):
       - terminal states are at (0,0) and (3,3)
       - every non-terminal step gives reward -1
       - terminal states are absorbing with reward 0

    2) Solves the MDP with two classic Dynamic Programming control methods:
       - Value Iteration: repeatedly applies the Bellman optimality backup to compute V* (optimal value function)
       - Policy Iteration: alternates policy evaluation and greedy improvement until stable

    3) Prints results in a grid form:
       - the value function (V* for value iteration, and V^pi for policy iteration)
       - the corresponding greedy policy as arrows (↑ → ↓ ←), with T marking terminal states

    CLI parameters:
    - --gamma : discount factor
    - --theta : convergence threshold (stopping criterion)
    """
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(f"Running GridworldDP with gamma={args.gamma}, theta={args.theta}")

    mdp = GridworldMDP(height=4, width=4, terminal_states=((0, 0), (3, 3)), step_reward=-1.0)

    # Terminal mask is useful for plotting (mark terminals as 'T' and skip policy arrows there)
    terminal_positions = [mdp.state_to_pos(s) for s in mdp.terminal_states]
    terminal_mask = positions_mask(height=mdp.height, width=mdp.width, positions=terminal_positions)

    # ---------------------------
    # Value Iteration + deltas
    # ---------------------------

    deltas_vi: list[float] = list()
    V_vi, pi_vi = value_iteration(mdp, gamma=args.gamma, theta=args.theta, deltas=deltas_vi)

    print("\n=== Value Iteration ===")
    print("V*:")  # optimal values
    print(format_values(V_vi, mdp))
    print("\nGreedy policy:")
    print(format_policy(pi_vi, mdp))

    V_vi_grid = values_to_grid(V=V_vi, height=mdp.height, width=mdp.width, state_to_pos=mdp.state_to_pos)
    pi_vi_grid = policy_to_grid(
        policy=pi_vi,
        height=mdp.height,
        width=mdp.width,
        state_to_pos=mdp.state_to_pos,
        terminal_mask=terminal_mask,
    )

    save_value_heatmap(
        V_grid=V_vi_grid,
        out_path=out_dir/"gridworld_value_iteration_values.png",
        title="Gridworld (Value Iteration) - V* heatmap",
        terminal_mask=terminal_mask,
        annotate=True,
    )

    save_value_heatmap_with_policy(
        V_grid=V_vi_grid,
        policy_grid=pi_vi_grid,
        out_path=out_dir/"gridworld_value_iteration_values_policy.png",
        title="Gridworld (Value Iteration) - V* + greedy policy",
        terminal_mask=terminal_mask,
    )

    save_convergence_curve(
        deltas=deltas_vi,
        out_path=out_dir/"gridworld_value_iteration_convergence.png",
        title="Value Iteration convergence (max |V_{k+1} - V_k|)",
        logy=True,
    )

    # ---------------------------
    # Policy Iteration
    # ---------------------------

    V_pi, pi_pi = policy_iteration(mdp, gamma=args.gamma, theta=args.theta)

    print("\n=== Policy Iteration ===")
    print("V^pi:")
    print(format_values(V_pi, mdp))
    print("\nFinal policy:")
    print(format_policy(pi_pi, mdp))

    V_pi_grid = values_to_grid(V=V_pi, height=mdp.height, width=mdp.width, state_to_pos=mdp.state_to_pos)
    pi_pi_grid = policy_to_grid(
        policy=pi_pi,
        height=mdp.height,
        width=mdp.width,
        state_to_pos=mdp.state_to_pos,
        terminal_mask=terminal_mask,
    )

    save_value_heatmap(
        V_grid=V_pi_grid,
        out_path=out_dir/"gridworld_policy_iteration_values.png",
        title="Gridworld (Policy Iteration) - V^pi heatmap",
        terminal_mask=terminal_mask,
        annotate=True,
    )

    save_value_heatmap_with_policy(
        V_grid=V_pi_grid,
        policy_grid=pi_pi_grid,
        out_path=out_dir/"gridworld_policy_iteration_values_policy.png",
        title="Gridworld (Policy Iteration) - V^pi + final policy",
        terminal_mask=terminal_mask,
    )

    print(f"\nSaved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
