"""
Train SARSA vs Expected SARSA vs Q-learning on CliffWalking (tabular control).

Run from repo root:
    python examples/02_tabular/train_cliffwalking.py

It saves plots to:
    assets/plots/tabular_cliffwalking_returns.png
    assets/plots/tabular_cliffwalking_cliff_rate.png
    assets/plots/tabular_cliffwalking_steps.png
    assets/plots/tabular_cliffwalking_eval_returns.png

Tip:
    Use --print-policies to print the greedy policies learned by each agent.
    This is often the most intuitive way to see the difference between SARSA and Q-learning.
    Also, it is useful to study gymnasium's documentation, since we will import the environment from there.

About the "cliff rate" plot:
    In CliffWalking, stepping into the cliff typically yields a reward of -100.
    We count an episode as "fell off" if it sees at least one reward <= -100.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from rl_algorithms_guide.common.plotting import save_lines_with_bands
from rl_algorithms_guide.common.seeding import seed_everything
from rl_algorithms_guide.common.gym_utils import make_first_available
from rl_algorithms_guide.tabular import SARSAgent, QLearningAgent, ExpectedSARSAgent


ARROWS = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←",
}


def is_cliff_reward(reward: float) -> bool:
    """
    Detect whether a reward likely corresponds to stepping into the cliff.

    In the standard CliffWalking setup:
    - normal step: -1
    - cliff: -100

    We use a <= check to be tolerant to float casting and minor differences.

    :param reward: Reward from env.step(action).
        :type reward: float

    :return: True if the reward looks like a cliff penalty.
        :rtype: bool
    """
    return float(reward) <= -100.0


def greedy_policy_from_Q(Q: np.ndarray) -> np.ndarray:
    """
    Build a greedy (deterministic) policy from a tabular Q table.
    It returns, for each state, the action with max Q.

    :param Q: Q-table of shape (n_states, n_actions).
        :type Q: np.ndarray

    :return: Policy array of shape (n_states,), each entry is an action index.
        :rtype: np.ndarray
    """
    return np.argmax(Q, axis=1).astype(np.int64)


def format_cliffwalking_policy(env, policy: np.ndarray, debug: bool = True) -> str:
    """
    Format a CliffWalking greedy policy as a grid of arrows.
    It prints a policy like a grid with:
        - 'S' start
        - 'G' goal
        - 'C' cliff
        - arrows elsewhere

    We try to read env-specific attributes (shape, start_state, terminal_state, cliff).
    If they are not available (different gym versions), we fall back to a generic print.
    We can get something like this as output:
        S → → → → → → → → → → ↓
        ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↓
        ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↓
        C C C C C C C C C C C G

    :param env: Gymnasium CliffWalking environment (or compatible).
        :type env: object
    :param policy: Policy array of shape (n_states,).
        :type policy: np.ndarray
    :param debug: If True it will show the attributes that exist in the environment.
        :type debug: bool

    :return: Multi-line string of the policy grid.
        :rtype: str
    """
    unwrapped = getattr(env, "unwrapped", env)  # Gymnasium environments are often wrapped

    # what we try to read
    wanted = ["shape", "start_state_index", "terminal_state", "cliff"]
    # what we get from the env
    got = {name: getattr(unwrapped, name, None) for name in wanted}
    # what we don't get
    missing = [name for name, val in got.items() if val is None]

    goal_state = None
    cliff_ = None

    # If any key attribute is missing, you cannot reliably label "S/G/C" in the right places, or even know how many columns to wrap lines
    if missing:
        if set(missing) == {"terminal_state", "cliff"} and getattr(unwrapped, "P", None) is not None:
            goal_state, cliff_ = infer_goal_and_cliff_from_P(unwrapped)
        else:
            if debug:
                # show what's missing + show the attributes that DO exist (helpful to rename)
                available = [a for a in dir(unwrapped) if not a.startswith("_")]
                print("[format_cliffwalking_policy] Missing attributes:", missing)
                print("[format_cliffwalking_policy] Available attributes (non-private):")
                print("  " + ", ".join(available[:200]) + (" ..." if len(available) > 200 else ""))
            n = min(len(policy), 42)  # 42 is just to keep the print readable
            return "Policy (first states): " + " ".join(ARROWS[int(a)] for a in policy[:n])  # just a fallback for when the env doesn't expose the internal info needed to draw a real grid
            # instead of crashing, it prints something simpler

    shape = getattr(unwrapped, "shape", None)  # (n_rows, n_cols)
    start_state = getattr(unwrapped, "start_state_index", None)  # integer index for the start cell
    terminal_state = getattr(unwrapped, "terminal_state", goal_state)  # integer index for the goal cell
    cliff = getattr(unwrapped, "cliff", cliff_)  # usually a boolean array shaped like the grid (or flattened) marking cliff cells

    n_rows, n_cols = int(shape[0]), int(shape[1])
    lines = list()  # it will store the grid's rows to print later

    if len(policy) != n_rows * n_cols:
        return "Policy has wrong size for env grid."

    if terminal_state is None:
        # Fallback goal if still None (works for standard CliffWalking)
        # I am adding this because it is happening
        terminal_state = n_rows * n_cols - 1  # bottom-right

    # states in which cliff is True (or nonzero)
    if cliff is None:
        cliff_set = set()
    elif isinstance(cliff, np.ndarray):
        cliff_set = set(int(s) for s in np.flatnonzero(cliff))
    else:
        cliff_set = set(int(s) for s in cliff)  # inferred set[int]

    for r in range(n_rows):
        row_syms = list()
        for c in range(n_cols):
            s = r * n_cols + c  # state

            if s == int(start_state):
                row_syms.append("S")
            elif terminal_state is not None and s == int(terminal_state):
                row_syms.append("G")
            elif s in cliff_set:
                row_syms.append("C")
            else:
                row_syms.append(ARROWS[int(policy[s])])
        lines.append(" ".join(row_syms))

    return "\n".join(lines)

def infer_goal_and_cliff_from_P(unwrapped, cliff_reward_threshold: float = -100.0) -> tuple[int | None, set[int]]:
    """
    Infer the goal (terminal) state index and the set of cliff cell indices from a Gymnasium Toy-Text env.
    This is a compatibility helper for Gymnasium versions where the CliffWalking environment does NOT expose
    convenient attributes like terminal_state and cliff.
    We use `env.P`, the explicit transition model used by Toy-Text environments:

        P[s][a] -> list of (prob, next_state, reward, done)

    The function tries to infer:
    1) Goal state:
       - we look for an absorbing terminal state: for every action, all outcomes have done=True and next_state == s
       - if none is found, we fall back to the standard CliffWalking goal: bottom-right cell (n_rows*n_cols - 1)

    2) Cliff cells:
       - in standard CliffWalking, stepping into the cliff yields a large negative reward (typically -100)
         and the env usually teleports you back to start
       - if we used next_state directly, we would mistakenly label the start state as cliff
       - instead, we compute the intended next grid cell from (s, a) using the grid geometry, and mark that cell as cliff
         whenever the transition reward is below the provided threshold

    Notes:
    - We remove start_state_index and goal from the cliff set to avoid mislabeling
    - This assumes the env is a gridworld with actions {0: up, 1: right, 2: down, 3: left}, like CliffWalking

    :param unwrapped: The environment (preferably env.unwrapped) exposing Toy-Text fields like P, shape,
        and optionally nS and start_state_index.
        :type unwrapped: object
    :param cliff_reward_threshold: Any transition reward <= this threshold is treated as a cliff penalty
        Default matches standard CliffWalking (-100)
        :type cliff_reward_threshold: float

    :return: (goal, cliff_cells)
        - goal: inferred terminal/goal state index, or None if inference is impossible
        - cliff_cells: set of state indices corresponding to cliff grid cells, or None if inference is impossible
        :rtype: tuple[int | None, set[int]]
    """
    P = getattr(unwrapped, "P", None)
    shape = getattr(unwrapped, "shape", None)
    start = getattr(unwrapped, "start_state_index", None)

    if P is None or shape is None:
        return None, set()

    n_rows, n_cols = int(shape[0]), int(shape[1])

    nS = getattr(unwrapped, "nS", None)
    if nS is None:
        nS = len(P)

    # Guess goal -> absorbing terminal (done=True and next_state==s for all actions)
    goal = None
    for s in P.keys():
        absorbing_terminal = True
        for a in P[s].keys():
            # In toy-text P[s][a] is list of (prob, ns, r, done)
            for (prob, ns, r, done) in P[s][a]:
                if (not done) or (int(ns) != int(s)):
                    absorbing_terminal = False
                    break
            if not absorbing_terminal:
                break
        if absorbing_terminal:
            goal = int(s)
            break

    # Fallback goal for standard CliffWalking -> bottom-right
    if goal is None:
        goal = n_rows * n_cols - 1

    # Infer cliff cells from intended move direction
    def intended_next_state(s: int, a: int) -> int:
        r, c = divmod(s, n_cols)
        if a == 0:  # up
            r2, c2 = max(0, r - 1), c
        elif a == 1:  # right
            r2, c2 = r, min(n_cols - 1, c + 1)
        elif a == 2:  # down
            r2, c2 = min(n_rows - 1, r + 1), c
        elif a == 3:  # left
            r2, c2 = r, max(0, c - 1)
        else:
            r2, c2 = r, c
        return r2 * n_cols + c2

    cliff_cells = set()

    # P is usually dict: P[s][a] -> list[(prob, ns, r, done)]
    for s, actions in P.items():
        for a, outcomes in actions.items():
            for (prob, ns, r, done) in outcomes:
                if float(r) <= cliff_reward_threshold:
                    cliff_cells.add(intended_next_state(int(s), int(a)))

    # Safety, never label start or goal as cliff
    if start is not None:
        cliff_cells.discard(int(start))
    cliff_cells.discard(int(goal))

    return goal, cliff_cells


def run_sarsa_episode(env, agent: SARSAgent, max_steps: int) -> tuple[float, bool, int]:
    """
    Run one SARSA episode with online TD updates.
    SARSA is on-policy and its update needs the next action A_{t+1}:
        Q(S_t,A_t) <- Q(S_t,A_t) + alpha * [R_{t+1} + gamma * Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]

    So the loop is:
        select A
        step -> observe S', R
        select A' (from the same ε-greedy behaviour policy)
        update using (S,A,R,S',A')

    :param env: Gymnasium environment with discrete spaces.
        :type env: object
    :param agent: SARSA agent.
        :type agent: SARSAgent
    :param max_steps: Max steps per episode (safety cap).
        :type max_steps: int

    :return: (episode_return, fell_off_cliff, steps)
        :rtype: tuple[float, bool, int]
    """
    state, _ = env.reset()
    action = agent.select_action(state)

    total_reward = 0.0
    fell_off = False
    steps = 0

    for _ in range(max_steps):
        steps += 1
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        r = float(reward)
        total_reward += r
        if is_cliff_reward(r):
            fell_off = True

        if done:
            # When done=True we do NOT bootstrap from Q(next_state, next_action)
            agent.update(state, action, r, next_state, next_action=0, done=True)
            break

        next_action = agent.select_action(next_state)
        agent.update(state, action, r, next_state, next_action, done=False)  # this is the main difference between SARSA and the other two
        # SARSA need next_action to update the agent, while the Expected SARSA ans Q-learning no
        # this is the reason why SARSA's loop is different

        state, action = next_state, next_action

    return total_reward, fell_off, steps


def run_expected_sarsa_episode(env, agent: ExpectedSARSAgent, max_steps: int) -> tuple[float, bool, int]:
    """
    Run one Expected SARSA episode with online TD updates.
    Expected SARSA is also on-policy, but instead of using the sampled next action A_{t+1},
    it uses the expected value under the ε-greedy behaviour policy:

        E_{a~mu}[Q(S_{t+1}, a)]

    That is why it doesn't need next_action in the runner.

    :param env: Gymnasium environment with discrete spaces.
        :type env: object
    :param agent: Expected SARSA agent.
        :type agent: ExpectedSARSAgent
    :param max_steps: Max steps per episode (safety cap).
        :type max_steps: int

    :return: (episode_return, fell_off_cliff, steps)
        :rtype: tuple[float, bool, int]
    """
    state, _ = env.reset()
    total_reward = 0.0
    fell_off = False
    steps = 0

    for _ in range(max_steps):
        steps += 1
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        r = float(reward)
        total_reward += r
        if is_cliff_reward(r):
            fell_off = True

        agent.update(state, action, r, next_state, done)
        state = next_state

        if done:
            break

    return total_reward, fell_off, steps


def run_q_learning_episode(env, agent: QLearningAgent, max_steps: int) -> tuple[float, bool, int]:
    """
    Run one Q-learning episode with online TD updates.
    Q-learning is off-policy. It behaves with ε-greedy exploration,
    but its target uses the greedy value at next state:

        max_a' Q(S_{t+1}, a')

    So it also doesn't need next_action in the runner.

    :param env: Gymnasium environment with discrete spaces.
        :type env: object
    :param agent: Q-learning agent.
        :type agent: QLearningAgent
    :param max_steps: Max steps per episode (safety cap).
        :type max_steps: int

    :return: (episode_return, fell_off_cliff, steps)
        :rtype: tuple[float, bool, int]
    """
    state, _ = env.reset()
    total_reward = 0.0
    fell_off = False
    steps = 0

    for _ in range(max_steps):
        steps += 1
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        r = float(reward)
        total_reward += r
        if is_cliff_reward(r):
            fell_off = True

        agent.update(state, action, r, next_state, done)
        state = next_state

        if done:
            break

    return total_reward, fell_off, steps


# Evaluation functions (done with ε=0)
# These functions mirror the training runners but skip updates and force ε=0
def eval_sarsa_episode(env, agent: SARSAgent, max_steps: int) -> tuple[float, bool, int]:
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    try:
        state, _ = env.reset()
        action = agent.select_action(state)

        total_reward = 0.0
        fell_off = False
        steps = 0

        for _ in range(max_steps):
            steps += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            r = float(reward)
            total_reward += r
            if is_cliff_reward(r):
                fell_off = True

            if done:
                break

            state = next_state
            action = agent.select_action(state)

        return total_reward, fell_off, steps
    finally:
        agent.epsilon = old_eps


def eval_expected_sarsa_episode(env, agent: ExpectedSARSAgent, max_steps: int) -> tuple[float, bool, int]:
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    try:
        state, _ = env.reset()
        total_reward = 0.0
        fell_off = False
        steps = 0

        for _ in range(max_steps):
            steps += 1
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            r = float(reward)
            total_reward += r
            if is_cliff_reward(r):
                fell_off = True

            state = next_state
            if done:
                break

        return total_reward, fell_off, steps
    finally:
        agent.epsilon = old_eps


def eval_q_learning_episode(env, agent: QLearningAgent, max_steps: int) -> tuple[float, bool, int]:
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    try:
        state, _ = env.reset()
        total_reward = 0.0
        fell_off = False
        steps = 0

        for _ in range(max_steps):
            steps += 1
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            r = float(reward)
            total_reward += r
            if is_cliff_reward(r):
                fell_off = True

            state = next_state
            if done:
                break

        return total_reward, fell_off, steps
    finally:
        agent.epsilon = old_eps


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser(description="Train SARSA vs Expected SARSA vs Q-learning on CliffWalking.")
    p.add_argument("--episodes", type=int, default=1000, help="Episodes per run.")
    p.add_argument("--runs", type=int, default=20, help="Number of runs to average.")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    p.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument("--epsilon", type=float, default=0.1, help="Exploration probability.")
    p.add_argument("--seed", type=int, default=0, help="Master seed.")
    p.add_argument("--smooth", type=int, default=10, help="Smoothing window for plotting.")
    p.add_argument("--print-policies", action="store_true", help="Print the greedy policy learned by each method.")  # If you don't pass this flag, the value is False
    p.add_argument("--band-k", type=float, default=1.0, help="Shaded band width in std units (mean ± band_k * std). Use 0 to disable bands.")
    p.add_argument("--eval-every", type=int, default=100, help="Run evaluation every N episodes.")
    p.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes.")

    return p.parse_args()


def main():
    """
    Main training loop:
    - for each run: create envs + agents with a different seed
    - for each episode: run 1 episode per agent and log metrics
    - compute mean/std across runs and save plots (with shaded std bands)
    """
    args = parse_args()
    seed_everything(args.seed)

    env_id_candidates = ["CliffWalking-v1", "CliffWalking-v0"]

    # Probe env to read discrete sizes once
    env_probe = make_first_available(env_id_candidates)
    n_states = int(env_probe.observation_space.n)
    n_actions = int(env_probe.action_space.n)
    env_probe.close()

    # arrays for logging returns per algorithm
    # Store per-run curves to compute mean/std
    returns_sarsa_runs = np.zeros(shape=(args.runs, args.episodes), dtype=np.float64)
    returns_expected_runs = np.zeros(shape=(args.runs, args.episodes), dtype=np.float64)
    returns_q_runs = np.zeros(shape=(args.runs, args.episodes), dtype=np.float64)

    # Per-run cliff-event rate (0/1 per episode, then averaged across runs).
    cliff_sarsa_runs = np.zeros(shape=(args.runs, args.episodes), dtype=np.float64)
    cliff_expected_runs = np.zeros(shape=(args.runs, args.episodes), dtype=np.float64)
    cliff_q_runs = np.zeros(shape=(args.runs, args.episodes), dtype=np.float64)

    # Per-run steps
    steps_sarsa_runs = np.zeros(shape=(args.runs, args.episodes), dtype=np.float64)
    steps_expected_runs = np.zeros(shape=(args.runs, args.episodes), dtype=np.float64)
    steps_q_runs = np.zeros(shape=(args.runs, args.episodes), dtype=np.float64)

    # Evaluation arrays
    eval_every = max(1, args.eval_every)
    eval_points = args.episodes // eval_every
    eval_x = (np.arange(eval_points, dtype=np.int64) + 1) * eval_every

    eval_returns_sarsa_runs = np.zeros(shape=(args.runs, eval_points), dtype=np.float64)
    eval_returns_expected_runs = np.zeros(shape=(args.runs, eval_points), dtype=np.float64)
    eval_returns_q_runs = np.zeros(shape=(args.runs, eval_points), dtype=np.float64)

    rng = np.random.default_rng(args.seed)

    last_sarsa: SARSAgent | None = None
    last_expected: ExpectedSARSAgent | None = None
    last_q: QLearningAgent | None = None

    # multi-run loop
    for run_idx in range(args.runs):
        run_seed = int(rng.integers(low=0, high=1_000_000))

        # create three separate env instances
        env_s = make_first_available(env_id_candidates)
        env_e = make_first_available(env_id_candidates)
        env_q = make_first_available(env_id_candidates)

        # seed envs
        env_s.reset(seed=run_seed)
        env_e.reset(seed=run_seed + 1)
        env_q.reset(seed=run_seed + 2)

        # create three separate env instances for evaluation and seed them
        env_s_eval = make_first_available(env_id_candidates)
        env_e_eval = make_first_available(env_id_candidates)
        env_q_eval = make_first_available(env_id_candidates)

        env_s_eval.reset(seed=run_seed + 10_000)
        env_e_eval.reset(seed=run_seed + 10_001)
        env_q_eval.reset(seed=run_seed + 10_002)

        # create agents (tabular Q tables start at 0)
        sarsa = SARSAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=run_seed,
        )
        expected = ExpectedSARSAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=run_seed,
        )
        ql = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=run_seed,
        )

        # run episodes accumulating returns and cliff flags
        for ep in range(args.episodes):
            ret_s, fell_s, steps_s = run_sarsa_episode(env_s, sarsa, max_steps=args.max_steps)
            ret_e, fell_e, steps_e = run_expected_sarsa_episode(env_e, expected, max_steps=args.max_steps)
            ret_q, fell_q, steps_q = run_q_learning_episode(env_q, ql, max_steps=args.max_steps)

            returns_sarsa_runs[run_idx, ep] = ret_s
            returns_expected_runs[run_idx, ep] = ret_e
            returns_q_runs[run_idx, ep] = ret_q

            cliff_sarsa_runs[run_idx, ep] = 1.0 if fell_s else 0.0
            cliff_expected_runs[run_idx, ep] = 1.0 if fell_e else 0.0
            cliff_q_runs[run_idx, ep] = 1.0 if fell_q else 0.0

            steps_sarsa_runs[run_idx, ep] = float(steps_s)
            steps_expected_runs[run_idx, ep] = float(steps_e)
            steps_q_runs[run_idx, ep] = float(steps_q)

            if (ep + 1) % eval_every == 0:
                k = (ep + 1) // eval_every - 1

                # average over eval_episodes
                rs = list()
                re = list()
                rq = list()
                for _ in range(args.eval_episodes):
                    r_s, _, _ = eval_sarsa_episode(env_s_eval, sarsa, args.max_steps)
                    r_e, _, _ = eval_expected_sarsa_episode(env_e_eval, expected, args.max_steps)
                    r_q, _, _ = eval_q_learning_episode(env_q_eval, ql, args.max_steps)
                    rs.append(r_s)
                    re.append(r_e)
                    rq.append(r_q)

                eval_returns_sarsa_runs[run_idx, k] = float(np.mean(rs))
                eval_returns_expected_runs[run_idx, k] = float(np.mean(re))
                eval_returns_q_runs[run_idx, k] = float(np.mean(rq))

        last_sarsa = sarsa
        last_expected = expected
        last_q = ql

        # close envs
        env_s.close()
        env_e.close()
        env_q.close()
        env_s_eval.close()
        env_e_eval.close()
        env_q_eval.close()

        if (run_idx + 1) % max(1, args.runs // 5) == 0:
            print(f"Completed run {run_idx + 1}/{args.runs}")

    # Mean + std across runs
    # ddof=1 -> sample std -> divide by N-1 in std (we are estimating from a finite sample)
    ddof = 1 if args.runs > 1 else 0
    # returns across runs
    returns_sarsa_mean = returns_sarsa_runs.mean(axis=0)
    returns_expected_mean = returns_expected_runs.mean(axis=0)
    returns_q_mean = returns_q_runs.mean(axis=0)

    returns_sarsa_std = returns_sarsa_runs.std(axis=0, ddof=ddof)
    returns_expected_std = returns_expected_runs.std(axis=0, ddof=ddof)
    returns_q_std = returns_q_runs.std(axis=0, ddof=ddof)

    # Convert cliff flags to percent
    cliff_sarsa_pct = cliff_sarsa_runs.mean(axis=0) * 100.0
    cliff_expected_pct = cliff_expected_runs.mean(axis=0) * 100.0
    cliff_q_pct = cliff_q_runs.mean(axis=0) * 100.0

    cliff_sarsa_pct_std = cliff_sarsa_runs.std(axis=0, ddof=ddof) * 100.0
    cliff_expected_pct_std = cliff_expected_runs.std(axis=0, ddof=ddof) * 100.0
    cliff_q_pct_std = cliff_q_runs.std(axis=0, ddof=ddof) * 100.0

    # keep the bands in a sensible range
    cliff_sarsa_pct = np.clip(cliff_sarsa_pct, a_min=0.0, a_max=100.0)
    cliff_expected_pct = np.clip(cliff_expected_pct, a_min=0.0, a_max=100.0)
    cliff_q_pct = np.clip(cliff_q_pct, a_min=0.0, a_max=100.0)

    # steps across runs
    steps_sarsa_mean = steps_sarsa_runs.mean(axis=0)
    steps_expected_mean = steps_expected_runs.mean(axis=0)
    steps_q_mean = steps_q_runs.mean(axis=0)

    steps_sarsa_std = steps_sarsa_runs.std(axis=0, ddof=ddof)
    steps_expected_std = steps_expected_runs.std(axis=0, ddof=ddof)
    steps_q_std = steps_q_runs.std(axis=0, ddof=ddof)

    # Eval metrics
    eval_returns_sarsa_mean = eval_returns_sarsa_runs.mean(axis=0)
    eval_returns_expected_mean = eval_returns_expected_runs.mean(axis=0)
    eval_returns_q_mean = eval_returns_q_runs.mean(axis=0)

    eval_returns_sarsa_std = eval_returns_sarsa_runs.std(axis=0, ddof=ddof)
    eval_returns_expected_std = eval_returns_expected_runs.std(axis=0, ddof=ddof)
    eval_returns_q_std = eval_returns_q_runs.std(axis=0, ddof=ddof)

    # plots
    out_dir = Path("assets/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_lines_with_bands(
        ys_mean=[returns_sarsa_mean, returns_expected_mean, returns_q_mean],
        ys_std=[returns_sarsa_std, returns_expected_std, returns_q_std],
        labels=["SARSA", "Expected SARSA", "Q-learning"],
        title="CliffWalking: average return per episode (mean ± 1 std)",
        xlabel="Episode",
        ylabel="Return",
        out_path=out_dir / "tabular_cliffwalking_returns.png",
        smooth_window=args.smooth,
        band_k=args.band_k
    )

    save_lines_with_bands(
        ys_mean=[cliff_sarsa_pct, cliff_expected_pct, cliff_q_pct],
        ys_std=[cliff_sarsa_pct_std, cliff_expected_pct_std, cliff_q_pct_std],
        labels=["SARSA", "Expected SARSA", "Q-learning"],
        title="CliffWalking: % episodes with at least one cliff fall (mean ± 1 std)",
        xlabel="Episode",
        ylabel="% episodes with cliff fall",
        out_path=out_dir / "tabular_cliffwalking_cliff_rate.png",
        smooth_window=args.smooth,
        band_k=args.band_k
    )

    save_lines_with_bands(
        ys_mean=[steps_sarsa_mean, steps_expected_mean, steps_q_mean],
        ys_std=[steps_sarsa_std, steps_expected_std, steps_q_std],
        labels=["SARSA", "Expected SARSA", "Q-learning"],
        title="CliffWalking: average steps per episode (mean ± 1 std)",
        xlabel="Episode",
        ylabel="Steps",
        out_path=out_dir / "tabular_cliffwalking_steps.png",
        smooth_window=args.smooth,
        band_k=args.band_k
    )

    save_lines_with_bands(
        ys_mean=[eval_returns_sarsa_mean, eval_returns_expected_mean, eval_returns_q_mean],
        ys_std=[eval_returns_sarsa_std, eval_returns_expected_std, eval_returns_q_std],
        labels=["SARSA (eval)", "Expected SARSA (eval)", "Q-learning (eval)"],
        title=f"CliffWalking: evaluation return (epsilon=0), every {eval_every} eps",
        xlabel="Episode",
        ylabel="Return",
        out_path=out_dir / "tabular_cliffwalking_eval_returns.png",
        smooth_window=1,  # usually don't smooth eval points
        band_k=args.band_k,
        x=eval_x
    )

    print("Saved plots:")
    print(f"  {(out_dir/'tabular_cliffwalking_returns.png').resolve()}")
    print(f"  {(out_dir/'tabular_cliffwalking_cliff_rate.png').resolve()}")
    print(f"  {(out_dir/'tabular_cliffwalking_steps.png').resolve()}")
    print(f"  {(out_dir/'tabular_cliffwalking_eval_returns.png').resolve()}")

    tail = min(100, args.episodes)
    print("\nFinal performance (mean return over last episodes):")
    print(f"  SARSA:          {returns_sarsa_mean[-tail:].mean():.2f}")
    print(f"  Expected SARSA: {returns_expected_mean[-tail:].mean():.2f}")
    print(f"  Q-learning:     {returns_q_mean[-tail:].mean():.2f}")

    print("\nFinal cliff rate (mean % over last episodes):")
    print(f"  SARSA:          {cliff_sarsa_pct[-tail:].mean():.2f}%")
    print(f"  Expected SARSA: {cliff_expected_pct[-tail:].mean():.2f}%")
    print(f"  Q-learning:     {cliff_q_pct[-tail:].mean():.2f}%")

    if args.print_policies and last_sarsa is not None and last_expected is not None and last_q is not None:
        env_print = make_first_available(env_id_candidates)

        print("\nGreedy policy learned by SARSA:")
        print(format_cliffwalking_policy(env=env_print, policy=greedy_policy_from_Q(last_sarsa.Q), debug=False))

        print("\nGreedy policy learned by Expected SARSA:")
        print(format_cliffwalking_policy(env=env_print, policy=greedy_policy_from_Q(last_expected.Q), debug=False))

        print("\nGreedy policy learned by Q-learning:")
        print(format_cliffwalking_policy(env=env_print, policy=greedy_policy_from_Q(last_q.Q), debug=False))

        env_print.close()


if __name__ == "__main__":
    main()
