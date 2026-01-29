from __future__ import annotations
from pathlib import Path
from typing import Sequence, Callable
import matplotlib.pyplot as plt
import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a simple moving average for smoothing curves.

    :param x: 1D array to smooth.
        :type x: np.ndarray
    :param window: Window size (>= 1). If 1, returns x unchanged.
        :type window: int

    :return: Smoothed array (same length as x).
        :rtype: np.ndarray
    """
    if window <= 1:
        return x

    pad = window - 1
    x_pad = np.pad(x, (pad, 0), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x_pad, kernel, mode="valid")


def save_lines(
    *,
    ys: Sequence[np.ndarray],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
    smooth_window: int = 1,
) -> None:
    """
    Save a line plot with multiple curves.

    :param ys: A sequence of 1D arrays (each array is one curve).
        :type ys: Sequence[np.ndarray]
    :param labels: Labels for the legend (same length as ys).
        :type labels: Sequence[str]
    :param title: Plot title.
        :type title: str
    :param xlabel: x-axis label.
        :type xlabel: str
    :param ylabel: y-axis label.
        :type ylabel: str
    :param out_path: Output path for the saved image.
        :type out_path: str | Path
    :param smooth_window: Moving average window for smoothing (1 means no smoothing).
        :type smooth_window: int

    :return: None.
        :rtype: None
    """
    if len(ys) != len(labels):
        raise ValueError("ys and labels must have the same length")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for y, label in zip(ys, labels):
        y_plot = moving_average(np.asarray(y, dtype=np.float64), smooth_window)
        plt.plot(y_plot, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_lines_with_bands(
    *,
    ys_mean: Sequence[np.ndarray],
    ys_std: Sequence[np.ndarray],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
    smooth_window: int = 1,
    band_k: float = 1.0,
    x: np.ndarray | None = None
) -> None:
    """
    Save a line plot with shaded uncertainty bands (mean ± k*std).
    Typical use: average learning curves over multiple runs/seeds and show variability.

    Notes:
    - We smooth both the mean and std with the same moving average window.
      This keeps the plot visually consistent.
    - We do NOT clip values -> if your metric is naturally bounded (e.g. % in [0,100]),
      you can clip before calling.

    :param ys_mean: Sequence of mean curves (each is 1D array).
        :type ys_mean: Sequence[np.ndarray]
    :param ys_std: Sequence of std curves (same shapes as ys_mean).
        :type ys_std: Sequence[np.ndarray]
    :param labels: Legend labels (same length as ys_mean).
        :type labels: Sequence[str]
    :param title: Plot title.
        :type title: str
    :param xlabel: x-axis label.
        :type xlabel: str
    :param ylabel: y-axis label.
        :type ylabel: str
    :param out_path: Output path for the saved image.
        :type out_path: str | Path
    :param smooth_window: Moving average window (1 means no smoothing).
        :type smooth_window: int
    :param band_k: Band width multiplier. 1.0 -> mean ± 1 std.
        :type band_k: float
    :param x: Optional x-axis values (shared for all curves). If provided, must be 1D
        and have the same length as each (smoothed) curve.
        :type x: np.ndarray | None

    :return: None.
        :rtype: None
    """
    if len(ys_mean) != len(labels) or len(ys_std) != len(labels):
        raise ValueError("ys_mean, ys_std, and labels must have the same length")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_arr: np.ndarray | None = None
    if x is not None:
        x_arr = np.asarray(x)
        if x_arr.ndim != 1:
            raise ValueError(f"x must be 1D, got shape {x_arr.shape}")

    plt.figure()

    for m, s, label in zip(ys_mean, ys_std, labels):
        m = np.asarray(m, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)

        if m.shape != s.shape:
            raise ValueError(f"mean/std shape mismatch for '{label}': {m.shape} vs {s.shape}")

        m_plot = moving_average(m, smooth_window)
        s_plot = moving_average(s, smooth_window)

        if x_arr is None:
            x_plot = np.arange(m_plot.shape[0], dtype=np.int64)
        else:
            if x_arr.shape[0] != m_plot.shape[0]:
                raise ValueError(
                    f"x length mismatch for '{label}': len(x)={x_arr.shape[0]} vs len(curve)={m_plot.shape[0]}"
                )
            x_plot = x_arr

        low = m_plot - float(band_k) * s_plot
        high = m_plot + float(band_k) * s_plot

        plt.plot(x_plot, m_plot, label=label)
        plt.fill_between(x_plot, low, high, alpha=0.2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -------------------------------------------------------
# Grid-style plots (useful for DP, and gridworlds)
# -------------------------------------------------------

_ACTION_TO_UV = {
    0: (0.0, -1.0),  # up
    1: (1.0, 0.0),   # right
    2: (0.0, 1.0),   # down
    3: (-1.0, 0.0),  # left
}


def positions_mask(
    *,
    height: int,
    width: int,
    positions: Sequence[tuple[int, int]],
) -> np.ndarray:
    """
    Create a boolean mask for a set of (row, col) positions.

    :param height: Grid height (rows).
        :type height: int
    :param width: Grid width (cols).
        :type width: int
    :param positions: Sequence of (row, col) positions to mark as True.
        :type positions: Sequence[tuple[int, int]]

    :return: Boolean array of shape (height, width) where True marks the provided positions.
        :rtype: np.ndarray
    """
    mask = np.zeros(shape=(int(height), int(width)), dtype=bool)
    for r, c in positions:
        mask[int(r), int(c)] = True
    return mask


def values_to_grid(
    *,
    V: np.ndarray,
    height: int,
    width: int,
    state_to_pos: Callable[[int], tuple[int, int]],
    invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert a flat value vector V(s) into a 2D grid for plotting.

    :param V: State-value array, shape (n_states,).
        :type V: np.ndarray
    :param height: Grid height (rows).
        :type height: int
    :param width: Grid width (cols).
        :type width: int
    :param state_to_pos: Function mapping state index -> (row, col).
        :type state_to_pos: Callable[[int], tuple[int, int]]
    :param invalid_mask: Optional boolean mask (True = invalid cell), shape (height, width).
    It means this cell is not a real state (e.g. a wall). Indeed, some gridworlds can have walls.
    So we place this just in case you want to create a gridworld with walls.
    In a heatmap, we want walls to appear as blank/NaN rather than fake values.
    In a policy plot, we want to skip arrows on walls.
    Note that the gridworld I am using now does not have walls. SO this argument is for those who come after.
        :type invalid_mask: np.ndarray | None

    :return: 2D value grid of shape (height, width). Invalid cells are set to NaN.
        :rtype: np.ndarray
    """
    h = int(height)
    w = int(width)

    V = np.asarray(V, dtype=np.float64)
    V_grid = np.full(shape=(h, w), fill_value=np.nan, dtype=np.float64)

    for s in range(V.shape[0]):
        r, c = state_to_pos(int(s))
        V_grid[int(r), int(c)] = float(V[s])

    if invalid_mask is not None:
        invalid_mask = np.asarray(invalid_mask, dtype=bool)
        if invalid_mask.shape != (h, w):
            raise ValueError(f"invalid_mask must have shape ({h}, {w}), got {invalid_mask.shape}")
        V_grid[invalid_mask] = np.nan

    return V_grid


def policy_to_grid(
    *,
    policy: np.ndarray,
    height: int,
    width: int,
    state_to_pos: Callable[[int], tuple[int, int]],
    terminal_mask: np.ndarray | None = None,
    invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert a flat deterministic policy (one action per state) into a 2D grid.

    Terminal/invalid cells are marked with -1 so they can be skipped in arrow plots.

    :param policy: Policy array, shape (n_states,), ints in [0, n_actions-1].
        :type policy: np.ndarray
    :param height: Grid height (rows).
        :type height: int
    :param width: Grid width (cols).
        :type width: int
    :param state_to_pos: Function mapping state index -> (row, col).
        :type state_to_pos: Callable[[int], tuple[int, int]]
    :param terminal_mask: Optional boolean mask for terminal cells, shape (height, width).
        :type terminal_mask: np.ndarray | None
    :param invalid_mask: Optional boolean mask for invalid cells, shape (height, width).
        :type invalid_mask: np.ndarray | None

    :return: 2D int array of shape (height, width). Non-usable cells are -1.
        :rtype: np.ndarray
    """
    h = int(height)
    w = int(width)

    policy = np.asarray(policy, dtype=np.int64)
    pol_grid = np.full(shape=(h, w), fill_value=-1, dtype=np.int64)

    for s in range(policy.shape[0]):
        r, c = state_to_pos(int(s))
        pol_grid[int(r), int(c)] = int(policy[s])

    if terminal_mask is not None:
        terminal_mask = np.asarray(terminal_mask, dtype=bool)
        if terminal_mask.shape != (h, w):
            raise ValueError(f"terminal_mask must have shape ({h}, {w}), got {terminal_mask.shape}")
        pol_grid[terminal_mask] = -1

    if invalid_mask is not None:
        invalid_mask = np.asarray(invalid_mask, dtype=bool)
        if invalid_mask.shape != (h, w):
            raise ValueError(f"invalid_mask must have shape ({h}, {w}), got {invalid_mask.shape}")
        pol_grid[invalid_mask] = -1

    return pol_grid


def save_value_heatmap(
    *,
    V_grid: np.ndarray,
    out_path: str | Path,
    title: str,
    terminal_mask: np.ndarray | None = None,
    annotate: bool = True,
) -> None:
    """
    Save a value heatmap for a grid MDP.

    :param V_grid: 2D grid of values (NaN allowed for invalid cells).
        :type V_grid: np.ndarray
    :param out_path: Where to save the image.
        :type out_path: str | Path
    :param title: Plot title.
        :type title: str
    :param terminal_mask: Optional boolean mask for terminal cells (used for 'T' labels).
        :type terminal_mask: np.ndarray | None
    :param annotate: If True, write value numbers in cells.
        :type annotate: bool

    :return: None.
        :rtype: None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    V_grid = np.asarray(V_grid, dtype=np.float64)

    fig, ax = plt.subplots()
    im = ax.imshow(V_grid, interpolation="nearest", origin="upper")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    n_rows, n_cols = V_grid.shape

    if terminal_mask is not None:
        terminal_mask = np.asarray(terminal_mask, dtype=bool)
        if terminal_mask.shape != (n_rows, n_cols):
            raise ValueError(f"terminal_mask must have shape {V_grid.shape}, got {terminal_mask.shape}")

    if annotate:
        for r in range(n_rows):
            for c in range(n_cols):
                if terminal_mask is not None and terminal_mask[r, c]:
                    ax.text(c, r, "T", ha="center", va="center")
                    continue

                val = V_grid[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_value_heatmap_with_policy(
    *,
    V_grid: np.ndarray,
    policy_grid: np.ndarray,
    out_path: str | Path,
    title: str,
    terminal_mask: np.ndarray | None = None,
) -> None:
    """
    Save a value heatmap with policy arrows overlaid.

    :param V_grid: 2D grid of values (NaN allowed).
        :type V_grid: np.ndarray
    :param policy_grid: 2D grid of actions; use -1 for cells where arrows should be skipped.
        :type policy_grid: np.ndarray
    :param out_path: Where to save the image.
        :type out_path: str | Path
    :param title: Plot title.
        :type title: str
    :param terminal_mask: Optional boolean mask for terminal cells (labels them as 'T').
        :type terminal_mask: np.ndarray | None

    :return: None.
        :rtype: None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    V_grid = np.asarray(V_grid, dtype=np.float64)
    policy_grid = np.asarray(policy_grid, dtype=np.int64)

    if V_grid.shape != policy_grid.shape:
        raise ValueError(f"V_grid and policy_grid must match shape, got {V_grid.shape} vs {policy_grid.shape}")

    n_rows, n_cols = V_grid.shape

    fig, ax = plt.subplots()
    im = ax.imshow(V_grid, interpolation="nearest", origin="upper")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    # Keep a nice grid look
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.grid(True)

    if terminal_mask is not None:
        terminal_mask = np.asarray(terminal_mask, dtype=bool)
        if terminal_mask.shape != (n_rows, n_cols):
            raise ValueError(f"terminal_mask must have shape {V_grid.shape}, got {terminal_mask.shape}")

    # Build arrow fields only for valid (non -1) actions
    ys, xs = np.where(policy_grid >= 0)
    U = np.zeros_like(xs, dtype=np.float64)
    Vv = np.zeros_like(ys, dtype=np.float64)

    for i, (r, c) in enumerate(zip(ys, xs)):
        a = int(policy_grid[r, c])
        u, v = _ACTION_TO_UV.get(a, (0.0, 0.0))
        U[i] = u
        Vv[i] = v

    ax.quiver(xs, ys, U, Vv, angles="xy", scale_units="xy", scale=1, pivot="middle")

    # Label terminal cells as 'T' for readability
    if terminal_mask is not None:
        for r in range(n_rows):
            for c in range(n_cols):
                if terminal_mask[r, c]:
                    ax.text(c, r, "T", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_convergence_curve(
    *,
    deltas: Sequence[float],
    out_path: str | Path,
    title: str,
    logy: bool = True,
) -> None:
    """
    Save a convergence curve: delta_k vs iteration.

    In value iteration/policy evaluation, a common convergence metric is:

        delta_k = max_s |V_{k+1}(s) - V_k(s)|

    If delta_k shrinks toward 0, values are stabilising.

    :param deltas: Sequence of delta values (one per iteration).
        :type deltas: Sequence[float]
    :param out_path: Where to save the image.
        :type out_path: str | Path
    :param title: Plot title.
        :type title: str
    :param logy: If True, use a log-scale y-axis (often clearer because deltas can drop fast).
        :type logy: bool

    :return: None.
        :rtype: None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y = np.asarray(list(deltas), dtype=np.float64)
    x = np.arange(len(y), dtype=np.int64)

    fig, ax = plt.subplots()
    if logy:
        ax.semilogy(x, y)
    else:
        ax.plot(x, y)

    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"max $|V_{k+1} - V_k|$")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
