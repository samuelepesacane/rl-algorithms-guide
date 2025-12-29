from __future__ import annotations
from pathlib import Path
from typing import Sequence
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
