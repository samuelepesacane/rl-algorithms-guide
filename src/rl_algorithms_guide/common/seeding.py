from __future__ import annotations
import random
import numpy as np
import os

def seed_everything(seed: int, use_torch: bool = False, deterministic_torch: bool = True) -> None:
    """
    Seed common RNGs for reproducibility.

    Seeds:
    - Python's random
    - NumPy
    - PyTorch (CPU + CUDA, if available) when use_torch=True

    Notes:
    - Full determinism on GPU is not always guaranteed and can reduce performance
    - deterministic_torch=True is a reasonable default for an educational repo

    :param seed: Master seed.
        :type seed: int
    :param use_torch: If True it will check for torch installation and, also, seed PyTorch RNGs (and optionally enable determinism).
        :type use_torch: bool
    :param deterministic_torch: If True, try to enforce deterministic PyTorch behavior.
        :type deterministic_torch: bool

    :return: None.
        :rtype: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not use_torch:
        return

    try:
        import torch
    except Exception as e:
        # If Torch is required for this run, then something failed
        raise ImportError("use_torch=True, but PyTorch could not be imported.") from e

    # Torch RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # cuDNN flags (mainly affects convnets -> harmless here, but consistent)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Some ops have no deterministic implementation
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        # Helps determinism in some CUDA BLAS calls (must be set early to fully apply)
        os.environ.setdefault(key="CUBLAS_WORKSPACE_CONFIG", value=":4096:8")
