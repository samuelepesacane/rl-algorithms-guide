from __future__ import annotations
import random
import numpy as np


def seed_everything(seed: int) -> None:
    """
    Seed common RNGs for reproducibility.

    This seeds:
    - Python's random
    - NumPy

    If PyTorch is installed, it also seeds torch RNGs and sets some determinism flags.

    :param seed: Master seed.
        :type seed: int

    :return: None.
        :rtype: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional at this stage.
        pass
