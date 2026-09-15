"""Small shared utilities."""

import json
import os
import platform
import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix every relevant random seed so runs are reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_environment_info(output_path: str) -> dict:
    """
    Documents the exact environment a training run happened in — the
    project brief explicitly asks reproducibility to include "documenting
    environment dependencies", not just fixed seeds.
    """
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import torchvision
        info["torchvision_version"] = torchvision.__version__
    except ImportError:
        info["torchvision_version"] = None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)
    return info
