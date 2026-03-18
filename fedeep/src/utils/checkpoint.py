"""Save and load model checkpoints."""

from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    state_dict: dict,
    config: dict,
    round_num: int,
    path: str,
) -> None:
    """
    Save model checkpoint with config and round metadata.

    Args:
        state_dict: Model state_dict to save.
        config:     Experiment config dict (stored alongside weights).
        round_num:  FL round number.
        path:       File path for the checkpoint.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "config": config,
            "round": round_num,
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load checkpoint.

    Args:
        path:   Checkpoint file path.
        device: Device to map tensors to. None = CPU.

    Returns:
        Dict with keys: state_dict, config, round.
    """
    map_location = device if device else "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)
