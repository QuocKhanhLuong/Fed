"""Save and load model checkpoints + FL state for resume."""

from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    state_dict: dict,
    config: dict,
    round_num: int,
    path: str,
    best_acc: float = 0.0,
    best_round: int = 0,
) -> None:
    """
    Save model checkpoint with config and round metadata.

    Args:
        state_dict: Model state_dict to save.
        config:     Experiment config dict.
        round_num:  FL round number.
        path:       File path for the checkpoint.
        best_acc:   Best accuracy so far.
        best_round: Round that achieved best accuracy.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "config": config,
            "round": round_num,
            "best_acc": best_acc,
            "best_round": best_round,
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load checkpoint.

    Returns:
        Dict with keys: state_dict, config, round, best_acc, best_round.
    """
    map_location = device if device else "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)


def save_fl_state(
    path: str,
    global_weights: dict,
    config: dict,
    round_num: int,
    best_acc: float,
    best_round: int,
    edpa_client_states: Optional[dict] = None,
    history: Optional[list] = None,
) -> None:
    """
    Save full FL state for resuming training.

    Includes global weights, EDPA client states, and history.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "global_weights": global_weights,
        "config": config,
        "round": round_num,
        "best_acc": best_acc,
        "best_round": best_round,
    }
    if edpa_client_states is not None:
        state["edpa_client_states"] = edpa_client_states
    if history is not None:
        state["history"] = history
    torch.save(state, path)


def load_fl_state(
    path: str,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load full FL state for resuming.

    Returns:
        Dict with: global_weights, config, round, best_acc, best_round,
                   edpa_client_states (optional), history (optional).
    """
    map_location = device if device else "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)
