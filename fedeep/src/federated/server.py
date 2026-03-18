"""
FL Server -- pure for-loop orchestration with progressive phases.

Phase schedule:
  Phase 0 (rounds  1..P1):    Backbone warmup (only Exit4)
  Phase 1 (rounds  P1+1..P2): Multi-exit CE (all 4 exits)
  Phase 2 (rounds  P2+1..P3): Fast/Slow split + CMS
  Phase 3 (rounds  P3+1..P4): + Knowledge Distillation chain
  Phase 4 (rounds  P4+1..R):  Full system

Strategy:
  "fedavg"  -- standard FedAvg
  "fedprox" -- same aggregation, proximal term is client-side
  "edpa"    -- Exit-Depth-aware Personalized Aggregation
"""

import logging
import random
from collections import OrderedDict
from typing import Dict, List, Optional

from src.federated.aggregation import (
    build_param_depth_map,
    edpa_aggregate,
    fedavg,
)

logger = logging.getLogger("fedeep.server")

PHASE_NAMES = {
    0: "Backbone Warmup (Exit4 only)",
    1: "Multi-Exit CE (4 exits)",
    2: "Fast/Slow + CMS",
    3: "Self-Distillation (KD chain)",
    4: "Full FedEEP",
}


def get_current_phase(server_round: int, milestones: dict) -> int:
    """Map server_round -> phase index 0..4."""
    if server_round <= milestones["phase_1_round"]:
        return 0
    if server_round <= milestones["phase_2_round"]:
        return 1
    if server_round <= milestones["phase_3_round"]:
        return 2
    if server_round <= milestones["phase_4_round"]:
        return 3
    return 4


def run_fl(
    clients: list,
    global_model,
    config: dict,
    evaluate_fn=None,
) -> List[Dict]:
    """
    Run the full federated learning loop.

    Args:
        clients:      List of Client instances.
        global_model: ConvNeXtEarlyExit model (used for initial weights
                      and param_depth_map).
        config:       Dict with keys:
                        num_rounds, local_epochs, lr, strategy,
                        fraction_train, milestones (dict with phase_*_round),
                        edpa_gamma (for EDPA strategy).
        evaluate_fn:  Optional callable(global_weights, round) -> metrics.
                      If None, uses first client's test_loader.

    Returns:
        List of per-round metrics dicts (the training history).
    """
    num_rounds = config["num_rounds"]
    local_epochs = config["local_epochs"]
    lr = config["lr"]
    strategy = config.get("strategy", "fedavg")
    fraction = config.get("fraction_train", 1.0)
    milestones = config.get("milestones", {
        "phase_1_round": 20,
        "phase_2_round": 40,
        "phase_3_round": 60,
        "phase_4_round": 80,
    })

    # Initial global weights
    global_weights = OrderedDict(
        (k, v.detach().cpu()) for k, v in global_model.state_dict().items()
    )

    # EDPA state
    param_depth_map: Optional[Dict[int, int]] = None
    edpa_client_states: Dict[int, OrderedDict] = {}
    if strategy == "edpa":
        param_depth_map = build_param_depth_map(global_model)
        logger.info(f"EDPA param_depth_map: {len(param_depth_map)} params")

    history: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"FedEEP — Federated Early-Exit with Progressive Phases")
    print(f"{'='*60}")
    print(f"Strategy  : {strategy.upper()}")
    print(f"Rounds    : {num_rounds}")
    print(f"Clients   : {len(clients)} (fraction={fraction})")
    print(f"LR        : {lr}")
    print(f"Milestones: {milestones}")
    print(f"{'='*60}\n")

    for server_round in range(1, num_rounds + 1):
        phase = get_current_phase(server_round, milestones)

        # Log phase transitions
        if server_round == 1 or phase != get_current_phase(
            server_round - 1, milestones
        ):
            print(
                f"\n>>> Round {server_round}: entering Phase {phase} "
                f"— {PHASE_NAMES[phase]}"
            )

        # ── Client selection ──────────────────────────────────────────────
        num_selected = max(1, int(len(clients) * fraction))
        if num_selected < len(clients):
            selected = random.sample(clients, num_selected)
        else:
            selected = clients

        # ── Local training ────────────────────────────────────────────────
        client_weights_list = []
        client_sizes_list = []
        client_ids_list = []

        for client in selected:
            client.set_weights(global_weights)
            client.local_train(
                epochs=local_epochs,
                phase=phase,
                learning_rate=lr,
            )
            client_weights_list.append(client.get_weights())
            client_sizes_list.append(client.num_samples)
            client_ids_list.append(client.id)

        # ── Aggregation ───────────────────────────────────────────────────
        if strategy == "edpa" and param_depth_map is not None:
            global_weights, edpa_client_states = edpa_aggregate(
                client_weights=client_weights_list,
                client_sizes=client_sizes_list,
                client_ids=client_ids_list,
                param_depth_map=param_depth_map,
                gamma=config.get("edpa_gamma", 0.5),
                client_states=edpa_client_states,
            )
        else:
            global_weights = fedavg(client_weights_list, client_sizes_list)

        # ── Evaluation ────────────────────────────────────────────────────
        round_metrics = {
            "round": server_round,
            "phase": phase,
            "strategy": strategy,
            "num_clients": len(selected),
        }

        if evaluate_fn is not None:
            eval_metrics = evaluate_fn(global_weights, server_round)
            round_metrics.update(eval_metrics)
        else:
            # Default: load global weights into first client and evaluate
            clients[0].set_weights(global_weights)
            eval_metrics = clients[0].evaluate()
            round_metrics.update(eval_metrics)

        history.append(round_metrics)

        acc = round_metrics.get("accuracy", 0.0)
        loss = round_metrics.get("loss", 0.0)
        logger.info(
            f"Round {server_round}/{num_rounds} phase={phase} "
            f"loss={loss:.4f} acc={acc:.4f}"
        )
        print(
            f"  Round {server_round:3d}/{num_rounds} | "
            f"Phase {phase} | "
            f"loss={loss:.4f} | acc={acc:.4f}"
        )

    print(f"\n{'='*60}")
    print(f"Training complete! {num_rounds} rounds.")
    print(f"{'='*60}")

    return history
