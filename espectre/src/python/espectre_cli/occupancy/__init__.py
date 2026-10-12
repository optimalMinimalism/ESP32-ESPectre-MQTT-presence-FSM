# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Multi-node, rule-based room occupancy aggregation."""

from .config import NodeConfig, OccupancyConfig
from .fusion import WindowFusion
from .logging import EventLogger
from .model import FusionSnapshot, NodeObservation, OccupancyEvent, OccupancyState
from .service import _device_id_from_topic, run_occupancy_service
from .state_machine import OccupancyStateMachine

__all__ = [
    "EventLogger",
    "FusionSnapshot",
    "NodeConfig",
    "NodeObservation",
    "OccupancyConfig",
    "OccupancyEvent",
    "OccupancyState",
    "OccupancyStateMachine",
    "WindowFusion",
    "_device_id_from_topic",
    "run_occupancy_service",
]
