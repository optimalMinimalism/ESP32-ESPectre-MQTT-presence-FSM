# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Transport-neutral occupancy data model."""

from dataclasses import dataclass
from enum import Enum


class OccupancyState(str, Enum):
    EMPTY = "EMPTY"
    POSSIBLE_ENTRY = "POSSIBLE_ENTRY"
    OCCUPIED_MOVING = "OCCUPIED_MOVING"
    OCCUPIED_STILL = "OCCUPIED_STILL"
    POSSIBLE_EXIT = "POSSIBLE_EXIT"


@dataclass(frozen=True)
class NodeObservation:
    node_id: str
    timestamp: str
    received_monotonic: float
    motion_score: float
    motion_state: str
    rssi: float | None = None
    confidence: float | None = None
    device_timestamp_ms: int | None = None


@dataclass(frozen=True)
class FusionSnapshot:
    timestamp: str
    scores: dict[str, float | None]
    fused_score: float
    active_nodes: tuple[str, ...]
    doorway_active: bool
    internal_active: bool
    internal_strong: bool
    missing_nodes: tuple[str, ...]


@dataclass(frozen=True)
class OccupancyEvent:
    timestamp: str
    observation: NodeObservation
    fusion: FusionSnapshot
    state_before: OccupancyState
    state_after: OccupancyState
    occupancy_confidence: float
    reason: str
    last_entry: str | None
    last_exit: str | None
