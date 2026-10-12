# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Occupancy node map and tuning configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    device_id: str
    role: str
    weight: float = 1.0


@dataclass(frozen=True)
class OccupancyConfig:
    nodes: tuple[NodeConfig, ...]
    window_seconds: float = 4.0
    stale_after_seconds: float = 3.0
    motion_threshold: float = 0.35
    minimum_samples: int = 2
    strong_node_count: int = 2
    entry_window_seconds: float = 8.0
    doorway_only_exit_seconds: float = 1.5
    exit_window_seconds: float = 8.0
    exit_confirmation_seconds: float = 5.0
    stillness_seconds: float = 20.0
    occupied_threshold: float = 0.75
    empty_threshold: float = 0.20

    @classmethod
    def load(cls, path: Path) -> "OccupancyConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("occupancy config must be a YAML object")
        raw_nodes = raw.pop("nodes", None)
        if not isinstance(raw_nodes, list) or len(raw_nodes) < 2:
            raise ValueError("occupancy config must contain at least two nodes")
        nodes = tuple(NodeConfig(**node) for node in raw_nodes)
        config = cls(nodes=nodes, **raw)
        config.validate()
        return config

    def validate(self) -> None:
        if len({node.node_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("node_id values must be unique")
        if len({node.device_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("device_id values must be unique")
        if not any(node.role == "doorway" for node in self.nodes):
            raise ValueError("at least one node must have role: doorway")
        if not any(node.role == "internal" for node in self.nodes):
            raise ValueError("at least one node must have role: internal")
        if any(node.role not in {"doorway", "internal"} for node in self.nodes):
            raise ValueError("node roles must be doorway or internal")
        if not 0 <= self.empty_threshold < self.occupied_threshold <= 1:
            raise ValueError("confidence thresholds must satisfy 0 <= empty < occupied <= 1")
        positive = (
            self.window_seconds,
            self.stale_after_seconds,
            self.motion_threshold,
            self.entry_window_seconds,
            self.doorway_only_exit_seconds,
            self.exit_window_seconds,
            self.exit_confirmation_seconds,
            self.stillness_seconds,
        )
        if any(value <= 0 for value in positive) or self.minimum_samples < 1 or self.strong_node_count < 1:
            raise ValueError("timing, motion, and sample settings must be positive")
