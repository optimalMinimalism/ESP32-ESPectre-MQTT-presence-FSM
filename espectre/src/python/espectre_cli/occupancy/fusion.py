# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Short-window multi-node motion fusion."""

from collections import deque

from .config import OccupancyConfig
from .model import FusionSnapshot, NodeObservation


class WindowFusion:
    """Fuse recent generic observations without depending on ESPectre internals."""

    def __init__(self, config: OccupancyConfig):
        self.config = config
        self._samples: dict[str, deque[NodeObservation]] = {
            node.node_id: deque() for node in config.nodes
        }
        self._nodes = {node.node_id: node for node in config.nodes}

    def add(self, observation: NodeObservation) -> FusionSnapshot:
        now = observation.received_monotonic
        samples = self._samples[observation.node_id]
        samples.append(observation)
        cutoff = now - self.config.window_seconds
        for node_samples in self._samples.values():
            while node_samples and node_samples[0].received_monotonic < cutoff:
                node_samples.popleft()

        scores: dict[str, float | None] = {}
        active: list[str] = []
        missing: list[str] = []
        weighted_sum = 0.0
        total_weight = 0.0
        for node_id, node_samples in self._samples.items():
            fresh = bool(node_samples) and now - node_samples[-1].received_monotonic <= self.config.stale_after_seconds
            if not fresh:
                scores[node_id] = None
                missing.append(node_id)
                continue
            score = sum(sample.motion_score for sample in node_samples) / len(node_samples)
            scores[node_id] = score
            recent_motion = node_samples[-1].motion_state.lower() == "motion"
            if len(node_samples) >= self.config.minimum_samples and (
                score >= self.config.motion_threshold or recent_motion
            ):
                active.append(node_id)
            weight = self._nodes[node_id].weight
            weighted_sum += score * weight
            total_weight += weight

        doorway_active = any(self._nodes[node_id].role == "doorway" for node_id in active)
        internal_active_ids = [node_id for node_id in active if self._nodes[node_id].role == "internal"]
        return FusionSnapshot(
            timestamp=observation.timestamp,
            scores=scores,
            fused_score=weighted_sum / total_weight if total_weight else 0.0,
            active_nodes=tuple(active),
            doorway_active=doorway_active,
            internal_active=bool(internal_active_ids),
            internal_strong=len(internal_active_ids) >= self.config.strong_node_count,
            missing_nodes=tuple(missing),
        )
