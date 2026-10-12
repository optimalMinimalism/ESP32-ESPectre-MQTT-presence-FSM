# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Live terminal occupancy dashboard."""

from .model import OccupancyEvent


class TerminalDashboard:
    def __init__(self, node_ids: tuple[str, ...], enabled: bool = True):
        self.node_ids = node_ids
        self.enabled = enabled
        self._rendered = False

    def render(self, event: OccupancyEvent) -> None:
        if not self.enabled:
            if event.state_before != event.state_after:
                print(f"{event.timestamp} {event.state_before.value} -> {event.state_after.value}: {event.reason}")
            return
        if self._rendered:
            print(f"\033[{len(self.node_ids) + 5}A", end="")
        print("ESPectre room occupancy")
        print(f"State       {event.state_after.value:<28}")
        print(f"Confidence  {event.occupancy_confidence:>6.1%}    Fused score {event.fusion.fused_score:>6.3f}")
        for node_id in self.node_ids:
            score = event.fusion.scores.get(node_id)
            label = "--" if score is None else f"{score:.3f}"
            status = "ACTIVE" if node_id in event.fusion.active_nodes else "idle"
            print(f"{node_id:<12} {label:>7}  {status:<8}")
        print(f"Last entry  {event.last_entry or '--'}")
        print(f"Last exit   {event.last_exit or '--'}")
        self._rendered = True
