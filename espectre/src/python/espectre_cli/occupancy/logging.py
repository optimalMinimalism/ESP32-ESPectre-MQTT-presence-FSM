# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""CSV and JSONL occupancy event logging."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from .model import OccupancyEvent


class EventLogger:
    FIELDNAMES = (
        "timestamp",
        "node_scores",
        "fused_score",
        "active_nodes",
        "missing_nodes",
        "state_before",
        "state_after",
        "occupancy_confidence",
        "reason",
        "source_node",
        "source_motion_state",
        "source_rssi",
    )

    def __init__(self, csv_path: Path | None, jsonl_path: Path | None):
        self._csv_file: TextIO | None = None
        self._jsonl_file: TextIO | None = None
        self._csv_writer: csv.DictWriter | None = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            existed = csv_path.exists() and csv_path.stat().st_size > 0
            self._csv_file = csv_path.open("a", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDNAMES)
            if not existed:
                self._csv_writer.writeheader()
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = jsonl_path.open("a", encoding="utf-8")

    def write(self, event: OccupancyEvent) -> None:
        if self._jsonl_file is not None:
            json.dump(_event_dict(event), self._jsonl_file, separators=(",", ":"))
            self._jsonl_file.write("\n")
            self._jsonl_file.flush()
        if self._csv_writer is not None and self._csv_file is not None:
            self._csv_writer.writerow(
                {
                    "timestamp": event.timestamp,
                    "node_scores": json.dumps(event.fusion.scores, sort_keys=True, separators=(",", ":")),
                    "fused_score": event.fusion.fused_score,
                    "active_nodes": ",".join(event.fusion.active_nodes),
                    "missing_nodes": ",".join(event.fusion.missing_nodes),
                    "state_before": event.state_before.value,
                    "state_after": event.state_after.value,
                    "occupancy_confidence": event.occupancy_confidence,
                    "reason": event.reason,
                    "source_node": event.observation.node_id,
                    "source_motion_state": event.observation.motion_state,
                    "source_rssi": event.observation.rssi,
                }
            )
            self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        if self._jsonl_file is not None:
            self._jsonl_file.close()


def _event_dict(event: OccupancyEvent) -> dict[str, Any]:
    result = asdict(event)
    result["state_before"] = event.state_before.value
    result["state_after"] = event.state_after.value
    return result
