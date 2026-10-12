# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""MQTT adapter and occupancy service lifecycle."""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..mqtt_shell import _make_mqtt_client
from .config import OccupancyConfig
from .dashboard import TerminalDashboard
from .fusion import WindowFusion
from .logging import EventLogger
from .model import NodeObservation
from .state_machine import OccupancyStateMachine


class OccupancyService:
    def __init__(self, args: Any, config: OccupancyConfig):
        self.args = args
        self.config = config
        self._device_to_node = {node.device_id: node.node_id for node in config.nodes}
        self._fusion = WindowFusion(config)
        self._machine = OccupancyStateMachine(config)
        self._logger = EventLogger(
            Path(args.csv_log) if args.csv_log else None,
            Path(args.jsonl_log) if args.jsonl_log else None,
        )
        self._dashboard = TerminalDashboard(tuple(node.node_id for node in config.nodes), not args.no_dashboard)
        self._lock = threading.Lock()
        self._client = _make_mqtt_client(args.username, args.password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def run(self) -> int:
        try:
            self._client.connect(self.args.broker, self.args.port_mqtt, keepalive=60)
            self._client.loop_forever()
        except KeyboardInterrupt:
            return 0
        finally:
            self._client.disconnect()
            self._logger.close()
        return 0

    def _on_connect(self, client, _userdata, _flags, rc) -> None:
        if rc != 0:
            print(f"MQTT connection failed with return code {rc}")
            return
        topic = f"{self.args.topic_prefix.rstrip('/')}/+/motion"
        client.subscribe(topic, qos=0)
        print(f"Listening for {len(self.config.nodes)} configured nodes on {topic}")

    def _on_message(self, _client, _userdata, message) -> None:
        try:
            device_id = _device_id_from_topic(message.topic, self.args.topic_prefix)
            node_id = self._device_to_node.get(device_id)
            if node_id is None:
                return
            payload = json.loads(message.payload.decode("utf-8"))
            observation = _observation(node_id, payload)
            with self._lock:
                fusion = self._fusion.add(observation)
                event = self._machine.update(observation, fusion)
                self._logger.write(event)
                self._dashboard.render(event)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            print(f"Ignoring invalid motion message on {message.topic}: {exc}")


def _observation(node_id: str, payload: dict[str, Any]) -> NodeObservation:
    score = float(payload["score"])
    if not math.isfinite(score) or score < 0:
        raise ValueError("score must be a finite non-negative number")
    state = str(payload["state"])
    if state not in {"idle", "motion"}:
        raise ValueError("state must be idle or motion")
    received = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return NodeObservation(
        node_id=node_id,
        timestamp=received,
        received_monotonic=time.monotonic(),
        motion_score=score,
        motion_state=state,
        rssi=_optional_float(payload.get("rssi", payload.get("wifi_rssi_dbm"))),
        confidence=_optional_float(payload.get("confidence")),
        device_timestamp_ms=int(payload["timestamp_ms"]) if "timestamp_ms" in payload else None,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _device_id_from_topic(topic: str, topic_prefix: str) -> str:
    prefix = topic_prefix.rstrip("/") + "/"
    if not topic.startswith(prefix) or not topic.endswith("/motion"):
        raise ValueError("unexpected MQTT topic")
    device_id = topic[len(prefix) : -len("/motion")]
    if not device_id or "/" in device_id:
        raise ValueError("invalid device id in MQTT topic")
    return device_id


def run_occupancy_service(args: Any) -> int:
    config = OccupancyConfig.load(Path(args.config))
    return OccupancyService(args, config).run()
