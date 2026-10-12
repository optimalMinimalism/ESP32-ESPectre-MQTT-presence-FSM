# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Tests for multi-node occupancy fusion and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from espectre_cli.occupancy import (
    EventLogger,
    FusionSnapshot,
    NodeConfig,
    NodeObservation,
    OccupancyConfig,
    OccupancyState,
    OccupancyStateMachine,
    WindowFusion,
    _device_id_from_topic,
)


@pytest.fixture
def config() -> OccupancyConfig:
    return OccupancyConfig(
        nodes=(
            NodeConfig("S3_A", "a", "doorway"),
            NodeConfig("S3_B", "b", "internal"),
            NodeConfig("S3_C", "c", "internal"),
            NodeConfig("S3_D", "d", "internal"),
        ),
        window_seconds=2.0,
        stale_after_seconds=2.0,
        motion_threshold=0.5,
        minimum_samples=2,
        strong_node_count=2,
        entry_window_seconds=5.0,
        doorway_only_exit_seconds=1.5,
        exit_window_seconds=5.0,
        exit_confirmation_seconds=2.0,
        stillness_seconds=3.0,
    )


def observation(node_id: str, at: float, score: float = 0.8, state: str = "motion") -> NodeObservation:
    return NodeObservation(
        node_id=node_id,
        timestamp=f"2026-10-06T12:00:{at:06.3f}Z",
        received_monotonic=at,
        motion_score=score,
        motion_state=state,
    )


def snapshot(
    at: float,
    *,
    doorway: bool = False,
    internal: bool = False,
    strong: bool = False,
) -> FusionSnapshot:
    active = (["S3_A"] if doorway else []) + (["S3_B", "S3_C"] if strong else ["S3_B"] if internal else [])
    return FusionSnapshot(
        timestamp=f"2026-10-06T12:00:{at:06.3f}Z",
        scores={"S3_A": 0.8 if doorway else 0.1, "S3_B": 0.8 if internal or strong else 0.1},
        fused_score=0.8 if active else 0.1,
        active_nodes=tuple(active),
        doorway_active=doorway,
        internal_active=internal or strong,
        internal_strong=strong,
        missing_nodes=(),
    )


def update(machine: OccupancyStateMachine, at: float, **flags) -> None:
    machine.update(observation("S3_A", at), snapshot(at, **flags))


def test_single_node_spike_is_not_strong_motion(config: OccupancyConfig) -> None:
    fusion = WindowFusion(config)

    first = fusion.add(observation("S3_B", 0.0))
    second = fusion.add(observation("S3_B", 0.2))

    assert not first.internal_active
    assert second.internal_active
    assert not second.internal_strong


def test_entry_stillness_and_confirmed_exit(config: OccupancyConfig) -> None:
    machine = OccupancyStateMachine(config)

    update(machine, 0.0, doorway=True)
    assert machine.state == OccupancyState.POSSIBLE_ENTRY
    update(machine, 0.5, doorway=True, strong=True)
    assert machine.state == OccupancyState.OCCUPIED_MOVING
    assert machine.confidence >= config.occupied_threshold

    update(machine, 1.0, doorway=False)
    update(machine, 4.0)
    assert machine.state == OccupancyState.OCCUPIED_STILL
    assert machine.confidence > config.empty_threshold

    update(machine, 5.0, doorway=True)
    update(machine, 6.6, doorway=True)
    assert machine.state == OccupancyState.POSSIBLE_EXIT
    update(machine, 7.0)
    assert machine.state == OccupancyState.POSSIBLE_EXIT
    update(machine, 9.1)
    assert machine.state == OccupancyState.EMPTY
    assert machine.confidence < config.empty_threshold


def test_internal_activity_alone_confirms_occupancy(config: OccupancyConfig) -> None:
    machine = OccupancyStateMachine(config)

    update(machine, 0.0, internal=True)

    assert machine.state == OccupancyState.OCCUPIED_MOVING
    assert machine.last_entry is not None


def test_doorway_only_activity_must_last_before_possible_exit(config: OccupancyConfig) -> None:
    machine = OccupancyStateMachine(config)
    update(machine, 0.0, internal=True)

    update(machine, 1.0, doorway=True)
    update(machine, 2.4, doorway=True)
    assert machine.state in {OccupancyState.OCCUPIED_MOVING, OccupancyState.OCCUPIED_STILL}

    update(machine, 2.6, doorway=True)
    assert machine.state == OccupancyState.POSSIBLE_EXIT

def test_internal_motion_cancels_possible_exit(config: OccupancyConfig) -> None:
    machine = OccupancyStateMachine(config)
    update(machine, 0.0, internal=True)
    update(machine, 1.0, doorway=True)
    update(machine, 2.6, doorway=True)
    assert machine.state == OccupancyState.POSSIBLE_EXIT

    update(machine, 3.0, internal=True)

    assert machine.state == OccupancyState.OCCUPIED_MOVING


def test_possible_exit_confirms_only_after_doorway_is_inactive(config: OccupancyConfig) -> None:
    machine = OccupancyStateMachine(config)
    update(machine, 0.0, internal=True)
    update(machine, 1.0, doorway=True)
    update(machine, 2.6, doorway=True)

    update(machine, 10.0, doorway=True)
    assert machine.state == OccupancyState.POSSIBLE_EXIT

    update(machine, 10.5)
    update(machine, 11.9)
    assert machine.state == OccupancyState.POSSIBLE_EXIT

    update(machine, 12.6)
    assert machine.state == OccupancyState.EMPTY


def test_simultaneous_door_and_first_internal_activity_is_not_an_exit(config: OccupancyConfig) -> None:
    machine = OccupancyStateMachine(config)
    update(machine, 0.0, doorway=True)
    update(machine, 0.5, doorway=True, strong=True)
    update(machine, 1.0, doorway=False)
    update(machine, 10.0)
    assert machine.state == OccupancyState.OCCUPIED_STILL

    update(machine, 11.0, doorway=True, strong=True)

    assert machine.state == OccupancyState.OCCUPIED_MOVING


def test_stillness_never_transitions_to_empty(config: OccupancyConfig) -> None:
    machine = OccupancyStateMachine(config)
    update(machine, 0.0, doorway=True)
    update(machine, 0.5, doorway=True, strong=True)
    update(machine, 1.0, doorway=False)

    for at in (4.0, 30.0, 300.0):
        update(machine, at)

    assert machine.state == OccupancyState.OCCUPIED_STILL
    assert machine.confidence > config.empty_threshold


def test_config_loads_node_map(config: OccupancyConfig, tmp_path: Path) -> None:
    path = tmp_path / "occupancy.yaml"
    path.write_text(
        """nodes:
  - {node_id: door, device_id: abc, role: doorway}
  - {node_id: room, device_id: def, role: internal}
motion_threshold: 0.4
""",
        encoding="utf-8",
    )

    loaded = OccupancyConfig.load(path)

    assert loaded.nodes[0].node_id == "door"
    assert loaded.motion_threshold == 0.4


def test_device_id_is_read_from_canonical_topic() -> None:
    assert _device_id_from_topic("espectre/v1/devices/abc/motion", "espectre/v1/devices") == "abc"
    with pytest.raises(ValueError, match="invalid device id"):
        _device_id_from_topic("espectre/v1/devices/a/b/motion", "espectre/v1/devices")


def test_event_logger_writes_csv_and_jsonl(config: OccupancyConfig, tmp_path: Path) -> None:
    machine = OccupancyStateMachine(config)
    event = machine.update(observation("S3_A", 0.0), snapshot(0.0, doorway=True))
    csv_path = tmp_path / "events.csv"
    jsonl_path = tmp_path / "events.jsonl"
    logger = EventLogger(csv_path, jsonl_path)

    logger.write(event)
    logger.close()

    assert "POSSIBLE_ENTRY" in csv_path.read_text(encoding="utf-8")
    assert '"reason":"door_activity"' in jsonl_path.read_text(encoding="utf-8")
