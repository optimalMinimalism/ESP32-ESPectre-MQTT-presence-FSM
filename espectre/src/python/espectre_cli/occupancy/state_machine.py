# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Persistent room occupancy state machine."""

from .config import OccupancyConfig
from .model import FusionSnapshot, NodeObservation, OccupancyEvent, OccupancyState


class OccupancyStateMachine:
    """Keep occupancy through stillness and clear it only after a confirmed exit."""

    def __init__(self, config: OccupancyConfig):
        self.config = config
        self.state = OccupancyState.EMPTY
        self.confidence = 0.0
        self.last_entry: str | None = None
        self.last_exit: str | None = None
        self._last_update: float | None = None
        self._door_started: float | None = None
        self._last_strong_motion: float | None = None
        self._doorway_only_started: float | None = None
        self._exit_started: float | None = None
        self._doorway_inactive_started: float | None = None

    def update(self, observation: NodeObservation, fusion: FusionSnapshot) -> OccupancyEvent:
        now = observation.received_monotonic
        before = self.state
        dt = 0.0 if self._last_update is None else min(1.0, max(0.0, now - self._last_update))
        self._last_update = now
        reason = "state_held"

        if fusion.internal_strong:
            self._last_strong_motion = now

        if self.state == OccupancyState.EMPTY:
            self.confidence = max(0.0, self.confidence - 0.08 * dt)
            if fusion.internal_active:
                self._confirm_occupancy(observation)
                reason = "internal_activity_confirms_occupancy"
            elif fusion.doorway_active:
                self.state = OccupancyState.POSSIBLE_ENTRY
                self._door_started = now
                self.confidence = max(self.confidence, 0.35)
                reason = "door_activity"

        elif self.state == OccupancyState.POSSIBLE_ENTRY:
            if fusion.internal_active:
                self._confirm_occupancy(observation)
                reason = "internal_activity_confirms_occupancy"
            elif not self._within(now, self._door_started, self.config.entry_window_seconds):
                self.state = OccupancyState.EMPTY
                self.confidence = min(self.confidence, self.config.empty_threshold)
                reason = "entry_window_expired"

        elif self.state in {OccupancyState.OCCUPIED_MOVING, OccupancyState.OCCUPIED_STILL}:
            self._update_occupied_confidence(dt, fusion)
            if fusion.internal_active:
                self._doorway_only_started = None
                if self.state == OccupancyState.OCCUPIED_STILL:
                    self.state = OccupancyState.OCCUPIED_MOVING
                    reason = "internal_motion_resumed"
            elif fusion.doorway_active:
                if self._doorway_only_started is None:
                    self._doorway_only_started = now
                    reason = "doorway_only_exit_timing"
                elif now - self._doorway_only_started >= self.config.doorway_only_exit_seconds:
                    self.state = OccupancyState.POSSIBLE_EXIT
                    self._exit_started = now
                    self._doorway_inactive_started = None
                    reason = "doorway_active_without_internal_activity"
            else:
                self._doorway_only_started = None

            if (
                self.state == OccupancyState.OCCUPIED_MOVING
                and not fusion.internal_active
                and not self._within(now, self._last_strong_motion, self.config.stillness_seconds)
            ):
                self.state = OccupancyState.OCCUPIED_STILL
                reason = "motion_low_but_no_confirmed_exit"

        else:  # POSSIBLE_EXIT
            self.confidence = max(self.config.empty_threshold + 0.01, self.confidence - 0.01 * dt)
            if fusion.internal_active:
                self._confirm_occupancy(observation, record_entry=False)
                reason = "internal_motion_resumed_exit_cancelled"
            elif fusion.doorway_active:
                self._doorway_inactive_started = None
                reason = "exit_waiting_for_doorway_clear"
            elif self._doorway_inactive_started is None:
                self._doorway_inactive_started = now
                reason = "exit_confirming_doorway_clear"
            elif now - self._doorway_inactive_started >= self.config.exit_confirmation_seconds:
                self.state = OccupancyState.EMPTY
                self.confidence = min(0.10, self.config.empty_threshold)
                self.last_exit = observation.timestamp
                self._exit_started = None
                self._doorway_only_started = None
                self._doorway_inactive_started = None
                reason = "doorway_cleared_without_internal_return"

        return OccupancyEvent(
            timestamp=observation.timestamp,
            observation=observation,
            fusion=fusion,
            state_before=before,
            state_after=self.state,
            occupancy_confidence=round(self.confidence, 4),
            reason=reason,
            last_entry=self.last_entry,
            last_exit=self.last_exit,
        )

    def _confirm_occupancy(self, observation: NodeObservation, *, record_entry: bool = True) -> None:
        self.state = OccupancyState.OCCUPIED_MOVING
        self.confidence = max(self.confidence, min(1.0, self.config.occupied_threshold + 0.10))
        if record_entry:
            self.last_entry = observation.timestamp
        self._doorway_only_started = None
        self._exit_started = None
        self._doorway_inactive_started = None

    def _update_occupied_confidence(self, dt: float, fusion: FusionSnapshot) -> None:
        if fusion.internal_strong:
            self.confidence = min(1.0, self.confidence + 0.06 * dt)
        elif fusion.internal_active:
            self.confidence = min(1.0, self.confidence + 0.015 * dt)
        else:
            self.confidence = max(self.config.empty_threshold + 0.01, self.confidence - 0.002 * dt)

    @staticmethod
    def _within(now: float, started: float | None, window: float) -> bool:
        return started is not None and 0.0 <= now - started <= window
