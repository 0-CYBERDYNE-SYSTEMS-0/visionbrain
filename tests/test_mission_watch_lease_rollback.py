"""Watch adapter leases are rolled back when their durable snapshot can't commit."""

from __future__ import annotations

import asyncio

import pytest

from visionbrain.mission_contracts import (
    SourceBinding,
    WatchLease,
    WatchLeaseRelease,
    WatchProposal,
)
from visionbrain.mission_runtime import MissionRuntime


class _Clock:
    def now_ms(self) -> int:
        return 1_000

    def monotonic(self) -> float:
        return 50.0


class _Store:
    def __init__(self) -> None:
        self.snapshot = {
            "mission_id": "mission-1",
            "revision": 3,
            "execution_generation": 7,
            "state": "running",
            "mode": "watch",
            "configuration_revision": 11,
            "watch_lease": None,
        }

    def get_mission(self, mission_id: str) -> dict:
        assert mission_id == "mission-1"
        return dict(self.snapshot)


class _WatchAdapter:
    def __init__(self) -> None:
        self.applied: WatchLease | None = None
        self.released: list[tuple[WatchLease, str]] = []
        self.configuration_revision = 11

    def current_configuration_revision(self) -> int:
        return self.configuration_revision

    def apply(self, request) -> WatchLease:
        assert request.expected_configuration_revision == self.configuration_revision
        self.configuration_revision += 1
        self.applied = WatchLease(
            lease_id="watch-lease-1",
            mission_id=request.mission_id,
            source_binding=request.source_binding,
            configuration_revision=self.configuration_revision,
            targets=request.targets,
            task=request.task,
            expires_at_ms=request.expires_at_ms,
        )
        return self.applied

    def release(self, lease: WatchLease, reason: str) -> WatchLeaseRelease:
        self.released.append((lease, reason))
        self.configuration_revision += 1
        return WatchLeaseRelease(released=True, configuration_revision=self.configuration_revision)


def _runtime(store: _Store, adapter: _WatchAdapter) -> MissionRuntime:
    return MissionRuntime(
        store=store,
        planner=object(),
        tools=object(),
        source_provider=lambda _binding: None,
        watch_adapter=adapter,
        event_sink=lambda _event: None,
        clock=_Clock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "false", "raises"])
async def test_failed_watch_snapshot_commit_releases_applied_lease_without_timer(failure):
    store = _Store()
    adapter = _WatchAdapter()
    runtime = _runtime(store, adapter)
    scheduled: list[tuple] = []
    runtime._schedule_lease_timeout = lambda *args: scheduled.append(args)

    async def fail_commit(*_args, **_kwargs):
        if failure == "raises":
            raise OSError("injected durable write failure")
        return False if failure == "false" else None

    runtime._change_snapshot = fail_commit
    binding = SourceBinding("scout-1", "41")
    call = runtime._apply_watch(
        "mission-1",
        7,
        binding,
        WatchProposal(("container",), "detect"),
        frame_received_monotonic=50.0,
    )

    if failure == "raises":
        with pytest.raises(OSError, match="injected durable write failure"):
            await call
    else:
        await call

    assert adapter.applied is not None
    assert adapter.released == [(adapter.applied, "watch_lease_commit_failed")]
    assert scheduled == []
    assert store.snapshot["watch_lease"] is None
