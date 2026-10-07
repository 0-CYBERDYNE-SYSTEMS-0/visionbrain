"""CPU-only probe for runtime native-work admission during lifecycle churn."""

import asyncio
import base64
import hashlib
import threading
from io import BytesIO

from PIL import Image

from visionbrain.mission_contracts import (
    Decision,
    Principal,
    ToolResult,
    WatchLeaseRelease,
)
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), (15, 90, 170)).save(output, format="JPEG")
    return output.getvalue()


class _ProbePlanner:
    def __init__(self) -> None:
        self.native_lock = threading.RLock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self._counts_lock = threading.Lock()
        self.calls_started = 0
        self.calls_finished = 0
        self.native_active = 0
        self.max_native_active = 0

    def available(self, model_key: str) -> bool:
        return model_key == "gemma"

    def plan(self, _context) -> Decision:
        with self._counts_lock:
            self.calls_started += 1
            call_number = self.calls_started
        with self.native_lock:
            with self._counts_lock:
                self.native_active += 1
                self.max_native_active = max(self.max_native_active, self.native_active)
            try:
                if call_number == 1:
                    self.entered.set()
                    if not self.release.wait(2.0):
                        raise TimeoutError("fake native-call release barrier timed out")
                return Decision(1, "finish")
            finally:
                with self._counts_lock:
                    self.native_active -= 1
                    self.calls_finished += 1

    def close(self) -> None:
        return None


class _Tools:
    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, _request, _context):
        return ToolResult("empty")


class _Watch:
    def current_configuration_revision(self) -> int:
        return 0

    def release(self, _lease, _reason):
        return WatchLeaseRelease(True, 1)


async def _command(runtime, command, *, integrated=False):
    return await runtime.handle(command, Principal("native-queue-probe"), SCOPES, integrated=integrated)


async def _create(runtime):
    return await _command(runtime, {
        "type": "mission_command", "schema_version": 1, "request_id": "probe-create",
        "command": "create", "mission_id": None,
        "args": {
            "profile": {"id": "visual_inspection", "version": 1},
            "expertise": "bounded CPU queue probe", "mode": "inspect", "reasoning_model": "gemma",
        },
    })


async def _attach(runtime, snapshot):
    image = _jpeg()
    return await _command(runtime, {
        "type": "mission_command", "schema_version": 1, "request_id": "probe-attach",
        "command": "attach_evidence", "mission_id": snapshot["mission_id"],
        "expected_revision": snapshot["revision"],
        "args": {"jpeg_b64": base64.b64encode(image).decode(), "sha256": hashlib.sha256(image).hexdigest()},
    })


async def _update(runtime, snapshot, request_id, expertise):
    return await _command(runtime, {
        "type": "mission_command", "schema_version": 1, "request_id": request_id,
        "command": "update_brief", "mission_id": snapshot["mission_id"],
        "expected_revision": snapshot["revision"], "args": {"expertise": expertise},
    }, integrated=True)


async def _resume(runtime, snapshot, request_id):
    return await _command(runtime, {
        "type": "mission_command", "schema_version": 1, "request_id": request_id,
        "command": "resume", "mission_id": snapshot["mission_id"],
        "expected_revision": snapshot["revision"], "args": {},
    })


async def _wait_for(predicate, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError("bounded lifecycle wait expired")


def test_pause_resume_and_updates_do_not_queue_native_jobs_behind_blocked_call(tmp_path):
    async def scenario():
        planner = _ProbePlanner()
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store,
            planner,
            _Tools(),
            lambda _binding: None,
            _Watch(),
            lambda _event: None,
            qualified_models={"gemma": ("inspect",)},
        )
        try:
            created = await _create(runtime)
            attached = await _attach(runtime, created["result"]["snapshot"])
            snapshot = attached["result"]["snapshot"]
            resumed = await _resume(runtime, snapshot, "probe-resume-1")
            assert resumed["ok"]
            assert await asyncio.to_thread(planner.entered.wait, 1.0)
            snapshot = store.get_mission(snapshot["mission_id"])
            assert len(runtime._native_calls) == 1

            # While one real runtime planner call is blocked, active brief updates
            # can accept generations, but _start_task coalesces them per mission.
            first = await _update(runtime, snapshot, "probe-update-1", "first bounded update")
            assert first["ok"], first.get("error")
            first_generation = first["result"]["snapshot"]["execution_generation"]
            second = await _update(runtime, first["result"]["snapshot"], "probe-update-2", "latest bounded update")
            assert second["ok"]
            second_snapshot = second["result"]["snapshot"]
            assert second_snapshot["execution_generation"] > first_generation
            assert runtime._pending_generations == {snapshot["mission_id"]: second_snapshot["execution_generation"]}

            paused = await _command(runtime, {
                "type": "mission_command", "schema_version": 1, "request_id": "probe-pause",
                "command": "pause", "mission_id": snapshot["mission_id"],
                "expected_revision": second_snapshot["revision"], "args": {},
            })
            assert paused["ok"]
            paused_snapshot = paused["result"]["snapshot"]
            assert paused_snapshot["state"] == "paused"

            blocked_resume = await _resume(runtime, paused_snapshot, "probe-resume-while-draining")
            assert not blocked_resume["ok"]
            assert blocked_resume["error"]["code"] == "runtime_busy"

            updated_paused = await _update(
                runtime, paused_snapshot, "probe-update-paused", "paused update stays paused"
            )
            assert updated_paused["ok"]
            assert updated_paused["result"]["execution_outcome"] == "paused"
            assert updated_paused["result"]["snapshot"]["state"] == "paused"
            retry = await _resume(runtime, updated_paused["result"]["snapshot"], "probe-resume-still-draining")
            assert not retry["ok"]
            assert retry["error"]["code"] == "runtime_busy"

            # Let scheduled coroutines/default-executor work settle while the fake
            # native operation remains blocked, then count submitted/queued work.
            await asyncio.sleep(0.03)
            with planner._counts_lock:
                started = planner.calls_started
                active = planner.native_active
                max_active = planner.max_native_active
            assert started == 1
            assert active == 1
            assert max_active == 1
            assert len(runtime._native_calls) == 1
            assert len(runtime._tasks) == 1

            # Pause invalidated the coalesced update generation. Drain the one
            # accepted native call and ensure its stale generation is discarded.
            planner.release.set()
            await _wait_for(lambda: not runtime._tasks and not runtime._pending_generations)
            with planner._counts_lock:
                assert planner.calls_started == 1
                assert planner.calls_finished == 1
                assert planner.native_active == 0
            assert not runtime._native_calls
            assert store.get_mission(snapshot["mission_id"])["state"] == "paused"
        finally:
            planner.release.set()
            await asyncio.wait_for(runtime.close(), timeout=3.0)
            store.close()

    asyncio.run(scenario())
