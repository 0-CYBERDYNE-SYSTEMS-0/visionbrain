"""CPU-only regressions for local live inference ownership and drain."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from visionbrain.inference_admission import InferenceAdmission


@contextmanager
def _held_by_child(lock_path: Path):
    """Hold a real flock in another process until the context exits."""
    source_dir = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_dir), env.get("PYTHONPATH", "")]
    )
    code = (
        "import sys; "
        "from visionbrain.inference_admission import InferenceAdmission; "
        "handle = InferenceAdmission(sys.argv[1]).try_acquire('external-owner'); "
        "assert handle is not None; "
        "print('ready', flush=True); "
        "sys.stdin.read(); "
        "handle.release()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(5), "external owner did not acquire admission"
        assert process.stdout.readline().strip() == "ready"
        yield
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        if process.stderr is not None:
            process.stderr.close()
        if process.stdout is not None:
            process.stdout.close()


def _external_probe(lock_path: Path) -> str:
    source_dir = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_dir), env.get("PYTHONPATH", "")]
    )
    code = (
        "import sys; "
        "from visionbrain.inference_admission import InferenceAdmission; "
        "handle = InferenceAdmission(sys.argv[1]).try_acquire('probe'); "
        "print('busy' if handle is None else 'acquired', flush=True); "
        "handle.release() if handle is not None else None"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(lock_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def live_state(monkeypatch, tmp_path):
    import visionbrain.live_engine as le

    monkeypatch.setenv("VB_INFERENCE_LOCK", str(tmp_path / "inference.lock"))
    monkeypatch.setattr(le, "_worker", None)
    monkeypatch.setattr(le, "_sinks", set())
    monkeypatch.setattr(le, "_pending_zones", None)
    monkeypatch.setattr(le, "_pending_triggers", None)
    monkeypatch.setattr(le, "_pending_targets", None)
    monkeypatch.setattr(le, "_ask_busy", False)
    monkeypatch.setattr(le, "_watch_cfg", dict(le.WATCH_DEFAULTS))
    monkeypatch.setattr(le, "_watch_thread", None)
    monkeypatch.setattr(le, "_watch_stop", None)
    monkeypatch.setattr(le, "_watch_busy", False)
    monkeypatch.setattr(le, "_ENGINE_JOIN_TIMEOUT_S", 0.05)
    return le


def _client(le):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(le.router)
    return TestClient(app)


def _start(ws) -> None:
    ws.send_text(json.dumps({
        "type": "start", "source": "webcam", "camera": 0, "prompts": ["car"]
    }))


def _wait(event: threading.Event, note: str, timeout: float = 3.0) -> None:
    assert event.wait(timeout), note


def _wait_dead(worker, timeout: float = 3.0) -> None:
    assert worker.thread is not None
    worker.thread.join(timeout)
    assert not worker.thread.is_alive(), "live worker did not drain"


def test_external_owner_blocks_live_start_without_running_native_body(
    live_state, monkeypatch, tmp_path
) -> None:
    le = live_state
    lock_path = tmp_path / "inference.lock"
    native_calls = []
    monkeypatch.setattr(
        le._EngineWorker, "_run", lambda self: native_calls.append(self)
    )
    client = _client(le)
    with _held_by_child(lock_path):
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            _start(ws)
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "status"
            assert "inference admission busy" in reply["note"]
            assert "external-owner" in reply["note"]
            assert le._worker is None
            assert native_calls == []


def test_live_admission_filesystem_error_fails_closed(live_state, monkeypatch, tmp_path):
    le = live_state
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    monkeypatch.setenv(
        "VB_INFERENCE_LOCK", str(blocker / "child" / "inference.lock")
    )
    native_calls = []
    monkeypatch.setattr(
        le._EngineWorker, "_run", lambda self: native_calls.append(self)
    )
    client = _client(le)
    with client.websocket_connect("/api/live/ws") as ws:
        ws.receive_text()
        _start(ws)
        reply = json.loads(ws.receive_text())
        assert "inference admission unavailable" in reply["note"]
        assert native_calls == []
        assert le._worker is None


def test_thread_start_failure_releases_live_admission(live_state, monkeypatch, tmp_path):
    le = live_state
    original_start = threading.Thread.start
    native_calls = []
    monkeypatch.setattr(
        le._EngineWorker, "_run", lambda self: native_calls.append(self)
    )

    def fail_worker_start(thread):
        if thread.name == "live-engine-worker":
            raise RuntimeError("thread start rejected")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_worker_start)
    client = _client(le)
    with client.websocket_connect("/api/live/ws") as ws:
        ws.receive_text()
        _start(ws)
        reply = json.loads(ws.receive_text())
        assert "failed to start" in reply["note"]
        assert le._worker is None
        assert native_calls == []
        assert _external_probe(tmp_path / "inference.lock") == "acquired"


def test_stopping_before_worker_body_skips_native_imports_and_releases(live_state, tmp_path):
    le = live_state
    native_calls = []
    worker = le._EngineWorker(
        {"source": "webcam", "camera": 0, "prompts": ["car"]}, set()
    )
    worker._admission_handle = InferenceAdmission(
        tmp_path / "inference.lock"
    ).try_acquire("test-live")
    assert worker._admission_handle is not None
    worker._run = lambda: native_calls.append(True)
    worker.stop_event.set()

    worker._thread_entry()

    assert native_calls == []
    assert _external_probe(tmp_path / "inference.lock") == "acquired"


def test_shutdown_timeout_retains_worker_cache_and_admission_until_retry(
    live_state, monkeypatch, tmp_path
) -> None:
    le = live_state
    entered = threading.Event()
    release_native = threading.Event()

    def blocked_body(_worker) -> None:
        entered.set()
        release_native.wait(3.0)

    monkeypatch.setattr(le._EngineWorker, "_run", blocked_body)
    cache_module = ModuleType("visionbrain.sam3_inference")
    cache_module._sam_model_cache = {"sentinel": object()}
    monkeypatch.setitem(sys.modules, "visionbrain.sam3_inference", cache_module)
    client = _client(le)
    try:
        with client.websocket_connect("/api/live/ws") as ws:
            ws.receive_text()
            _start(ws)
            _wait(entered, "native body did not enter")
            worker = le._worker
            assert worker is not None
            ws.send_text(json.dumps({"type": "shutdown"}))
            draining = json.loads(ws.receive_text())
            assert "draining" in draining["note"]
            assert le._worker is worker
            assert worker.thread.is_alive()
            assert "sentinel" in cache_module._sam_model_cache
            assert _external_probe(tmp_path / "inference.lock") == "busy"

            worker._latest_frame = (1, object())
            worker._last_frame_items = []
            monkeypatch.setattr(
                le, "_vlm_label", lambda: pytest.fail("late ask reached VLM")
            )
            ws.send_text(json.dumps({"type": "ask", "question": "late?"}))
            refused = json.loads(ws.receive_text())
            assert refused == {
                "type": "error", "error": "engine is stopping — ask refused"
            }

            release_native.set()
            _wait_dead(worker)
            ws.send_text(json.dumps({"type": "shutdown"}))
            while True:
                freed = json.loads(ws.receive_text())
                if freed.get("type") == "status" and "note" in freed:
                    break
            assert freed["note"] == "model freed"
            assert le._worker is None
            assert cache_module._sam_model_cache == {}
            assert _external_probe(tmp_path / "inference.lock") == "acquired"
    finally:
        release_native.set()


def test_accepted_ask_drains_after_detector_and_holds_admission(live_state, monkeypatch, tmp_path):
    le = live_state
    detector_entered = threading.Event()
    ask_entered = threading.Event()
    release_ask = threading.Event()
    events = []

    def detector_body(worker) -> None:
        detector_entered.set()
        worker.stop_event.wait(30.0)

    monkeypatch.setattr(le._EngineWorker, "_run", detector_body)
    client = _client(le)
    try:
        with client.websocket_connect("/api/live/ws") as ws:
            ws.receive_text()
            _start(ws)
            _wait(detector_entered, "detector worker did not start")
            worker = le._worker
            assert worker is not None
            original_push = worker.push

            def record_push(message) -> None:
                events.append(message)
                original_push(message)

            worker.push = record_push
            worker._latest_frame = (1, object())
            worker._last_frame_items = []
            import visionbrain.vlm_registry as vlmr

            def blocked_ask(*_args, **_kwargs):
                ask_entered.set()
                release_ask.wait(3.0)
                return "done"

            monkeypatch.setattr(vlmr, "ask", blocked_ask)
            monkeypatch.setattr(vlmr, "current_key", lambda: "gemma")
            ws.send_text(json.dumps({"type": "ask", "question": "what?"}))
            assert json.loads(ws.receive_text())["type"] == "ask_ack"
            _wait(ask_entered, "accepted Ask did not start")

            ws.send_text(json.dumps({"type": "stop"}))
            _wait(worker.stop_event, "stop was not accepted")
            deadline = time.monotonic() + 3.0
            while worker.thread.is_alive() and time.monotonic() < deadline:
                if worker._is_stopping():
                    break
                time.sleep(0.005)
            assert worker._is_stopping()
            assert worker.thread.is_alive()
            assert not any(m.get("type") == "engine_stopped" for m in events)
            assert not worker._register_native_job()
            assert _external_probe(tmp_path / "inference.lock") == "busy"

            release_ask.set()
            _wait_dead(worker)
            assert [m["type"] for m in events if m.get("type") in {
                "answer", "engine_stopped"
            }] == ["answer", "engine_stopped"]
            assert _external_probe(tmp_path / "inference.lock") == "acquired"
    finally:
        release_ask.set()


def test_cancelled_ask_ack_runs_last_viewer_cleanup_and_drains(live_state, monkeypatch, tmp_path):
    le = live_state
    detector_entered = threading.Event()
    ack_entered = threading.Event()
    release_ack = threading.Event()

    def detector_body(worker) -> None:
        detector_entered.set()
        worker.stop_event.wait(30.0)

    monkeypatch.setattr(le._EngineWorker, "_run", detector_body)
    real_safe_send = le._safe_send_json

    async def cancel_during_ack(websocket, message):
        if message.get("type") == "ask_ack":
            ack_entered.set()
            await asyncio.to_thread(release_ack.wait)
            raise asyncio.CancelledError()
        return await real_safe_send(websocket, message)

    monkeypatch.setattr(le, "_safe_send_json", cancel_during_ack)
    client = _client(le)
    cancelled = False
    try:
        with client.websocket_connect("/api/live/ws") as ws:
            ws.receive_text()
            _start(ws)
            _wait(detector_entered, "detector worker did not start")
            worker = le._worker
            assert worker is not None
            worker._latest_frame = (1, object())
            worker._last_frame_items = []
            ws.send_text(json.dumps({"type": "ask", "question": "cancel me"}))
            _wait(ack_entered, "Ask acknowledgment was not reached")
            assert worker._native_job_count == 1
            release_ack.set()
            _wait(worker.stop_event, "cancelled last viewer did not request worker stop")
            _wait_dead(worker)
            assert worker._native_job_count == 0
            assert not le._sinks
            assert _external_probe(tmp_path / "inference.lock") == "acquired"
    except concurrent.futures.CancelledError:
        cancelled = True
    finally:
        release_ack.set()
    assert cancelled, "live_ws swallowed cancellation during Ask acknowledgment"
    assert le._worker is None or le._worker is worker
    assert not worker.thread.is_alive()


def test_cancelled_receive_runs_last_viewer_cleanup_and_drains(
    live_state, monkeypatch, tmp_path
) -> None:
    le = live_state
    detector_entered = threading.Event()
    ask_entered = threading.Event()
    release_ask = threading.Event()
    receive_entered = threading.Event()
    release_receive = threading.Event()
    original_receive_text = le.WebSocket.receive_text
    receive_calls = 0
    events = []

    def detector_body(worker) -> None:
        detector_entered.set()
        worker.stop_event.wait(30.0)

    async def cancel_receive_after_start(websocket):
        nonlocal receive_calls
        receive_calls += 1
        if receive_calls <= 2:
            return await original_receive_text(websocket)
        receive_entered.set()
        await asyncio.to_thread(release_receive.wait)
        raise asyncio.CancelledError()

    monkeypatch.setattr(le._EngineWorker, "_run", detector_body)
    monkeypatch.setattr(le.WebSocket, "receive_text", cancel_receive_after_start)
    client = _client(le)
    cancelled = False
    try:
        with client.websocket_connect("/api/live/ws") as ws:
            ws.receive_text()
            _start(ws)
            _wait(detector_entered, "detector worker did not start")
            worker = le._worker
            assert worker is not None
            original_push = worker.push

            def record_push(message) -> None:
                events.append(message)
                original_push(message)

            worker.push = record_push
            worker._latest_frame = (1, object())
            worker._last_frame_items = []
            import visionbrain.vlm_registry as vlmr

            def blocked_ask(*_args, **_kwargs):
                ask_entered.set()
                release_ask.wait(30.0)
                return "done"

            monkeypatch.setattr(vlmr, "ask", blocked_ask)
            monkeypatch.setattr(vlmr, "current_key", lambda: "gemma")
            ws.send_text(json.dumps({"type": "ask", "question": "hold owner"}))
            assert json.loads(ws.receive_text())["type"] == "ask_ack"
            _wait(ask_entered, "accepted Ask did not start")
            assert worker._native_job_count == 1
            _wait(receive_entered, "receive_text cancellation point was not reached")
            release_receive.set()
            _wait(worker.stop_event, "cancelled last viewer did not request worker stop")
            assert worker.thread.is_alive()
            assert _external_probe(tmp_path / "inference.lock") == "busy"
            release_ask.set()
            _wait_dead(worker)
            assert worker._native_job_count == 0
            assert [m["type"] for m in events if m.get("type") in {
                "answer", "engine_stopped"
            }] == ["answer", "engine_stopped"]
            assert not le._sinks
            assert _external_probe(tmp_path / "inference.lock") == "acquired"
    except concurrent.futures.CancelledError:
        cancelled = True
    finally:
        release_receive.set()
        release_ask.set()
    assert cancelled, "live_ws swallowed cancellation from receive_text"
    assert le._worker is None or le._worker is worker
    assert not worker.thread.is_alive()
