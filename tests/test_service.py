"""Tests for visionbrain.service (token auth + job queue) and web_app wiring.

CI-safe: no MLX, no model weights, no real subprocesses. asyncio is driven
with asyncio.run inside sync tests (the repo has no pytest-asyncio config).
"""

from __future__ import annotations

import asyncio
import json

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Token auth primitives
# ──────────────────────────────────────────────────────────────────────────────
class TestTokenAuth:
    def test_token_enabled_true_when_set(self, monkeypatch):
        from visionbrain.service import token_enabled

        monkeypatch.setenv("VB_TOKEN", "sekrit")
        assert token_enabled() is True

    def test_token_enabled_false_when_unset_or_empty(self, monkeypatch):
        from visionbrain.service import token_enabled

        monkeypatch.delenv("VB_TOKEN", raising=False)
        assert token_enabled() is False
        monkeypatch.setenv("VB_TOKEN", "")
        assert token_enabled() is False

    def test_check_token_correct(self, monkeypatch):
        from visionbrain.service import check_token

        monkeypatch.setenv("VB_TOKEN", "sekrit")
        assert check_token("sekrit") is True

    def test_check_token_wrong(self, monkeypatch):
        from visionbrain.service import check_token

        monkeypatch.setenv("VB_TOKEN", "sekrit")
        assert check_token("nope") is False

    def test_check_token_empty_or_none(self, monkeypatch):
        from visionbrain.service import check_token

        monkeypatch.setenv("VB_TOKEN", "sekrit")
        assert check_token("") is False
        assert check_token(None) is False

    def test_check_token_disabled_is_always_false(self, monkeypatch):
        from visionbrain.service import check_token

        monkeypatch.delenv("VB_TOKEN", raising=False)
        assert check_token("sekrit") is False
        assert check_token(None) is False


# ──────────────────────────────────────────────────────────────────────────────
# VB_MAX_JOBS parsing
# ──────────────────────────────────────────────────────────────────────────────
class TestMaxJobs:
    def test_default_when_unset_or_empty(self, monkeypatch):
        from visionbrain.service import MAX_JOBS_DEFAULT, max_jobs

        monkeypatch.delenv("VB_MAX_JOBS", raising=False)
        assert max_jobs() == MAX_JOBS_DEFAULT
        monkeypatch.setenv("VB_MAX_JOBS", "")
        assert max_jobs() == MAX_JOBS_DEFAULT

    def test_valid_value(self, monkeypatch):
        from visionbrain.service import max_jobs

        monkeypatch.setenv("VB_MAX_JOBS", "2")
        assert max_jobs() == 2

    def test_clamped_to_range(self, monkeypatch):
        from visionbrain.service import max_jobs

        monkeypatch.setenv("VB_MAX_JOBS", "0")
        assert max_jobs() == 1
        monkeypatch.setenv("VB_MAX_JOBS", "-3")
        assert max_jobs() == 1
        monkeypatch.setenv("VB_MAX_JOBS", "99")
        assert max_jobs() == 4

    def test_invalid_falls_back_to_default(self, monkeypatch):
        from visionbrain.service import MAX_JOBS_DEFAULT, max_jobs

        for raw in ("abc", "2.5", "one"):
            monkeypatch.setenv("VB_MAX_JOBS", raw)
            assert max_jobs() == MAX_JOBS_DEFAULT, raw


# ──────────────────────────────────────────────────────────────────────────────
# JobQueue (pure asyncio)
# ──────────────────────────────────────────────────────────────────────────────
class TestJobQueue:
    def test_immediate_acquire_returns_zero(self):
        from visionbrain.service import JobQueue

        async def main():
            q = JobQueue(2)
            assert await q.acquire("a") == 0
            assert await q.acquire("b") == 0
            q.release("a")
            q.release("b")

        asyncio.run(main())

    def test_fifo_positions_and_release_hands_off(self):
        from visionbrain.service import JobQueue

        async def main():
            q = JobQueue(1)
            assert await q.acquire("a") == 0

            b = asyncio.create_task(q.acquire("b"))
            await asyncio.sleep(0)  # let b enqueue
            c = asyncio.create_task(q.acquire("c"))
            await asyncio.sleep(0)  # let c enqueue
            assert q.queued_count == 2

            q.release("a")  # grants the slot to b
            assert await asyncio.wait_for(b, timeout=2) == 1
            assert q.queued_count == 1

            q.release("b")  # grants the slot to c
            assert await asyncio.wait_for(c, timeout=2) == 2
            assert q.queued_count == 0

            q.release("c")

        asyncio.run(main())

    def test_cancelled_waiter_is_skipped_cleanly(self):
        from visionbrain.service import JobQueue

        async def main():
            q = JobQueue(1)
            assert await q.acquire("a") == 0

            b = asyncio.create_task(q.acquire("b"))
            await asyncio.sleep(0)
            c = asyncio.create_task(q.acquire("c"))
            await asyncio.sleep(0)
            assert q.queued_count == 2

            b.cancel()
            with pytest.raises(asyncio.CancelledError):
                await b

            q.release("a")  # must skip cancelled b and grant c
            assert await asyncio.wait_for(c, timeout=2) == 2
            assert q.queued_count == 0

            # The slot is properly held by c and freed again.
            d = asyncio.create_task(q.acquire("d"))
            await asyncio.sleep(0)
            assert q.queued_count == 1  # d is waiting, c still active
            q.release("c")
            assert await asyncio.wait_for(d, timeout=2) == 1
            q.release("d")

        asyncio.run(main())

    def test_release_unknown_key_is_noop(self):
        from visionbrain.service import JobQueue

        async def main():
            q = JobQueue(1)
            q.release("ghost")
            assert await q.acquire("a") == 0
            q.release("ghost")

        asyncio.run(main())


# ──────────────────────────────────────────────────────────────────────────────
# Web app integration: auth middleware
# ──────────────────────────────────────────────────────────────────────────────
def _patch_status_dependencies(monkeypatch):
    """Keep /api/status hermetic — no model-dir stats, no backend probes."""
    import visionbrain.gemma_inference as gi
    import visionbrain.loader as loader

    monkeypatch.setattr(loader, "all_records", lambda: [])
    monkeypatch.setattr(gi, "available_backend", lambda: None)
    monkeypatch.setattr(gi, "custom_backend_configured", lambda: False)


class TestWebAppTokenAuth:
    def test_auth_off_by_default(self, monkeypatch):
        from fastapi.testclient import TestClient

        from visionbrain import web_app

        _patch_status_dependencies(monkeypatch)
        monkeypatch.delenv("VB_TOKEN", raising=False)
        with TestClient(web_app.app) as client:
            assert client.get("/api/status").status_code == 200
            assert client.get("/api/healthz").status_code == 200

    def test_api_paths_require_token(self, monkeypatch):
        from fastapi.testclient import TestClient

        from visionbrain import web_app

        _patch_status_dependencies(monkeypatch)
        monkeypatch.setenv("VB_TOKEN", "sekrit")
        with TestClient(web_app.app) as client:
            r = client.get("/api/status")
            assert r.status_code == 401
            assert r.json() == {"error": "unauthorized"}

            r = client.get("/api/status", headers={"X-Auth-Token": "wrong"})
            assert r.status_code == 401

            assert client.get("/api/status",
                              headers={"X-Auth-Token": "sekrit"}).status_code == 200
            assert client.get("/api/status?token=sekrit").status_code == 200

    def test_healthz_is_exempt(self, monkeypatch):
        from fastapi.testclient import TestClient

        from visionbrain import web_app

        monkeypatch.setenv("VB_TOKEN", "sekrit")
        with TestClient(web_app.app) as client:
            r = client.get("/api/healthz")
            assert r.status_code == 200
            assert r.json()["ok"] is True

    def test_non_api_paths_are_not_enforced(self, monkeypatch):
        from fastapi.testclient import TestClient

        from visionbrain import web_app

        monkeypatch.setenv("VB_TOKEN", "sekrit")
        with TestClient(web_app.app) as client:
            # Root serves the static shell without a token.
            assert client.get("/").status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# Web app integration: job queue on heavy endpoints
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def queued_client(monkeypatch, tmp_path):
    """TestClient with a capacity-1 queue, a stubbed _exec, and a fake upload.

    Uses the context-manager form so one event loop serves all requests —
    background _exec tasks must survive across them.
    """
    from fastapi.testclient import TestClient

    from visionbrain import service, web_app

    monkeypatch.setattr(web_app, "_job_queue", service.JobQueue(1))
    monkeypatch.setattr(web_app, "UPLOADS", tmp_path)
    (tmp_path / "clip.mp4").write_bytes(b"fake")
    log: list[str] = []
    with TestClient(web_app.app) as client:
        yield web_app, client, log


class TestWebAppQueue:
    def test_first_job_starts_immediately(self, queued_client, monkeypatch):
        web_app, client, log = queued_client

        async def fake_exec(job, cmd, outputs):
            log.append(job["id"])
            job["status"] = "done"

        monkeypatch.setattr(web_app, "_exec", fake_exec)
        r = client.post("/api/job/analyze", data={"file_id": "clip"})
        assert r.status_code == 200
        body = r.json()
        assert body["queued"] is False
        assert body["position"] == 0
        assert body["job_id"]

    def test_second_job_queues_and_reports_position(self, queued_client, monkeypatch):
        web_app, client, log = queued_client

        async def fake_exec(job, cmd, outputs):
            log.append(job["id"])
            if len(log) == 1:
                # Hold the first slot briefly so the second launch queues up.
                await asyncio.sleep(0.3)
            job["status"] = "done"

        monkeypatch.setattr(web_app, "_exec", fake_exec)

        first = client.post("/api/job/analyze", data={"file_id": "clip"}).json()
        assert first["position"] == 0 and first["queued"] is False

        # The second launch blocks in the handler until a slot frees, then
        # reports its submit-time line position.
        second = client.post("/api/job/fastscan", data={"file_id": "clip"}).json()
        assert second["queued"] is True
        assert second["position"] == 1

        assert client.get(f"/api/job/{second['job_id']}").json()["queue_position"] == 1

    def test_sse_heartbeat_carries_queue_fields(self, queued_client):
        web_app, client, _log = queued_client
        jid = "queuedjob1"
        # A job that was queued (position 2) and has finished: the stream
        # emits one heartbeat before the done event, then ends cleanly —
        # no mid-stream disconnect needed.
        web_app._jobs[jid] = {
            "id": jid, "kind": "analyze", "status": "done",
            "phase": "queued", "queue_position": 2,
            "output": [], "results": {}, "error": None,
        }
        hb_payload = None
        try:
            with client.stream("GET", f"/api/job/{jid}/stream", timeout=10) as resp:
                for line in resp.iter_lines():
                    if hb_payload is None and line.startswith("data:") and "heartbeat" in line:
                        hb_payload = json.loads(line[len("data:"):])
                    if "done" in line:
                        break
        finally:
            web_app._jobs.pop(jid, None)
        assert hb_payload is not None
        assert hb_payload["queue_position"] == 2
        assert hb_payload["queued"] is True

    def test_light_image_jobs_bypass_queue(self, queued_client, monkeypatch):
        web_app, client, log = queued_client

        async def fake_exec(job, cmd, outputs):
            log.append(job["id"])
            job["status"] = "done"

        monkeypatch.setattr(web_app, "_exec", fake_exec)
        r = client.post("/api/job/detect", data={"file_id": "clip"})
        assert r.status_code == 200
        body = r.json()
        # Light jobs keep the original response shape and never take a slot.
        assert set(body.keys()) == {"job_id", "created_at"}
        assert web_app._queue().queued_count == 0
