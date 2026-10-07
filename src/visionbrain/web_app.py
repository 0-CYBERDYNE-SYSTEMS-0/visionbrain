#!/usr/bin/env python3
"""VisionBrain Web UI — FastAPI ground control server.

Launch: python -m visionbrain ui
        or: uvicorn visionbrain.web_app:app --port 7860
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import service
from .live_engine import (
    configure as _live_configure,
    router as _live_router,
    sanitize_clip_name as _sanitize_clip_name,
)

app = FastAPI(title="VisionBrain — Aerial Ground Control", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.state.started_at = time.time()


# ── Shared-token auth (VB_TOKEN; off by default) ──────────────────────────────
@app.middleware("http")
async def _token_auth(request: Request, call_next) -> Response:
    """Enforce the shared token on /api/* (except /api/healthz) when enabled.

    Token sources, in order: the X-Auth-Token header, then the ?token= query
    parameter. WebSocket scopes never pass through HTTP middleware, so the
    live WebSocket route enforces the same token in-handler (?token=,
    close 4401 before accept) — see DEPLOY.md ("Auth (shared token)").
    """
    if service.token_enabled():
        path = request.url.path
        if (
            (path.startswith("/api/") and path != "/api/healthz")
            or path.startswith("/uploads/")
        ):
            provided = request.headers.get("x-auth-token") or request.query_params.get("token")
            if not service.check_token(provided):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


# ── Job queue (VB_MAX_JOBS heavy-subprocess slot limiter) ─────────────────────
_job_queue: Optional[service.JobQueue] = None


def _queue() -> service.JobQueue:
    """Lazily-built queue singleton so VB_MAX_JOBS is read at first use."""
    global _job_queue
    if _job_queue is None:
        _job_queue = service.JobQueue(service.max_jobs())
    return _job_queue

# ── Directories ────────────────────────────────────────────────────────────────
WORK_DIR   = Path(tempfile.gettempdir()) / "visionbrain_ui"
UPLOADS    = WORK_DIR / "uploads"
RESULTS    = WORK_DIR / "results"
CLIPS      = WORK_DIR / "clips"
for d in (WORK_DIR, UPLOADS, RESULTS, CLIPS):
    d.mkdir(exist_ok=True)

PYTHON     = sys.executable          # same env that launched us
STATIC_DIR = Path(__file__).parent / "static"

# ── Local live engine (SAM 3.1 streamed over /api/live/ws) ────────────────────
_live_configure(UPLOADS, CLIPS)
app.include_router(_live_router)

# ── Job store ──────────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_MAX_JOBS_REMEMBERED = 200  # finished jobs evicted (oldest first) past this cap


def _new_job(kind: str) -> dict:
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    job = dict(id=jid, kind=kind, status="pending",
                ts=now, started_at=None, ended_at=None,
                last_heartbeat_at=now, last_output_at=None, phase="pending",
                output=[], results={}, error=None)
    _jobs[jid] = job
    _prune_jobs()
    return job


def _prune_jobs() -> None:
    """Keep the job store bounded — full stdout logs otherwise accumulate
    for the life of a long-running hub. Never evicts unfinished jobs."""
    excess = len(_jobs) - _MAX_JOBS_REMEMBERED
    if excess <= 0:
        return
    finished = sorted(
        (j for j in _jobs.values() if j.get("status") in ("done", "error")),
        key=lambda j: j.get("ts", 0.0),
    )
    for job in finished[:excess]:
        _jobs.pop(job["id"], None)


async def _exec(job: dict, cmd: list[str], outputs: dict[str, str]) -> None:
    """Run a job's subprocess, mapping every failure into job state.

    A launch error (spawn failure, cancelled task) must leave the job in
    ``error`` — an exception escaping here would strand the job as
    ``running`` forever, heartbeating an SSE stream that never ends.
    """
    try:
        await _exec_run(job, cmd, outputs)
    except Exception as exc:  # noqa: BLE001 — job state is the error channel
        job["status"] = "error"
        job["phase"] = "error"
        job["ended_at"] = time.time()
        job["last_heartbeat_at"] = time.time()
        job["error"] = f"launch failed: {exc}"


async def _exec_run(job: dict, cmd: list[str], outputs: dict[str, str]) -> None:
    """Run CLI command async; stream stdout into job.output[]."""
    now = time.time()
    job["status"] = "running"
    job["started_at"] = now
    job["last_heartbeat_at"] = now
    job["phase"] = "running"
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    while True:
        job["last_heartbeat_at"] = time.time()
        if proc.stdout is None:
            break
        try:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
        except asyncio.TimeoutError:
            if proc.returncode is not None:
                break
            continue
        if not raw:
            if proc.returncode is not None:
                break
            continue
        job["last_output_at"] = time.time()
        job["output"].append(raw.decode("utf-8", errors="replace").rstrip())

    await proc.wait()
    job["last_heartbeat_at"] = time.time()
    job["ended_at"] = time.time()
    if proc.returncode == 0:
        job["status"] = "done"
        job["phase"] = "done"
        for k, p in outputs.items():
            if p and Path(p).exists():
                job["results"][k] = p
    else:
        job["status"] = "error"
        job["phase"] = "error"
        job["error"] = f"exit {proc.returncode}"


async def _run_queued(job: dict, cmd: list[str], outputs: dict[str, str]) -> int:
    """Acquire a queue slot for *job*, then run its subprocess in background.

    The handler awaits this only until a slot is granted, so the launch
    response can report the submit-time queue position (returned); the
    subprocess itself keeps running via create_task and releases the slot
    in a done-callback, so a client disconnect never leaks the slot.
    """
    queue = _queue()
    job["phase"] = "queued"
    position = await queue.acquire(job["id"])
    job["queue_position"] = position
    task = asyncio.create_task(_exec(job, cmd, outputs))
    task.add_done_callback(lambda _t: queue.release(job["id"]))
    return position


def _launch_payload(job: dict, position: int) -> dict:
    """Standard response body for a job-launch endpoint."""
    return {"job_id": job["id"], "created_at": job["ts"],
            "queued": position > 0, "queue_position": position}


def _job_queue_position(job: dict) -> Optional[int]:
    """Live line spot while *job* still waits, else its submit-time position.

    While queued the recorded submit-time value goes stale as jobs ahead
    finish, so poll the queue; once a slot is granted the recorded value
    is returned (real _exec flips the phase to running immediately).
    """
    if job.get("phase") == "queued" and job.get("queue_position") is None:
        return _queue().wait_position(job["id"])
    return job.get("queue_position")


# ── Status ─────────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    from .loader import all_records
    from .gemma_inference import available_backend, custom_backend_configured

    # all_records() stats multi-GB model dirs; available_backend() does network
    # probes. Both are blocking — keep them off the event loop. One
    # available_backend() call feeds both the gemma flag and the vlm dict.
    def _backend_status() -> tuple[bool, dict]:
        backend = available_backend()
        return (
            backend is not None,
            {"backend": backend or "none", "custom_configured": custom_backend_configured()},
        )

    recs, (gemma_ok, vlm) = await asyncio.gather(
        asyncio.to_thread(all_records),
        asyncio.to_thread(_backend_status),
    )
    return {
        "models": [
            dict(id=r.hf_id, name=r.hf_id.split("/")[-1],
                 ready=r.can_load, cached=r.is_cached,
                 gb=r.disk_gb, note=r.note)
            for r in recs
        ],
        "gemma_remote": gemma_ok,
        "vlm": vlm,
    }

@app.get("/api/healthz")
async def api_healthz():
    now = time.time()
    running_jobs = sum(1 for j in _jobs.values() if j.get("status") == "running")
    return {
        "ok": True,
        "server_time": now,
        "uptime_s": round(now - app.state.started_at, 3),
        "running_jobs": running_jobs,
    }


# ── VLM settings (custom OpenAI-compatible backend) ───────────────────────────
def _settings_payload() -> dict:
    """Redacted view of the saved VLM settings — the api_key never leaves here."""
    from .gemma_inference import custom_backend_configured, load_vlm_settings

    settings = load_vlm_settings()
    return {
        "configured": custom_backend_configured(),
        "base_url": settings["base_url"],
        "model": settings["model"],
        "has_key": bool(settings["api_key"]),
    }


@app.get("/api/settings")
async def api_get_settings() -> dict:
    return _settings_payload()


@app.post("/api/settings")
async def api_set_settings(request: Request) -> dict:
    from .gemma_inference import save_vlm_settings

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(400, "Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON body must be an object")

    fields = {
        "base_url": body.get("base_url", ""),
        "model": body.get("model", ""),
        "api_key": body.get("api_key", ""),
    }
    for name, value in fields.items():
        if not isinstance(value, str):
            raise HTTPException(400, f"'{name}' must be a string")
    clear_key = body.get("clear_key", False)
    if not isinstance(clear_key, bool):
        raise HTTPException(400, "'clear_key' must be a boolean")

    save_vlm_settings(
        base_url=fields["base_url"],
        model=fields["model"],
        api_key=fields["api_key"],
        clear_key=clear_key,
    )
    return _settings_payload()


# ── File upload ────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    fid  = uuid.uuid4().hex[:8]
    suf  = Path(file.filename or "file").suffix
    dest = UPLOADS / f"{fid}{suf}"
    data = await file.read()
    dest.write_bytes(data)
    return {"file_id": fid, "name": file.filename, "size": len(data), "suffix": suf}


def _find_upload(fid: str) -> Path:
    """Resolve a file_id to its uploaded file, never a sidecar directory.

    ``{fid}*`` can match more than the upload itself: the analyze pipeline
    writes a ``{fid}_stills`` directory beside the video. Skip directories
    (scandir order is arbitrary, so ``matches[0]`` was a coin flip between
    the media file and the stills dir) and prefer the shortest name when
    several files share the prefix (``{fid}.mp4`` beats ``{fid}_small.mp4``).
    """
    files = [p for p in UPLOADS.glob(f"{fid}*") if p.is_file()]
    if not files:
        raise HTTPException(404, "Upload not found")
    return min(files, key=lambda p: (len(p.name), p.name))


# ── Analyze ────────────────────────────────────────────────────────────────────
@app.post("/api/job/analyze")
async def job_analyze(
    file_id:        str   = Form(...),
    query:          str   = Form("people and vehicles"),
    question:       str   = Form(""),
    prompts:        str   = Form("person vehicle animal"),
    threshold:      float = Form(0.05),
    resolution:     int   = Form(512),
    every:          int   = Form(5),
    backbone_every: int   = Form(1),
    opacity:        float = Form(0.6),
    report:         bool  = Form(True),
    report_type:    str   = Form("field"),
    falcon_refine:  bool  = Form(False),
    falcon_frames:  int   = Form(6),
    max_tokens:     int   = Form(512),
    include_video:  bool  = Form(True),   # UI plays/downloads the annotated mp4
    # ── Fast-path + adaptive ────────────────────────────────
    fast:           bool  = Form(False),
    fast_output:    str   = Form(""),
    adaptive:       bool  = Form(False),
    motion_threshold: float = Form(0.03),
    propagate:      int   = Form(0),
    relevance_filter: bool = Form(False),
    parallel_falcon: bool = Form(True),
    # ── Chunking (large video support) ──────────────────────────
    chunk_duration: int  = Form(0),     # 0 = auto-detect
    chunk_overlap:  int  = Form(3),
):
    src = _find_upload(file_id)
    job = _new_job("analyze")
    jid = job["id"]
    out_v = str(RESULTS / f"{jid}_analyzed.mp4")
    out_j = str(RESULTS / f"{jid}_detections.json")
    out_r = str(RESULTS / f"{jid}_report.txt")
    out_f = str(RESULTS / f"{jid}_fast.json") if fast or fast_output else ""

    cmd = [PYTHON, "-u", "-m", "visionbrain", "analyze",
           "--video", str(src),
           "--query", query,
           "--prompts", *prompts.split(),
           "--output", out_v,
           "--json-output", out_j,
           "--report-output", out_r,
           "--threshold", str(threshold),
           "--every", str(every),
           "--backbone-every", str(backbone_every),
           "--resolution", str(resolution),
           "--opacity", str(opacity),
           "--report-type", report_type,
           "--max-tokens", str(max_tokens)]
    if report:
        cmd.append("--report")
    if include_video:
        cmd.append("--include-video")
    if falcon_refine:
        cmd += ["--falcon-refine", "--falcon-frames", str(falcon_frames)]
    if fast:
        cmd.append("--fast")
        if out_f:
            cmd += ["--fast-output", out_f]
    if adaptive:
        cmd.append("--adaptive")
    if motion_threshold != 0.03:
        cmd += ["--motion-threshold", str(motion_threshold)]
    if propagate > 0:
        cmd += ["--propagate", str(propagate)]
    if relevance_filter:
        cmd.append("--relevance-filter")
    if not parallel_falcon:
        cmd.append("--sequential-falcon")
    if question.strip():
        cmd += ["--question", question]
    if chunk_duration != 0:
        cmd += ["--chunk-duration", str(chunk_duration)]
    if chunk_overlap != 3:
        cmd += ["--chunk-overlap", str(chunk_overlap)]

    position = await _run_queued(job, cmd, {"video": out_v, "json": out_j, "report": out_r,
                                            "fast_json": out_f if out_f else ""})
    return _launch_payload(job, position)


# ── FastScan ──────────────────────────────────────────────────────────────────
@app.post("/api/job/fastscan")
async def job_fastscan(
    file_id:       str   = Form(...),
    query:         str   = Form("person"),
    every:         float = Form(5.0),
    max_frames:    int   = Form(60),
    resolution:    int   = Form(360),
    min_relevance: float = Form(0.2),
):
    src = _find_upload(file_id)
    job = _new_job("fastscan")
    jid = job["id"]
    out = str(RESULTS / f"{jid}_fast.json")

    cmd = [PYTHON, "-u", "-m", "visionbrain", "fastscan",
           "--video", str(src),
           "--query", query,
           "--every", str(every),
           "--max-frames", str(max_frames),
           "--resolution", str(resolution),
           "--min-relevance", str(min_relevance),
           "--output", out]

    position = await _run_queued(job, cmd, {"fast_json": out})
    return _launch_payload(job, position)


# ── Detect ─────────────────────────────────────────────────────────────────────
@app.post("/api/job/detect")
async def job_detect(
    file_id:    str = Form(...),
    query:      str = Form("person"),
    max_tokens: int = Form(200),
):
    src = _find_upload(file_id)
    job = _new_job("detect")
    jid = job["id"]
    out = str(RESULTS / f"{jid}_detected.jpg")
    cmd = [PYTHON, "-u", "-m", "visionbrain", "detect",
           "--image", str(src), "--query", query,
           "--max-tokens", str(max_tokens), "--output", out]
    asyncio.create_task(_exec(job, cmd, {"image": out}))
    return {"job_id": jid, "created_at": job["ts"]}


# ── Segment ────────────────────────────────────────────────────────────────────
@app.post("/api/job/segment")
async def job_segment(
    file_id:    str = Form(...),
    query:      str = Form("person"),
    max_tokens: int = Form(2048),
):
    src = _find_upload(file_id)
    job = _new_job("segment")
    jid = job["id"]
    out = str(RESULTS / f"{jid}_segmented.jpg")
    cmd = [PYTHON, "-u", "-m", "visionbrain", "segment",
           "--image", str(src), "--query", query,
           "--max-tokens", str(max_tokens), "--output", out]
    asyncio.create_task(_exec(job, cmd, {"image": out}))
    return {"job_id": jid, "created_at": job["ts"]}


# ── OCR ────────────────────────────────────────────────────────────────────────
@app.post("/api/job/ocr")
async def job_ocr(
    file_id:  str = Form(...),
    question: str = Form("read all text in the image"),
):
    src = _find_upload(file_id)
    job = _new_job("ocr")
    jid = job["id"]
    cmd = [PYTHON, "-u", "-m", "visionbrain", "ocr",
           "--image", str(src), "--question", question]
    asyncio.create_task(_exec(job, cmd, {}))
    return {"job_id": jid, "created_at": job["ts"]}


# ── Track ──────────────────────────────────────────────────────────────────────
@app.post("/api/job/track")
async def job_track(
    file_id:          str   = Form(...),
    prompts:          str   = Form("person"),
    threshold:        float = Form(0.15),
    every:            int   = Form(2),
    resolution:       int   = Form(1008),
    opacity:          float = Form(0.6),
    backbone_every:   int   = Form(1),
    json_output:      bool  = Form(False),
    supervision:      bool  = Form(False),
    persistent_ids:   bool  = Form(False),
    adaptive_motion:  bool  = Form(False),
    motion_threshold: float = Form(0.03),
    propagate:        int   = Form(0),
):
    src = _find_upload(file_id)
    job = _new_job("track")
    jid = job["id"]
    out = str(RESULTS / f"{jid}_tracked.mp4")
    out_j = str(RESULTS / f"{jid}_detections.json")
    cmd = [PYTHON, "-u", "-m", "visionbrain", "track",
           "--video", str(src),
           "--prompts", *prompts.split(),
           "--output", out,
           "--threshold", str(threshold),
           "--every", str(every),
           "--resolution", str(resolution),
           "--opacity", str(opacity),
           "--backbone-every", str(backbone_every)]
    if json_output:
        cmd += ["--json-output", out_j]
    if supervision:
        cmd.append("--supervision")
    if persistent_ids:
        cmd.append("--persistent-ids")
    if adaptive_motion:
        cmd.append("--adaptive-motion")
        if motion_threshold != 0.03:
            cmd += ["--motion-threshold", str(motion_threshold)]
    if propagate > 0:
        cmd += ["--propagate", str(propagate)]
    position = await _run_queued(job, cmd, {"video": out, "json": out_j if json_output else ""})
    return _launch_payload(job, position)


# ── SAM-3 ──────────────────────────────────────────────────────────────────────
@app.post("/api/job/sam3")
async def job_sam3(
    file_id:    str   = Form(...),
    prompts:    str   = Form("person"),
    task:       str   = Form("detect"),
    threshold:  float = Form(0.15),
    resolution: int   = Form(1008),
):
    src = _find_upload(file_id)
    job = _new_job("sam3")
    jid = job["id"]
    out = str(RESULTS / f"{jid}_sam3.jpg")
    cmd = [PYTHON, "-u", "-m", "visionbrain", "sam3",
           "--image", str(src),
           "--prompts", *prompts.split(),
           "--task", task,
           "--threshold", str(threshold),
           "--resolution", str(resolution),
           "--output", out]
    asyncio.create_task(_exec(job, cmd, {"image": out}))
    return {"job_id": jid, "created_at": job["ts"]}


# ── Agent ──────────────────────────────────────────────────────────────────────
@app.post("/api/job/agent")
async def job_agent(
    file_id:  str = Form(...),
    question: str = Form("what do you see?"),
    api_key:  str = Form(""),
    model:    str = Form(""),
    base_url: str = Form(""),
) -> dict:
    src = _find_upload(file_id)
    job = _new_job("agent")
    jid = job["id"]
    out = str(RESULTS / f"{jid}_agent.jpg")
    # Form fields override saved settings; read the settings file only when a
    # form field is empty.
    if not (api_key and model and base_url):
        from .gemma_inference import load_vlm_settings

        stored = load_vlm_settings()
        api_key = api_key or stored["api_key"]
        model = model or stored["model"]
        base_url = base_url or stored["base_url"]
    cmd = [PYTHON, "-u", "-m", "visionbrain", "agent",
           "--image", str(src), "--question", question, "--output", out]
    if api_key:
        cmd += ["--api-key", api_key]
    if model:
        cmd += ["--model", model]
    if base_url:
        cmd += ["--base-url", base_url]
    position = await _run_queued(job, cmd, {"image": out})
    return _launch_payload(job, position)


# ── Job query & SSE ────────────────────────────────────────────────────────────
@app.get("/api/job/{jid}")
async def get_job(jid: str):
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404)
    payload = {k: v for k, v in job.items() if k != "_proc"}
    payload["queue_position"] = _job_queue_position(job)
    payload["queued"] = job.get("phase") == "queued"
    return payload


def _result_path(jid: str, kind: str) -> Path:
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404)
    path = job["results"].get(kind)
    if not path or not Path(path).exists():
        raise HTTPException(404, f"No result '{kind}'")
    return Path(path)


@app.get("/api/job/{jid}/detections")
async def get_detections(jid: str):
    path = _result_path(jid, "json")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"Invalid detection JSON: {exc}") from exc


@app.get("/api/job/{jid}/report")
async def get_report(jid: str):
    path = _result_path(jid, "report")
    return {"text": path.read_text()}


@app.get("/api/job/{jid}/fast")
async def get_fast(jid: str):
    path = _result_path(jid, "fast_json")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"Invalid fast-scan JSON: {exc}") from exc


@app.get("/api/job/{jid}/stream")
async def stream_job(jid: str, request: Request):
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404)
    sent = 0
    last_hb_emit = 0.0

    async def gen() -> AsyncGenerator[str, None]:
        nonlocal sent, last_hb_emit
        while True:
            if await request.is_disconnected():
                break
            lines = job["output"]
            if len(lines) > sent:
                for ln in lines[sent:]:
                    yield f"data: {json.dumps({'type':'log','msg':ln})}\n\n"
                sent = len(lines)
            now = time.time()
            if now - last_hb_emit >= 1.0:
                started_at = job.get("started_at") or now
                hb = {
                    "type": "heartbeat",
                    "status": job["status"],
                    "phase": job.get("phase", "running"),
                    "queue_position": _job_queue_position(job),
                    "queued": job.get("phase") == "queued",
                    "ts": now,
                    "last_heartbeat_at": job.get("last_heartbeat_at", now),
                    "last_output_at": job.get("last_output_at"),
                    "elapsed_s": round(max(0.0, now - started_at), 1),
                }
                yield f"data: {json.dumps(hb)}\n\n"
                last_hb_emit = now
            if job["status"] in ("done", "error"):
                yield f"data: {json.dumps({'type':'done','status':job['status'],'results':job['results'],'error':job['error']})}\n\n"
                break
            await asyncio.sleep(0.08)

    return StreamingResponse(gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── File serving ───────────────────────────────────────────────────────────────
@app.get("/api/job/{jid}/file/{kind}")
async def serve_file(jid: str, kind: str):
    return FileResponse(_result_path(jid, kind))


@app.get("/uploads/{fid}")
async def serve_upload(fid: str):
    # _find_upload skips sidecar directories and prefers the shortest name —
    # the raw matches[0] here once served the {fid}_stills dir as a 500.
    return FileResponse(str(_find_upload(fid)))


@app.get("/api/clips/{name}")
async def serve_clip(name: str) -> FileResponse:
    # sanitize_clip_name allows only [A-Za-z0-9_.-] + ".mp4" — no path
    # separators, no traversal; anything else is a 404.
    clean = _sanitize_clip_name(name)
    if clean is None:
        raise HTTPException(404)
    path = CLIPS / clean
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(str(path))


# ── Static + root ──────────────────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    p = STATIC_DIR / "index.html"
    if p.exists():
        # FileResponse streams from the threadpool; read_text() blocked the loop.
        return FileResponse(str(p), media_type="text/html")
    return HTMLResponse("<h1>VisionBrain</h1><p>index.html not found.</p>")


# ── Dev runner ─────────────────────────────────────────────────────────────────
def run(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    import threading
    import webbrowser
    import uvicorn

    if open_browser:
        def _open() -> None:
            time.sleep(1.4)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    app.state.started_at = time.time()
    uvicorn.run(app, host=host, port=port, log_level="warning")
