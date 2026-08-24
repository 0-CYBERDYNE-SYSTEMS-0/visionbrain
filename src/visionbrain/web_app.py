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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="VisionBrain — Aerial Ground Control", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.state.started_at = time.time()

# ── Directories ────────────────────────────────────────────────────────────────
WORK_DIR   = Path(tempfile.gettempdir()) / "visionbrain_ui"
UPLOADS    = WORK_DIR / "uploads"
RESULTS    = WORK_DIR / "results"
for d in (WORK_DIR, UPLOADS, RESULTS):
    d.mkdir(exist_ok=True)

PYTHON     = sys.executable          # same env that launched us
STATIC_DIR = Path(__file__).parent / "static"

# ── Job store ──────────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}


def _new_job(kind: str) -> dict:
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    job = dict(id=jid, kind=kind, status="pending",
                ts=now, started_at=None, ended_at=None,
                last_heartbeat_at=now, last_output_at=None, phase="pending",
                output=[], results={}, error=None)
    _jobs[jid] = job
    return job


async def _exec(job: dict, cmd: list[str], outputs: dict[str, str]) -> None:
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


# ── Status ─────────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    from .loader import all_records
    from .gemma_inference import gemma_available
    # all_records() stats multi-GB model dirs; gemma_available() does a network
    # probe. Both are blocking — keep them off the event loop.
    recs, gemma_ok = await asyncio.gather(
        asyncio.to_thread(all_records),
        asyncio.to_thread(gemma_available),
    )
    return {
        "models": [
            dict(id=r.hf_id, name=r.hf_id.split("/")[-1],
                 ready=r.can_load, cached=r.is_cached,
                 gb=r.disk_gb, note=r.note)
            for r in recs
        ],
        "gemma_remote": gemma_ok,
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
    matches = list(UPLOADS.glob(f"{fid}*"))
    if not matches:
        raise HTTPException(404, "Upload not found")
    return matches[0]


# ── Analyze ────────────────────────────────────────────────────────────────────
@app.post("/api/job/analyze")
async def job_analyze(
    file_id:        str   = Form(...),
    query:          str   = Form("people and vehicles"),
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
    if chunk_duration != 0:
        cmd += ["--chunk-duration", str(chunk_duration)]
    if chunk_overlap != 3:
        cmd += ["--chunk-overlap", str(chunk_overlap)]

    asyncio.create_task(_exec(job, cmd, {"video": out_v, "json": out_j, "report": out_r,
                                          "fast_json": out_f if out_f else ""}))
    return {"job_id": jid, "created_at": job["ts"]}


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

    asyncio.create_task(_exec(job, cmd, {"fast_json": out}))
    return {"job_id": jid, "created_at": job["ts"]}


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
    file_id:    str   = Form(...),
    prompts:    str   = Form("person"),
    threshold:  float = Form(0.15),
    every:      int   = Form(2),
    resolution: int   = Form(1008),
    opacity:    float = Form(0.6),
):
    src = _find_upload(file_id)
    job = _new_job("track")
    jid = job["id"]
    out = str(RESULTS / f"{jid}_tracked.mp4")
    cmd = [PYTHON, "-u", "-m", "visionbrain", "track",
           "--video", str(src),
           "--prompts", *prompts.split(),
           "--output", out,
           "--threshold", str(threshold),
           "--every", str(every),
           "--resolution", str(resolution),
           "--opacity", str(opacity)]
    asyncio.create_task(_exec(job, cmd, {"video": out}))
    return {"job_id": jid, "created_at": job["ts"]}


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


# ── Job query & SSE ────────────────────────────────────────────────────────────
@app.get("/api/job/{jid}")
async def get_job(jid: str):
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404)
    return {k: v for k, v in job.items() if k != "_proc"}


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
    matches = list(UPLOADS.glob(f"{fid}*"))
    if not matches:
        raise HTTPException(404)
    return FileResponse(str(matches[0]))


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
