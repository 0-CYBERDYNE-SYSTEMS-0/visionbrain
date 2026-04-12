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

app = FastAPI(title="VisionBrain Ground Control", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    job = dict(id=jid, kind=kind, status="pending",
                ts=time.time(), output=[], results={}, error=None)
    _jobs[jid] = job
    return job


async def _exec(job: dict, cmd: list[str], outputs: dict[str, str]) -> None:
    """Run CLI command async; stream stdout into job.output[]."""
    job["status"] = "running"
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    async for raw in proc.stdout:
        job["output"].append(raw.decode("utf-8", errors="replace").rstrip())
    await proc.wait()
    if proc.returncode == 0:
        job["status"] = "done"
        for k, p in outputs.items():
            if p and Path(p).exists():
                job["results"][k] = p
    else:
        job["status"] = "error"
        job["error"] = f"exit {proc.returncode}"


# ── Status ─────────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    from .loader import all_records
    from .remote_gemma_inference import gemma_available
    recs = all_records()
    return {
        "models": [
            dict(id=r.hf_id, name=r.hf_id.split("/")[-1],
                 ready=r.can_load, cached=r.is_cached,
                 gb=r.disk_gb, note=r.note)
            for r in recs
        ],
        "gemma_remote": gemma_available(),
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
    file_id:       str   = Form(...),
    query:         str   = Form("cattle in the pasture"),
    prompts:       str   = Form("cow cattle animal"),
    threshold:     float = Form(0.05),
    resolution:    int   = Form(512),
    every:         int   = Form(5),
    backbone_every:int   = Form(1),
    opacity:       float = Form(0.6),
    report:        bool  = Form(True),
    report_type:   str   = Form("field"),
    falcon_refine: bool  = Form(False),
    falcon_frames: int   = Form(6),
    max_tokens:    int   = Form(512),
):
    src = _find_upload(file_id)
    job = _new_job("analyze")
    jid = job["id"]
    out_v = str(RESULTS / f"{jid}_analyzed.mp4")
    out_j = str(RESULTS / f"{jid}_detections.json")
    out_r = str(RESULTS / f"{jid}_report.txt")

    cmd = [PYTHON, "-m", "visionbrain", "analyze",
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
    if falcon_refine:
        cmd += ["--falcon-refine", "--falcon-frames", str(falcon_frames)]

    asyncio.create_task(_exec(job, cmd, {"video": out_v, "json": out_j, "report": out_r}))
    return {"job_id": jid}


# ── Detect ─────────────────────────────────────────────────────────────────────
@app.post("/api/job/detect")
async def job_detect(
    file_id:    str = Form(...),
    query:      str = Form("cattle"),
    max_tokens: int = Form(200),
):
    src = _find_upload(file_id)
    job = _new_job("detect")
    jid = job["id"]
    out = str(RESULTS / f"{jid}_detected.jpg")
    cmd = [PYTHON, "-m", "visionbrain", "detect",
           "--image", str(src), "--query", query,
           "--max-tokens", str(max_tokens), "--output", out]
    asyncio.create_task(_exec(job, cmd, {"image": out}))
    return {"job_id": jid}


# ── Segment ────────────────────────────────────────────────────────────────────
@app.post("/api/job/segment")
async def job_segment(
    file_id:    str = Form(...),
    query:      str = Form("cattle"),
    max_tokens: int = Form(2048),
):
    src = _find_upload(file_id)
    job = _new_job("segment")
    jid = job["id"]
    out = str(RESULTS / f"{jid}_segmented.jpg")
    cmd = [PYTHON, "-m", "visionbrain", "segment",
           "--image", str(src), "--query", query,
           "--max-tokens", str(max_tokens), "--output", out]
    asyncio.create_task(_exec(job, cmd, {"image": out}))
    return {"job_id": jid}


# ── OCR ────────────────────────────────────────────────────────────────────────
@app.post("/api/job/ocr")
async def job_ocr(
    file_id:  str = Form(...),
    question: str = Form("read all text in the image"),
):
    src = _find_upload(file_id)
    job = _new_job("ocr")
    jid = job["id"]
    cmd = [PYTHON, "-m", "visionbrain", "ocr",
           "--image", str(src), "--question", question]
    asyncio.create_task(_exec(job, cmd, {}))
    return {"job_id": jid}


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
    cmd = [PYTHON, "-m", "visionbrain", "track",
           "--video", str(src),
           "--prompts", *prompts.split(),
           "--output", out,
           "--threshold", str(threshold),
           "--every", str(every),
           "--resolution", str(resolution),
           "--opacity", str(opacity)]
    asyncio.create_task(_exec(job, cmd, {"video": out}))
    return {"job_id": jid}


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
    cmd = [PYTHON, "-m", "visionbrain", "sam3",
           "--image", str(src),
           "--prompts", *prompts.split(),
           "--task", task,
           "--threshold", str(threshold),
           "--resolution", str(resolution),
           "--output", out]
    asyncio.create_task(_exec(job, cmd, {"image": out}))
    return {"job_id": jid}


# ── Job query & SSE ────────────────────────────────────────────────────────────
@app.get("/api/job/{jid}")
async def get_job(jid: str):
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404)
    return {k: v for k, v in job.items() if k != "_proc"}


@app.get("/api/job/{jid}/stream")
async def stream_job(jid: str, request: Request):
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404)
    sent = 0

    async def gen() -> AsyncGenerator[str, None]:
        nonlocal sent
        while True:
            if await request.is_disconnected():
                break
            lines = job["output"]
            if len(lines) > sent:
                for ln in lines[sent:]:
                    yield f"data: {json.dumps({'type':'log','msg':ln})}\n\n"
                sent = len(lines)
            if job["status"] in ("done", "error"):
                yield f"data: {json.dumps({'type':'done','status':job['status'],'results':job['results'],'error':job['error']})}\n\n"
                break
            await asyncio.sleep(0.08)

    return StreamingResponse(gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── File serving ───────────────────────────────────────────────────────────────
@app.get("/api/job/{jid}/file/{kind}")
async def serve_file(jid: str, kind: str):
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404)
    path = job["results"].get(kind)
    if not path or not Path(path).exists():
        raise HTTPException(404, f"No result '{kind}'")
    return FileResponse(path)


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
        return HTMLResponse(p.read_text())
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

    uvicorn.run(app, host=host, port=port, log_level="warning")
