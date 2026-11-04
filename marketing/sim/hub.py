"""Field hub simulator — replays real VisionBrain track JSONs over the
observer-relay wire protocol the Ground Control live tab speaks.

Single task per connection: interleaves frame streaming with control-message
handling, so there are never concurrent sends on the socket.

Usage: python hub.py [env ...]     (default: cycle all environments)

Wire: binary frames  >III (frame_id, ts_ms32, jpeg_len) + JPEG + >I telem_len + JSON
JSON msgs: detections {items:[{box:[x1,y1,x2,y2] norm, label, score, track_id,
color_id, polygon?: [[x,y],...] norm mask outline}]}
"""
from __future__ import annotations

import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import websockets

SIM = Path("/tmp/vb_sim")
FRAMES_DIR = SIM / "frames"
TRACKS_DIR = SIM / "tracks"
FPS = 5
CYCLE_SECONDS = 40  # per environment when cycling

ENVS = ["highway_traffic", "quay_cranes", "trucks_terminal", "shipyard_basin", "vessel_outfitting"]

SITES = {
    "highway_traffic":  {"site": "corridor A3 · overpass 7",  "alt": 92.0,  "hdg": 274},
    "quay_cranes":      {"site": "port east · berth 12",      "alt": 118.0, "hdg": 131},
    "trucks_terminal":  {"site": "intermodal yard · gate 3",  "alt": 84.0,  "hdg": 208},
    "shipyard_basin":   {"site": "shipyard · basin 2",        "alt": 105.0, "hdg": 356},
    "vessel_outfitting": {"site": "outfitting pier · stand 4", "alt": 76.0, "hdg": 99},
}


def site_of(name: str) -> dict:
    """Site record for an env — custom envs get a derived label, not a KeyError."""
    return SITES.get(name, {"site": name.replace("_", " "), "alt": 90.0, "hdg": 0})


def load_env(name: str) -> dict:
    frames = sorted((FRAMES_DIR / name).glob("*.jpg"))
    tracks = {}
    tj = TRACKS_DIR / f"{name}.json"
    if tj.exists():
        data = json.loads(tj.read_text())
        res = data.get("resolution", "512x512")
        w, _, h = res.partition("x")
        w, h = float(w), float(h)
        pf = data.get("frames", [])
        n = len(frames)
        for i in range(n):
            j = min(len(pf) - 1, round(i * max(1, len(pf) - 1) / max(1, n - 1)))
            dets = []
            for d in pf[j].get("detections", []):
                x1, y1, x2, y2 = d["bbox_xyxy"]
                det = {
                    "box": [x1 / w, y1 / h, x2 / w, y2 / h],
                    "label": d["label"],
                    "score": d["score"],
                    "track_id": d["track_id"],
                    "color_id": d["track_id"],
                }
                if d.get("polygon"):
                    poly = [[float(p[0]), float(p[1])] for p in d["polygon"] if len(p) >= 2]
                    # Track JSONs are pixel-space; normalized polys pass through.
                    if poly and max(max(x, y) for x, y in poly) > 1.5:
                        poly = [[px / w, py / h] for px, py in poly]
                    det["polygon"] = poly
                dets.append(det)
            tracks[i] = dets
    return {"frames": frames, "tracks": tracks, "name": name}


class Hub:
    def __init__(self, env_order: list[str]):
        self.envs = [load_env(n) for n in env_order]
        self.frame_id = 0

    def current_env(self) -> dict:
        return self.envs[int(time.monotonic() / CYCLE_SECONDS) % len(self.envs)]

    def telemetry(self, env: dict, i: int) -> bytes:
        site = site_of(env["name"])
        return json.dumps({
            "site": site["site"], "env": env["name"],
            "battery": round(max(18, 96 - (time.monotonic() % 3600) / 90), 1),
            "altitude_m": site["alt"], "heading_deg": (site["hdg"] + i * 0.1) % 360,
            "ground_speed_ms": round(7.5 + 1.2 * (i % 13) / 12, 1),
            "gps": {"lat": 51.9244 + i * 1e-6, "lon": 4.4777 + i * 1e-6},
            "frame_id": self.frame_id, "fps": FPS,
        }).encode()

    def answer_for(self, env: dict) -> str:
        counts = sorted(len(v) for v in env["tracks"].values())
        n = counts[len(counts) // 2] if counts else 0
        return ("holding pattern over the quay. "
                f"{n} tracked objects in frame, movement nominal. "
                "no perimeter events in the last 60 s.")

    async def handler(self, ws):
        env = None
        idx = 0
        next_frame_at = time.monotonic()
        while True:
            recv_task = asyncio.ensure_future(ws.recv())
            delay = max(0.0, next_frame_at - time.monotonic())
            done, _pending = await asyncio.wait({recv_task}, timeout=delay)
            if recv_task in done:
                try:
                    raw = recv_task.result()
                except websockets.ConnectionClosed:
                    return
                if raw is None:
                    return
                try:
                    msg = json.loads(raw)
                except Exception:
                    msg = {}
                t = msg.get("type")
                if t == "hello":
                    await ws.send(json.dumps({"type": "status", "note": "hub ready · 5 fps relay"}))
                elif t == "set_engine":
                    on = [k for k in ("sam", "falcon", "lfm") if msg.get(k)]
                    await ws.send(json.dumps({"type": "status", "note": "engines · " + ", ".join(on or ["none"])}))
                elif t == "set_vlm":
                    await ws.send(json.dumps({"type": "status", "note": f"vlm · {msg.get('model')}"}))
                elif t == "ask":
                    await ws.send(json.dumps({"type": "ask_ack", "model": "gemma4:e2b"}))
                    await asyncio.sleep(1.2)
                    await ws.send(json.dumps({"type": "answer", "answer": self.answer_for(self.current_env())}))
                elif t == "report":
                    await asyncio.sleep(0.8)
                    await ws.send(json.dumps({
                        "type": "report_result",
                        "text": ("field report — " + site_of(self.current_env()["name"])["site"] + ", 14:32\n"
                                 "operation nominal across the surveyed cell. tracked objects held "
                                 "their lanes; no perimeter breaches. recommend continuing the "
                                 "sweep on the current heading.")}))
                continue

            recv_task.cancel()
            cur = self.current_env()
            if cur is not env:
                env = cur
                idx = 0
                await ws.send(json.dumps({"type": "status", "note": f"relay · {site_of(env['name'])['site']}"}))
            n = len(env["frames"])
            i = idx % n
            dets = env["tracks"].get(i, [])
            await ws.send(json.dumps({"type": "detections", "items": dets}))
            jpeg = env["frames"][i].read_bytes()
            telem = self.telemetry(env, i)
            hdr = struct.pack(">III", self.frame_id % 2**32, int(time.time() * 1000) % 2**32, len(jpeg))
            await ws.send(hdr + jpeg + struct.pack(">I", len(telem)) + telem)
            self.frame_id += 1
            idx += 1
            next_frame_at = time.monotonic() + 1.0 / FPS


async def main():
    order = sys.argv[1:] or ENVS
    hub = Hub(order)
    async with websockets.serve(hub.handler, "127.0.0.1", 8765, max_size=8 * 2**20):
        print("hub on ws://127.0.0.1:8765 — envs:", ", ".join(order), flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
