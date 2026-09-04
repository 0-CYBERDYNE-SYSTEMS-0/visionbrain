# Deploying VisionBrain

Ops guide for running the Ground Control server on a box that analysts reach
from their browsers. Same app, two shapes: a Mac on the office LAN, or a
hosted Mac (MacStadium-class). MLX requires Apple Silicon — for Linux/GPU
servers you need a CUDA inference path, not this document.

## Requirements

- Apple Silicon Mac (M-series), macOS 12+
- Python 3.12+ (repo `.venv` uses 3.14), ffmpeg on PATH
- Model weights (local Hugging Face cache; **no network needed at runtime**):

```bash
huggingface-cli download mlx-community/sam3.1-bf16        # ~3.5 GB — tracking
huggingface-cli download tiiuae/Falcon-Perception          # 0.6B — grounding/OCR
huggingface-cli download tiiuae/Falcon-OCR                 # 0.3B — registry only (vLLM upstream)
ollama pull gemma4:e2b                                     # 7.2 GB — reasoning (or configure a custom VLM endpoint in the UI)
```

## Install & run

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m visionbrain ui --host 0.0.0.0 --port 7860 --no-browser
# or: .venv/bin/uvicorn visionbrain.web_app:app --host 0.0.0.0 --port 7860
```

Analysts open `http://<server-ip>:7860`. The browser is a display only — all
models load in server processes on this box.

## Auth (shared token)

Off by default. Set `VB_TOKEN` to require a shared token on every `/api/*`
route except `/api/healthz`:

```bash
export VB_TOKEN="pick-a-long-random-string"
```

Clients send it as the `X-Auth-Token` header or `?token=` query parameter:

```bash
curl -H "X-Auth-Token: pick-a-long-random-string" http://host:7860/api/status
```

**Known gap:** the live-engine WebSocket (`/api/live/ws`) is not
token-enforced yet — protect it at the network boundary (LAN firewall rules
or a reverse-proxy ACL on that path) until it is.

## Concurrency

`VB_MAX_JOBS` (1–4, default 1) caps concurrent heavy jobs (`analyze`,
`fastscan`, `track`, `agent`) through a FIFO queue; queued jobs report their
position in launch responses and SSE heartbeats. Light image jobs
(`detect`, `segment`, `sam3`, `ocr`) bypass the queue.

Memory is the constraint — SAM 3.1 (~3 GB), Falcon (~1.2 GB) and a reasoning
model are resident in unified memory:

| Unified memory | VB_MAX_JOBS |
|---|---|
| 16 GB | 1 |
| 32 GB | 1–2 |
| 48–64 GB+ | 2–4 |

## Keep it running (macOS launchd)

`~/Library/LaunchAgents/visionbrain.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>ai.visionbrain.server</string>
  <key>ProgramArguments</key><array>
    <string>/path/to/VisionBrain/.venv/bin/python</string>
    <string>-m</string><string>visionbrain</string><string>ui</string>
    <string>--host</string><string>0.0.0.0</string><string>--port</string><string>7860</string>
    <string>--no-browser</string>
  </array>
  <key>WorkingDirectory</key><string>/path/to/VisionBrain</string>
  <key>EnvironmentVariables</key><dict>
    <key>VB_TOKEN</key><string>pick-a-long-random-string</string>
    <key>VB_MAX_JOBS</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>
```

`launchctl load ~/Library/LaunchAgents/visionbrain.plist`

## TLS (analysts off the LAN)

Plain HTTP is fine on a closed LAN. For VPN/remote access, terminate TLS at a
reverse proxy — never expose 7860 raw to the internet:

**Caddy** (automatic certs):

```
visionbrain.example.com {
    reverse_proxy 127.0.0.1:7860
}
```

**nginx** (WebSocket upgrade included — needed by the live tab):

```nginx
location / {
    proxy_pass http://127.0.0.1:7860;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;      # WebSocket support
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

Optionally restrict `/api/live/ws` at the proxy if you don't use the live tab
remotely (see auth gap above).

## Data locations

| Path | Contents |
|---|---|
| `/tmp/visionbrain_ui/uploads` | uploaded media (cleared on reboot — move `WORK_DIR` off `/tmp` for persistence) |
| `/tmp/visionbrain_ui/results` | job outputs (MP4/JSON/reports) |
| `/tmp/visionbrain_ui/clips` | smart-capture clips (50 newest kept, oldest pruned) |
| `~/.visionbrain/settings.json` | custom VLM backend config (0600; api key never returned by the API) |

## Sellable-base checklist

- [ ] `VB_TOKEN` set, token distributed to analysts
- [ ] `VB_MAX_JOBS` sized to memory (table above)
- [ ] launchd/KeepAlive loaded, survives reboot
- [ ] TLS proxy in front for any non-LAN access
- [ ] Weights pre-downloaded; `python -m visionbrain status` shows all READY
- [ ] Licenses reviewed for your customer (SAM: custom permissive; Falcon and
      LFM: check TII / Liquid AI terms for the deployment size)
