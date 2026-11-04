#!/usr/bin/env bash
# VisionBrain launcher — starts Ground Control, nothing else.
#
#   ./launch.sh    start the web app in the foreground (takes no arguments)
#
# Run it from YOUR Terminal (not an automation shell): macOS attributes the
# camera permission to Terminal, which is what lets the live engine open a
# webcam. If macOS ever asks for camera access, click Allow (one time).
#
# Scope (this is all of it):
#   :7860  Ground Control web UI (foreground — this window stays open;
#          Ctrl-C stops it)
#
# No camera bridge, no field-hub simulator, no process management: nothing
# is ever killed by port. If :7860 is already occupied you get uvicorn's
# ordinary bind error and the existing process keeps it. The demo pieces
# (internet-camera bridge, field-hub simulator) are started by hand when
# wanted — see the usage message for the manual commands.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python

if [ "$#" -gt 0 ]; then
  echo "usage: ./launch.sh                start Ground Control (no arguments)" >&2
  echo "" >&2
  echo "This launcher only starts the web app at http://127.0.0.1:7860." >&2
  echo "The old bundled modes (--hub, stop, auto camera bridge) are gone;" >&2
  echo "run the demo pieces manually if you need them:" >&2
  echo "" >&2
  echo "  field-hub simulator (ws://127.0.0.1:8765):" >&2
  echo "    $PY marketing/sim/hub.py" >&2
  echo "" >&2
  echo "  live internet-camera bridge (HLS at http://127.0.0.1:8554/live.m3u8):" >&2
  echo "    mkdir -p /tmp/nyc_src /tmp/nyc_hls" >&2
  echo "    bash tools/nyc_cam_fetch.sh &" >&2
  echo "    $PY tools/nyc_cam_serve.py &" >&2
  echo "    ffmpeg -hide_banner -loglevel error \\" >&2
  echo "      -re -f image2 -stream_loop -1 -framerate 4 -i /tmp/nyc_src/frame.jpg \\" >&2
  echo "      -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g 8 \\" >&2
  echo "      -f hls -hls_time 1 -hls_list_size 3 \\" >&2
  echo "      -hls_segment_filename /tmp/nyc_hls/seg_%04d.ts /tmp/nyc_hls/live.m3u8 &" >&2
  exit 2
fi

# ── Ground Control in the foreground (Ctrl-C to stop).
echo ""
echo "  VisionBrain Ground Control → http://127.0.0.1:7860"
echo "  (Ctrl-C stops the foreground server; if :7860 is already in use you'll"
echo "   see uvicorn's ordinary bind error — nothing is killed automatically)"
echo ""
exec "$PY" -m uvicorn visionbrain.web_app:app --host 127.0.0.1 --port 7860
