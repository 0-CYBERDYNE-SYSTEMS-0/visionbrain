#!/usr/bin/env bash
# Refresh-loop for the live NYC street camera: pulls a fresh JPEG every
# second into /tmp/nyc_src/frame.jpg (atomic rename) for ffmpeg to read.
set -u
CAM_ID="${NYC_CAM_ID:-8a6bc417-4877-4ebe-8052-88c1b261baf1}"  # Central Park West @ 86 St
URL="https://webcams.nyctmc.org/api/cameras/$CAM_ID/image"
mkdir -p /tmp/nyc_src
while true; do
  curl -s -m 8 -o /tmp/nyc_src/.new.jpg "$URL" \
    && mv /tmp/nyc_src/.new.jpg /tmp/nyc_src/frame.jpg
  sleep 1
done
