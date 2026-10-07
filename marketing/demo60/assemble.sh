#!/bin/bash
# demo60 assembly — script_final.md → 60.0 s 1920×1080@30 mp4
# scene lengths L_i = d_i + 0.5 tail (scene 6 exactly 7.0); final xfade chain 0.5
# all overlay pngs MUST use -loop 1 -t <len>: a single-frame stream repeats its t=0
# frame under overlay, and alpha fades leave that frame fully transparent
set -euo pipefail
cd "$(dirname "$0")"
V=../demo/video; S=../shots; F=shots/f_; O=shots
ENC="-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p"
SCALE720="scale=1920:1080:flags=lanczos,fps=30,settb=AVTB,format=yuv420p"
ROT="scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,fps=30,settb=AVTB,format=yuv420p"
IMG="-loop 1 -framerate 30"

# ---- scene 1 · hook 9.0 (+0.5) : rotterdam clean 4.5 → marina masks 5.5, o1 overlay in at 4.0
# (raw excerpt — the analyzed mp4 flashes boxes on every 5th frame, unusable as clean establishing)
ffmpeg -y -v error -ss 6.0 -t 4.5 -i ../demo/rotterdam_excerpt_20s.mp4 \
  -ss 0 -t 5.5 -i $V/mission_marina_boats_everyframe.mp4 \
  -loop 1 -t 9.5 -i $O/f_o1_hook.png -filter_complex "
  [0:v]$ROT[a];[1:v]fps=30,settb=AVTB,scale=1920:1080:flags=lanczos,format=yuv420p[b];
  [a][b]xfade=transition=fade:duration=0.5:offset=4.0[x];
  [2:v]format=rgba,fade=t=in:st=4.0:d=0.5:alpha=1[o];
  [x][o]overlay=0:0:format=auto,fade=t=in:st=0:d=0.5,format=yuv420p" \
  $ENC -t 9.5 cuts/s1.mp4

# ---- scene 2 · pipeline 10.0 (+0.5) : dim → lit
ffmpeg -y -v error $IMG -t 6.1 -i ${F}f2_dim.png $IMG -t 5.0 -i ${F}f2_pipeline.png -filter_complex "
  [0:v]fps=30,settb=AVTB,format=yuv420p[a];[1:v]fps=30,settb=AVTB,format=yuv420p[b];
  [a][b]xfade=transition=fade:duration=0.6:offset=5.5" \
  $ENC -t 10.5 cuts/s2.mp4

# ---- scene 3 · recorded 11.0 (+0.5) : containers 7.8 → results+report still 4.1
ffmpeg -y -v error -ss 0 -t 7.8 -i $V/mission_containers_truck_everyframe.mp4 \
  $IMG -t 4.1 -i $S/analyze_results_1920.png \
  -loop 1 -t 7.8 -i $O/f_o3_a.png \
  -loop 1 -t 4.1 -i $O/f_o3_b.png -filter_complex "
  [0:v]$SCALE720[c0];[2:v]format=rgba,fade=t=in:st=0.2:d=0.3:alpha=1[o1];
  [c0][o1]overlay=0:0:format=auto[s1];
  [1:v]fps=30,settb=AVTB,format=yuv420p[r];[3:v]format=rgba,fade=t=in:st=0.2:d=0.3:alpha=1[o2];
  [r][o2]overlay=0:0:format=auto[s2];
  [s1][s2]xfade=transition=fade:duration=0.4:offset=7.4,format=yuv420p" \
  $ENC -t 11.5 cuts/s3.mp4

# ---- scene 4 · live 12.0 (+0.5) : yard masks 8.3 → harbor tracking 4.6
ffmpeg -y -v error -ss 0 -t 8.3 -i $V/live_masks_container_truck.mp4 \
  -ss 2 -t 4.6 -i $V/live_harbor_tracking.mp4 \
  -loop 1 -t 8.3 -i $O/f_o4_a.png \
  -loop 1 -t 4.6 -i $O/f_o4_b.png -filter_complex "
  [0:v]$SCALE720[c0];[2:v]format=rgba,fade=t=in:st=0.2:d=0.3:alpha=1[o1];
  [c0][o1]overlay=0:0:format=auto[s1];
  [1:v]$SCALE720[c1];[3:v]format=rgba,fade=t=in:st=0.2:d=0.3:alpha=1[o2];
  [c1][o2]overlay=0:0:format=auto[s2];
  [s1][s2]xfade=transition=fade:duration=0.4:offset=7.9,format=yuv420p" \
  $ENC -t 12.5 cuts/s4.mp4

# ---- scene 5 · ask 11.0 (+0.5) : live terminal 6.3 (bubbles stage) → results/report still 5.6
ffmpeg -y -v error $IMG -t 6.3 -i $S/live_terminal_1920.png \
  $IMG -t 5.6 -i $S/analyze_results_1920.png \
  -loop 1 -t 6.3 -i $O/f_o5_q.png \
  -loop 1 -t 6.3 -i $O/f_o5_a.png \
  -loop 1 -t 5.6 -i $O/f_o5_b.png -filter_complex "
  [0:v]fps=30,settb=AVTB,format=yuv420p[t];[2:v]format=rgba,fade=t=in:st=0.8:d=0.4:alpha=1[q];
  [t][q]overlay=0:0:format=auto[g1];
  [3:v]format=rgba,fade=t=in:st=2.8:d=0.4:alpha=1[a2];
  [g1][a2]overlay=0:0:format=auto[s1];
  [1:v]fps=30,settb=AVTB,format=yuv420p[p];[4:v]format=rgba,fade=t=in:st=0.3:d=0.3:alpha=1[b1];
  [p][b1]overlay=0:0:format=auto[s2];
  [s1][s2]xfade=transition=fade:duration=0.4:offset=5.9,format=yuv420p" \
  $ENC -t 11.5 cuts/s5.mp4

# ---- scene 6 · close 7.0 : rotterdam clean 2.6 → live quay 2.4 → end card 2.8, fade out
ffmpeg -y -v error -ss 11.0 -t 2.6 -i ../demo/rotterdam_excerpt_20s.mp4 \
  $IMG -t 2.4 -i $S/live_quay_1920.png \
  $IMG -t 2.8 -i ${F}f6_end.png -filter_complex "
  [0:v]$ROT[a];[1:v]fps=30,settb=AVTB,format=yuv420p[q];[2:v]fps=30,settb=AVTB,format=yuv420p[e];
  [a][q]xfade=transition=fade:duration=0.4:offset=2.2[x1];
  [x1][e]xfade=transition=fade:duration=0.4:offset=4.2,fade=t=out:st=6.4:d=0.6,format=yuv420p" \
  $ENC cuts/s6.mp4

# ---- narration: lead 0.6 s + VO, pad to scene length; s2/s6 tempo-fitted
ffmpeg -y -v error -i audio/s1.wav -filter_complex "[0]adelay=600|600,apad[a];[a]atrim=0:9.5" -ar 48000 -ac 2 audio/n1.wav
ffmpeg -y -v error -i audio/s2.wav -filter_complex "[0]atempo=1.08,adelay=600|600,apad[a];[a]atrim=0:10.5" -ar 48000 -ac 2 audio/n2.wav
ffmpeg -y -v error -i audio/s3.wav -filter_complex "[0]adelay=600|600,apad[a];[a]atrim=0:11.5" -ar 48000 -ac 2 audio/n3.wav
ffmpeg -y -v error -i audio/s4.wav -filter_complex "[0]adelay=600|600,apad[a];[a]atrim=0:12.5" -ar 48000 -ac 2 audio/n4.wav
ffmpeg -y -v error -i audio/s5.wav -filter_complex "[0]adelay=600|600,apad[a];[a]atrim=0:11.5" -ar 48000 -ac 2 audio/n5.wav
ffmpeg -y -v error -i audio/s6.wav -filter_complex "[0]atempo=1.10,adelay=600|600,apad[a];[a]atrim=0:7.0" -ar 48000 -ac 2 audio/n6.wav
ffmpeg -y -v error -i audio/n1.wav -i audio/n2.wav -i audio/n3.wav -i audio/n4.wav -i audio/n5.wav -i audio/n6.wav -filter_complex "
  [0][1]acrossfade=d=0.5[x1];[x1][2]acrossfade=d=0.5[x2];[x2][3]acrossfade=d=0.5[x3];
  [x3][4]acrossfade=d=0.5[x4];[x4][5]acrossfade=d=0.5,loudnorm=I=-16:TP=-1.5:LRA=11" \
  -ar 48000 -ac 2 audio/narration.wav

# ---- final: xfade chain at scene boundaries (9,19,30,42,53) + mux
ffmpeg -y -v error -i cuts/s1.mp4 -i cuts/s2.mp4 -i cuts/s3.mp4 -i cuts/s4.mp4 -i cuts/s5.mp4 -i cuts/s6.mp4 -i audio/narration.wav -filter_complex "
  [0:v]settb=AVTB[v0];[1:v]settb=AVTB[v1];[2:v]settb=AVTB[v2];[3:v]settb=AVTB[v3];[4:v]settb=AVTB[v4];[5:v]settb=AVTB[v5];
  [v0][v1]xfade=transition=fade:duration=0.5:offset=9.0[x1];
  [x1][v2]xfade=transition=fade:duration=0.5:offset=19.0[x2];
  [x2][v3]xfade=transition=fade:duration=0.5:offset=30.0[x3];
  [x3][v4]xfade=transition=fade:duration=0.5:offset=42.0[x4];
  [x4][v5]xfade=transition=fade:duration=0.5:offset=53.0,format=yuv420p[v]" \
  -map "[v]" -map 6:a -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart -t 60.0 visionbrain_demo_60s.mp4

echo "== final duration:"; ffprobe -v error -show_entries format=duration -of csv=p=0 visionbrain_demo_60s.mp4
