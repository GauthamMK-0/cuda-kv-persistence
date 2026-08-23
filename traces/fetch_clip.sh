#!/usr/bin/env bash
# Fetch a small real-video sequence for motion-trace generation.
# Priority: 1) traces/video/input.mp4 (user-supplied)  2) cached extraction
#           3) DAVIS-2017 trainval 480p zip -> extract ONLY blackswan -> rm zip.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VIDEO_DIR="$HERE/video"
SEQ_DIR="$VIDEO_DIR/DAVIS/JPEGImages/480p/blackswan"
ZIP_URL="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"

mkdir -p "$VIDEO_DIR"

if [ -f "$VIDEO_DIR/input.mp4" ]; then
    echo "clip source: $VIDEO_DIR/input.mp4 (user-supplied)"
    exit 0
fi

if [ -d "$SEQ_DIR" ] && [ "$(ls -1 "$SEQ_DIR" | wc -l)" -gt 10 ]; then
    echo "clip source: $SEQ_DIR ($(ls -1 "$SEQ_DIR" | wc -l) frames, cached)"
    exit 0
fi

echo "downloading $ZIP_URL ..."
ZIP="$VIDEO_DIR/davis480p.zip"
# -C -: resume a partially downloaded archive if one exists
curl -fL --retry 2 -sS -C - -o "$ZIP" "$ZIP_URL"
echo "downloaded $(du -h "$ZIP" | cut -f1); extracting only blackswan/"
unzip -q -o "$ZIP" "DAVIS/JPEGImages/480p/blackswan/*" -d "$VIDEO_DIR"
rm -f "$ZIP"   # storage hygiene: archive is regenerable
echo "extracted $SEQ_DIR ($(ls -1 "$SEQ_DIR" | wc -l) frames)"
