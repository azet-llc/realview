#!/usr/bin/env sh
# Fetches the released generator, the TAESD autoencoder (MIT) and a 60 s held-out comma2k19 clip (MIT) into ./weights
set -e
mkdir -p weights
base=https://github.com/azet-llc/realview/releases/download/v0.1.0
for f in realview.pt taesd_encoder.pth taesd_decoder.pth sample_day.mp4; do
  [ -f "weights/$f" ] || curl -fL -o "weights/$f" "$base/$f"
done
echo "weights ready"
