#!/usr/bin/env sh
# Fetches into ./weights: the generator, the TAESD autoencoder (MIT), a 60 s held-out comma2k19 clip (MIT) with its
# speed / yaw-rate log, the world model and the three steering policies.
set -e
mkdir -p weights
base=https://github.com/azet-llc/realview/releases/download/v0.2.0
for f in realview.pt taesd_encoder.pth taesd_decoder.pth sample_day.mp4 sample_day.npz \
         realview_world.pt policy_bc.pt policy_aug.pt policy_rl.pt; do
  [ -f "weights/$f" ] || curl -fL -o "weights/$f" "$base/$f"
done
echo "weights ready"
