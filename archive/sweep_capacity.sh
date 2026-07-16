#!/bin/bash
# Path 1: CVAE capacity sweep — latent_dim × hidden_dim
# Baseline: latent=32, hidden=64, +30K params, MissT=0.618
# All fp32, batch=32, num_workers=2

PYTHON=python3
DATA="casp_dataset/mosei.pkl"
EPOCHS=30
BS=32
WORKERS=2

declare -A CONFIGS
CONFIGS["A_lat64_hid128"]="64 128"
CONFIGS["B_lat64_hid256"]="64 256"
CONFIGS["C_lat128_hid128"]="128 128"
CONFIGS["D_lat128_hid256"]="128 256"

for cfg in A_lat64_hid128 B_lat64_hid256 C_lat128_hid128 D_lat128_hid256; do
    read LAT HID <<< "${CONFIGS[$cfg]}"
    echo ""
    echo "============================================================"
    echo "Path 1 Config $cfg: latent=$LAT, hidden=$HID"
    echo "============================================================"

    OUT="/tmp/sweep_${cfg}.txt"
    $PYTHON train_cvae.py \
      --mode cvae \
      --dataset mosei \
      --datapath $DATA \
      --num_epochs $EPOCHS \
      --batch_size $BS \
      --num_workers $WORKERS \
      --kl_weight 0.1 \
      --recon_weight 1.0 \
      --mc_samples 5 \
      --cvae_latent $LAT \
      --cvae_hidden $HID \
      --name checkpoints/mosei_cvae_${cfg}.pt \
      --log_interval 500 2>&1 | tee "$OUT"

    echo "Recording results..."
    $PYTHON record_results.py "$OUT" "mosei_cvae_${cfg}" \
      --config "mode=cvae,batch_size=32,cvae_latent=$LAT,cvae_hidden=$HID"
done

echo ""
echo "All capacity sweep experiments completed."
echo "Run manual evaluation for each checkpoint to get final test numbers."
