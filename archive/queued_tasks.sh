#!/bin/bash
# Queued training tasks — run sequentially after cycle consistency finishes
# fp32 default (FP16 caused NaN in CVAE exp ops; model too small to benefit)
# batch=32 + num_workers=4 for max GPU utilization
# VRAM est: fp32 batch=32 ~2-3GB/8GB

PYTHON=python
MOSEI="casp_dataset/mosei.pkl"
LOG_INT=200
EPOCHS=30
WORKERS=4
BS=32

echo "============================================================"
echo "Task 1: Random-z CVAE (batch=32, lr=2e-3, fp32)"
echo "============================================================"
T1_OUT="/tmp/task1_mc_v2_output.txt"
$PYTHON train_cvae.py \
  --mode cvae \
  --dataset mosei \
  --datapath $MOSEI \
  --num_epochs $EPOCHS \
  --batch_size $BS \
  --num_workers $WORKERS \
  --lr 2e-3 \
  --kl_weight 0.1 \
  --recon_weight 1.0 \
  --mc_samples 5 \
  --name checkpoints/mosei_cvae_mc_v2.pt \
  --log_interval $LOG_INT 2>&1 | tee "$T1_OUT"
echo "Training complete. Recording results..."
$PYTHON record_results.py "$T1_OUT" "mosei_cvae_mc_v2" \
  --config "mode=cvae,batch_size=32,lr=2e-3,kl_weight=0.1,recon_weight=1.0,mc_samples=5"

echo ""
echo "============================================================"
echo "Task 2: Progressive Dropout (batch=32, lr=1e-3, fp32)"
echo "============================================================"
T2_OUT="/tmp/task2_progdrop_output.txt"
$PYTHON train_cvae.py \
  --mode cvae \
  --dataset mosei \
  --datapath $MOSEI \
  --num_epochs $EPOCHS \
  --batch_size $BS \
  --num_workers $WORKERS \
  --lr 1e-3 \
  --kl_weight 0.1 \
  --recon_weight 1.0 \
  --mc_samples 5 \
  --dropout_schedule progressive \
  --dropout_start 0.05 \
  --dropout_end 0.30 \
  --name checkpoints/mosei_cvae_progdrop.pt \
  --log_interval $LOG_INT 2>&1 | tee "$T2_OUT"
echo "Training complete. Recording results..."
$PYTHON record_results.py "$T2_OUT" "mosei_cvae_progdrop" \
  --config "mode=cvae,batch_size=32,lr=1e-3,kl_weight=0.1,dropout_schedule=progressive,dropout_start=0.05,dropout_end=0.30"

echo ""
echo "All queued tasks completed."
