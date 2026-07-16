#!/bin/bash
# Path 1 fixes: three experiments to isolate effects
# All fp32, batch=32, num_workers=2, 30 epochs

cd /home/ly/stu_work/projects/missing-modality-msa
PYTHON=/usr/bin/python3
DATA="casp_dataset/mosei.pkl"
EPOCHS=30; BS=32; WORKERS=2

echo "============================================================"
echo "Fix 1: T-weighted dropout only (baseline CVAE + td=0.3)"
echo "============================================================"
$PYTHON train_cvae.py --mode cvae --dataset mosei --datapath $DATA \
  --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS \
  --kl_weight 0.1 --recon_weight 1.0 \
  --text_dropout 0.3 \
  --name checkpoints/mosei_cvae_fix1_tdrop.pt \
  --log_interval 500 2>&1 | tee /tmp/fix1_tdrop.txt

echo ""
echo "============================================================"
echo "Fix 2: Narrow CVAE only (latent=24, hidden=48, no tdrop)"
echo "============================================================"
$PYTHON train_cvae.py --mode cvae --dataset mosei --datapath $DATA \
  --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS \
  --kl_weight 0.1 --recon_weight 1.0 \
  --cvae_latent 24 --cvae_hidden 48 \
  --name checkpoints/mosei_cvae_fix2_narrow.pt \
  --log_interval 500 2>&1 | tee /tmp/fix2_narrow.txt

echo ""
echo "============================================================"
echo "Fix 3: Combined (narrow 24/48 + td=0.3 + wd=5e-4)"
echo "============================================================"
$PYTHON train_cvae.py --mode cvae --dataset mosei --datapath $DATA \
  --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS \
  --kl_weight 0.1 --recon_weight 1.0 \
  --cvae_latent 24 --cvae_hidden 48 \
  --text_dropout 0.3 --weight_decay 5e-4 \
  --name checkpoints/mosei_cvae_fix3_combined.pt \
  --log_interval 500 2>&1 | tee /tmp/fix3_combined.txt

echo ""
echo "All fix experiments done."
