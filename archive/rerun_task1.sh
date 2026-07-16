#!/bin/bash
# Re-run Task 1 (random-z CVAE) with reduced num_workers to avoid OOM
# Original crashed during epoch 3 due to OOM killer (4 workers + Task 2)

PYTHON=python
PYTHON train_cvae.py \
  --mode cvae \
  --dataset mosei \
  --datapath casp_dataset/mosei.pkl \
  --num_epochs 30 \
  --batch_size 32 \
  --num_workers 2 \
  --lr 2e-3 \
  --kl_weight 0.1 \
  --recon_weight 1.0 \
  --mc_samples 5 \
  --name checkpoints/mosei_cvae_mc_v2.pt \
  --log_interval 200 2>&1 | tee /tmp/task1_mc_v2_retry.txt
echo "Training complete. Recording results..."
python3 record_results.py /tmp/task1_mc_v2_retry.txt "mosei_cvae_mc_v2" \
  --config "mode=cvae,batch_size=32,lr=2e-3,kl_weight=0.1,recon_weight=1.0,mc_samples=5,num_workers=2"
echo "Done."
