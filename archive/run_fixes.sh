#!/bin/bash
cd /home/ly/stu_work/projects/missing-modality-msa
echo "=== Fix 1: T-dropout ==="
/usr/bin/python3 train_cvae.py --mode cvae --dataset mosei --datapath casp_dataset/mosei.pkl --num_epochs 30 --batch_size 32 --num_workers 2 --kl_weight 0.1 --recon_weight 1.0 --text_dropout 0.3 --name checkpoints/mosei_cvae_fix1_tdrop.pt --log_interval 500 2>&1 | tee /tmp/fix1_tdrop.txt
echo ""
echo "=== Fix 2: Narrow CVAE ==="
/usr/bin/python3 train_cvae.py --mode cvae --dataset mosei --datapath casp_dataset/mosei.pkl --num_epochs 30 --batch_size 32 --num_workers 2 --kl_weight 0.1 --recon_weight 1.0 --cvae_latent 24 --cvae_hidden 48 --name checkpoints/mosei_cvae_fix2_narrow.pt --log_interval 500 2>&1 | tee /tmp/fix2_narrow.txt
echo ""
echo "=== Fix 3: Combined ==="
/usr/bin/python3 train_cvae.py --mode cvae --dataset mosei --datapath casp_dataset/mosei.pkl --num_epochs 30 --batch_size 32 --num_workers 2 --kl_weight 0.1 --recon_weight 1.0 --cvae_latent 24 --cvae_hidden 48 --text_dropout 0.3 --weight_decay 5e-4 --name checkpoints/mosei_cvae_fix3_combined.pt --log_interval 500 2>&1 | tee /tmp/fix3_combined.txt
echo "All done."
