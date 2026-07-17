#!/bin/bash
# MOSI dataset experiments — 4 key configs
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

DATA="casp_dataset/mosi.pkl"
EPOCHS=30; BS=32; SEED=666; N_PARALLEL=4
LOG_DIR="/tmp/mosi_test"
mkdir -p "$LOG_DIR" "checkpoints/mosi"

echo "============================================================"
echo "MOSI Test: concat + CVAE best configs (4 runs)"
echo "============================================================"
echo "Started at: $(date)"

# Build commands directly for simplicity
declare -a CMDS

# 1. concat baseline
CMDS[0]="echo '[1/4] MOSI concat baseline' && \
  PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py \
    --mode concat --dataset mosi --datapath $DATA \
    --num_epochs 30 --batch_size $BS --num_workers 0 \
    --lr 0.001 --seed $SEED --log_interval 50 \
    --dropout_prob 0.2 \
    --name checkpoints/mosi/mosi_concat.pt \
    > ${LOG_DIR}/mosi_concat.txt 2>&1 && \
  /usr/bin/python3 record_results.py ${LOG_DIR}/mosi_concat.txt mosi_concat \
    --config=dataset=mosi --config=mode=concat --config=lr=0.001 --config=dropout=0.2"

# 2. KL=0.4 pure CVAE baseline
CMDS[1]="echo '[2/4] MOSI CVAE KL=0.4' && \
  PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py \
    --mode cvae --dataset mosi --datapath $DATA \
    --num_epochs 30 --batch_size $BS --num_workers 0 \
    --kl_weight 0.4 --recon_weight 1.0 --dropout_prob 0.2 --lr 0.001 \
    --seed $SEED --log_interval 50 \
    --name checkpoints/mosi/mosi_kl0.4.pt \
    > ${LOG_DIR}/mosi_kl0.4.txt 2>&1 && \
  /usr/bin/python3 record_results.py ${LOG_DIR}/mosi_kl0.4.txt mosi_kl0.4 \
    --config=dataset=mosi --config=mode=cvae --config=kl=0.4 --config=rw=1.0 --config=lr=0.001"

# 3. KL=0.4 + Contrastive cw=0.7 (best combo)
CMDS[2]="echo '[3/4] MOSI CVAE KL=0.4 + CT0.7' && \
  PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py \
    --mode cvae --dataset mosi --datapath $DATA \
    --num_epochs 30 --batch_size $BS --num_workers 0 \
    --kl_weight 0.4 --recon_weight 1.0 --dropout_prob 0.2 --lr 0.001 \
    --contrastive_weight 0.7 --seed $SEED --log_interval 50 \
    --name checkpoints/mosi/mosi_kl0.4_ct0.7.pt \
    > ${LOG_DIR}/mosi_kl0.4_ct0.7.txt 2>&1 && \
  /usr/bin/python3 record_results.py ${LOG_DIR}/mosi_kl0.4_ct0.7.txt mosi_kl0.4_ct0.7 \
    --config=dataset=mosi --config=mode=cvae --config=kl=0.4 --config=rw=1.0 --config=cw=0.7 --config=lr=0.001"

# 4. KL=0.8 pure CVAE (high-KL comparison)
CMDS[3]="echo '[4/4] MOSI CVAE KL=0.8' && \
  PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py \
    --mode cvae --dataset mosi --datapath $DATA \
    --num_epochs 30 --batch_size $BS --num_workers 0 \
    --kl_weight 0.8 --recon_weight 1.0 --dropout_prob 0.2 --lr 0.001 \
    --seed $SEED --log_interval 50 \
    --name checkpoints/mosi/mosi_kl0.8.pt \
    > ${LOG_DIR}/mosi_kl0.8.txt 2>&1 && \
  /usr/bin/python3 record_results.py ${LOG_DIR}/mosi_kl0.8.txt mosi_kl0.8 \
    --config=dataset=mosi --config=mode=cvae --config=kl=0.8 --config=rw=1.0 --config=lr=0.001"

TOTAL=${#CMDS[@]}
echo "Total: $TOTAL jobs"

for i in "${!CMDS[@]}"; do
  eval "${CMDS[$i]}" &
done
wait

echo ""
echo "============================================================"
echo "MOSI ALL DONE at $(date)"
echo "============================================================"
